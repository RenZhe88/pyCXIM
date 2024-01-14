# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 16:11:59 2024

@author: Lenovo
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares

from ..Common.Information_file_generator import InformationFileIO
from ..scan_reader.BSRF.pilatus_reader import BSRFPilatusImporter
from ..scan_reader.Desy.eiger_reader import DesyEigerImporter
from .RC2RSM_6C import cal_abs_q_pos

class calibration(object):
    def __init__(self, pathsave, filename="calibration.txt"):
        self.section_ar = ['General Information', 'Detector calibration', 'Crystal_information', 'Calibration scan %s_%d', 'Calculated UB matrix']
        pathinfor = os.path.join(pathsave, filename)
        self.infor = InformationFileIO(pathinfor)
        self.infor.infor_reader()
        return

    def init_beamtime(self, beamline, path, detector, pathmask):
        assert beamline in ['p10', '1w1a'], 'Now the code has only been implemented for the six circle diffractometer at P10 beamline, Desy and 1w1a beamline, BSRF! For other beamlines, please contact the author! renzhe@ihep.ac.cn'
        if beamline == 'p10':
            motor_names = ['om', 'del', 'chi', 'phi', 'gam', 'energy']
        elif beamline == '1w1a':
            motor_names = ['eta', 'del', 'chi', 'phi', 'nu', 'energy']

        self.infor.add_para('beamline', self.section_ar[0], beamline)
        self.infor.add_para('path', self.section_ar[0], path)
        self.infor.add_para('detector', self.section_ar[0], detector)
        self.infor.add_para('pathmask', self.section_ar[0], pathmask)
        self.infor.add_para('motor_names', self.section_ar[0], motor_names)
        self.infor.infor_writer()
        return

    def load_parameters(self, parameter_names, section=''):
        paralist = []
        for para_name in parameter_names:
            paralist.append(self.infor.get_para_value(para_name, section=section))
        return paralist

    def detector_calib(self, file_1w1a, scan_num):
        beamline, path, detector, pathmask = self.load_parameters(['beamline', 'path', 'detector', 'pathmask'], section=self.section_ar[0])

        if beamline == 'p10':
            scan = DesyEigerImporter(beamline, path, file_1w1a, scan_num, detector, pathmask=pathmask, creat_save_folder=False)
            print(scan)
            X_pos, Y_pos, int_ar = scan.eiger_peak_pos_per_frame()
        elif beamline == '1w1a':
            scan = BSRFPilatusImporter(beamline, path, file_1w1a, scan_num, detector, pathmask=pathmask, creat_save_folder=False)
            print(scan)
            X_pos, Y_pos, int_ar = scan.pilatus_peak_pos_per_frame()
        motor = scan.get_scan_motor()
        angle_ar = scan.get_scan_data(motor)
        energy = scan.get_motor_pos('energy')
        pixelsize = scan.get_detector_pixelsize()

        cch = np.zeros(2, dtype=int)
        angle_ar = angle_ar[int_ar > np.amax(int_ar) * 0.5]
        X_pos = X_pos[int_ar > np.amax(int_ar) * 0.5]
        Y_pos = Y_pos[int_ar > np.amax(int_ar) * 0.5]
        fit_x = np.polyfit(np.radians(angle_ar), X_pos, 1)
        fit_y = np.polyfit(np.radians(angle_ar), Y_pos, 1)
        distance = np.sqrt(fit_x[0]**2.0 + fit_y[0]**2.0) * pixelsize
        cch[0] = np.round(fit_y[1])
        cch[1] = np.round(fit_x[1])
        det_rot = -np.degrees(fit_x[0] / fit_y[0])

        print('fitted direct beam position: ' + str(cch))
        print('detector distance:           %f' % distance)
        print('detector rotation angle:     %f' % (-np.degrees(fit_x[0] / fit_y[0])))

        plt.plot(angle_ar, X_pos, 'r+', label='X position')
        plt.plot(angle_ar, np.poly1d(fit_x)(np.radians(angle_ar)), 'r-', label='X fit')
        plt.plot(angle_ar, Y_pos, 'b+', label='Y position')
        plt.plot(angle_ar, np.poly1d(fit_y)(np.radians(angle_ar)), 'b-', label='Y fit')
        plt.xlabel("angle (degree)")
        plt.ylabel('Beam position (pixel)')
        plt.legend()
        plt.show()

        self.infor.add_para('energy', self.section_ar[0], energy)
        self.infor.del_para_section(self.section_ar[1])
        self.infor.add_para('1W1A_newfile', self.section_ar[1], file_1w1a)
        self.infor.add_para('scan_number', self.section_ar[1], scan_num)
        self.infor.add_para('direct_beam_position', self.section_ar[1], list(cch))
        self.infor.add_para('detector_distance', self.section_ar[1], distance)
        self.infor.add_para('pixelsize', self.section_ar[1], pixelsize)
        self.infor.add_para('detector_rotation', self.section_ar[1], det_rot)
        self.infor.infor_writer()
        return

    def B_matrix_cal(self, lattice_constants):
        a, b, c = lattice_constants[:3]
        alpha, beta, gamma = np.deg2rad(lattice_constants[3:])

        f = np.sqrt(1 - np.square(np.cos(alpha)) - np.square(np.cos(beta)) - np.square(np.cos(gamma)) + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma))
        A_matrix = np.array([[a, b * np.cos(gamma), c * np.cos(beta)],
                             [0, b * np.sin(gamma), c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)],
                             [0, 0, c * f / np.sin(gamma)]])
        B_matrix = 2.0 * np.pi * np.linalg.inv(A_matrix).T
        self.infor.add_para('lattice_constants', self.section_ar[2], list(lattice_constants))
        self.infor.add_para('B_matrix', self.section_ar[2], B_matrix.tolist())
        return

    def expected_U_matrix_cal(self, surface_dir, inplane_dir):
        B_matrix = np.array(self.infor.get_para_value('B_matrix', self.section_ar[2]), dtype=float)
        surface_dir = np.dot(B_matrix, surface_dir)
        surface_dir = surface_dir / np.linalg.norm(surface_dir)
        inplane_dir1 = np.dot(B_matrix, inplane_dir)
        inplane_dir1 = inplane_dir1 - np.dot(inplane_dir1, surface_dir) * surface_dir
        inplane_dir1 = inplane_dir1 / np.linalg.norm(inplane_dir1)
        inplane_dir2 = np.cross(surface_dir, inplane_dir1)
        inplane_dir2 = inplane_dir2 / np.linalg.norm(inplane_dir2)

        inv_U = np.array([inplane_dir1, inplane_dir2, surface_dir]).T
        expected_U_matrix = np.linalg.inv(inv_U)
        self.infor.add_para('surface_direction', self.section_ar[2], list(surface_dir))
        self.infor.add_para('inplane_direction', self.section_ar[2], list(inplane_dir1))
        self.infor.add_para('expected_U_matrix', self.section_ar[2], expected_U_matrix.tolist())
        return

    def crystal_information(self, lattice_constants, surface_dir, inplane_dir):
        self.infor.del_para_section(self.section_ar[2])
        self.B_matrix_cal(lattice_constants)
        self.expected_U_matrix_cal(surface_dir, inplane_dir)
        self.infor.infor_writer()
        return

    def cal_possible_Bragg_angles(self, surface_dir, hkl_range=(6, 6, 6)):
        B_matrix = np.array(self.infor.get_para_value('B_matrix', self.section_ar[2]), dtype=float)
        q_surf = np.dot(B_matrix, surface_dir)

        energy = self.infor.get_para_value('energy', self.section_ar[0])
        if energy is not None:
            hc = 1.23984 * 10000.0
            wavelength = hc / energy
        else:
            raise RuntimeError('Could not find the energy in the information file, please perform the detector calibration first!')

        for h_index in range(hkl_range[0]):
            for k_index in range(hkl_range[1]):
                for l_index in range(hkl_range[2]):
                    if h_index * h_index + k_index * k_index + l_index * l_index != 0:
                        mill_index = np.array([h_index, k_index, l_index])
                        q = np.dot(B_matrix, mill_index)
                        Bragg_angle = np.rad2deg(np.arcsin(np.linalg.norm(q) * wavelength / 4 / np.pi))
                        angle2surf = np.rad2deg(np.arccos(np.dot(q, q_surf) / np.linalg.norm(q) / np.linalg.norm(q_surf)))
                        if not np.isnan(Bragg_angle):
                            print(str(mill_index) + ',    ' + str(angle2surf) + ',    ' + str(Bragg_angle))
        return

    def load_peak_infor(self, sample_name, scan_num):
        beamline, path, detector, pathmask = self.load_parameters(['beamline', 'path', 'detector', 'pathmask'], section=self.section_ar[0])
        detector_para = self.load_parameters(['detector_distance', 'pixelsize', 'detector_rotation', 'direct_beam_position'], section=self.section_ar[1])

        if beamline == 'p10':
            scan = DesyEigerImporter(beamline, path, sample_name, scan_num, detector, pathmask=pathmask, creat_save_folder=False)
            print(scan)
            pixel_position, motor_position = scan.load_6C_peak_infor()
        elif beamline == '1w1a':
            scan = BSRFPilatusImporter(beamline, path, sample_name, scan_num, detector, pathmask=pathmask, creat_save_folder=False)
            pixel_position, motor_position = scan.load_6C_peak_infor()

        q_vector = cal_abs_q_pos(pixel_position, motor_position, detector_para)
        self.infor.add_para('1W1A_newfile', self.section_ar[3] % (sample_name, scan_num), sample_name)
        self.infor.add_para('scan_number', self.section_ar[3] % (sample_name, scan_num), scan_num)
        self.infor.add_para('motor_position', self.section_ar[3] % (sample_name, scan_num), motor_position.tolist())
        self.infor.add_para('pixel_position', self.section_ar[3] % (sample_name, scan_num), pixel_position.tolist())
        self.infor.add_para('q_vector', self.section_ar[3] % (sample_name, scan_num), q_vector.tolist())
        self.infor.infor_writer()
        return pixel_position, motor_position, detector_para

    def q_error_single_peak(self, offsets, pixel_position, motor_position, detector_para,
                            para_selected, expected_q, full_offset_values):
        assert sum(para_selected) == 3, 'For single peak only three parameter error could be fitted!'
        assert len(offsets) == 3, 'For single peak only three parameter error could be fitted!'

        full_offset_values[para_selected] = np.array(offsets, dtype=float)
        motor_position = motor_position + full_offset_values
        q_vector = cal_abs_q_pos(pixel_position, motor_position, detector_para)
        error_ar = q_vector - expected_q
        return error_ar

    def single_Bragg_peak_tilt_cal(self, sample_name, scan_num, peak_index, error_source,
                                   known_error_values=np.zeros(6)):
        self.infor.del_para_section(self.section_ar[3] % (sample_name, scan_num))
        pixel_position, motor_position, detector_para = self.load_peak_infor(sample_name, scan_num)

        paranames = self.infor.get_para_value('motor_names', section=self.section_ar[0])
        para_selected = []
        for element in enumerate(paranames):
            if element in error_source:
                para_selected.append(1)
            else:
                para_selected.append(0)

        assert sum(para_selected) == 3, 'For single peak only three parameter error could be fitted! Please select from %s' % str(paranames)
        para_selected = np.array(para_selected, dtype='bool')

        peak_index = np.array(peak_index, dtype=float)
        B_matrix = np.array(self.infor.get_para_value('B_matrix', self.section_ar[2]), dtype=float)
        expected_U_matrix = np.array(self.infor.get_para_value('expected_U_matrix', self.section_ar[2]), dtype=float)
        expected_q = np.dot(expected_U_matrix, np.dot(B_matrix, peak_index))
        # change the q vector from [qx, qy, qz] to [qz, qy, qx]
        expected_q = np.flip(expected_q)

        leastsq_solution = least_squares(self.q_error_single_peak, [0.1, 0.1, 0.1], args=(pixel_position, motor_position, detector_para, para_selected, expected_q, known_error_values))

        full_offsets_ar = known_error_values
        full_offsets_ar[para_selected] = leastsq_solution.x
        self.infor.add_para('peak_index', self.section_ar[3] % (sample_name, scan_num), list(peak_index))
        self.infor.add_para('error_source', self.section_ar[3] % (sample_name, scan_num), error_source)
        self.infor.add_para('offsets_parameters', self.section_ar[3] % (sample_name, scan_num), list(full_offsets_ar))
        for i, paraname in enumerate(paranames):
            if paraname in error_source:
                print('%s_offset=%.2f' % (paraname, known_error_values[i]))
        self.infor.infor_writer()
        return

    def q_error_multiple_peak(self, euler_angles, hkl_ar, q_ar):
        hkl_ar = np.array(hkl_ar, dtype=float)
        B_matrix = np.array(self.infor.get_para_value('B_matrix', self.section_ar[2]), dtype=float)
        rotation_matrix = R.from_euler('yxz', euler_angles, degrees=True)
        U_matrix = rotation_matrix.as_matrix()
        error_ar = np.array([])
        for i in np.arange(hkl_ar.shape[0]):
            error = np.flip(np.dot(U_matrix, np.dot(B_matrix, hkl_ar[i, :]))) - q_ar[i, :]
            error_ar = np.append(error_ar, error)
        return error_ar

    def multi_Bragg_peak_U_matrix_cal(self, sample_name_ar, scan_num_ar, peak_index_ar):
        q_ar = np.zeros_like(peak_index_ar, dtype=float)
        for i, scan_num in enumerate(scan_num_ar):
            if (type(sample_name_ar) == list) and (len(scan_num_ar) != len(sample_name_ar)):
                sample_name = sample_name_ar[i]
            elif (type(sample_name_ar) == str):
                sample_name = sample_name_ar

            if not self.infor.section_exists(self.section_ar[3] % (sample_name, scan_num)):
                self.load_peak_infor(sample_name, scan_num)
            q_ar[i, :] = np.array(self.infor.get_para_value('q_vector', self.section_ar[3] % (sample_name, scan_num)), dtype=float)

        leastsq_solution = least_squares(self.q_error_multiple_peak, np.array([10.0, 10.0, 10.0]), args=(peak_index_ar, q_ar))
        print('Find UB matrix?')
        print(leastsq_solution.success)

        rotation_matrix = R.from_euler('yxz', leastsq_solution.x, degrees=True)
        U_matrix = rotation_matrix.as_matrix()
        print('measured UB matrix')
        print(np.around(U_matrix, 2))

        expected_U_matrix = self.infor.get_para_value('expected_U_matrix', self.section_ar[2])
        print('expected_UB')
        print(np.around(expected_U_matrix, 2))
        additional_rotation_matrix = np.dot(U_matrix, np.linalg.inv(expected_U_matrix))
        additional_rotation = R.from_matrix(additional_rotation_matrix)
        print('Angular deviations')
        angular_deviations = additional_rotation.as_euler('yxz', degrees=True)
        print(angular_deviations)
        self.infor.del_para_section(self.section_ar[4])
        self.infor.add_para('1W1A_newfiles', self.section_ar[4], sample_name_ar)
        self.infor.add_para('scan_num_list', self.section_ar[4], scan_num_ar)
        self.infor.add_para('peak_index_list', self.section_ar[4], peak_index_ar)
        self.infor.add_para('find_U_matrix', self.section_ar[4], leastsq_solution.success)
        self.infor.add_para('U_matrix', self.section_ar[4], U_matrix.tolist())
        self.infor.add_para('additional_rotation_matrix', self.section_ar[4], additional_rotation_matrix.tolist())
        self.infor.add_para('angular_deviations', self.section_ar[4], angular_deviations.tolist())
        self.infor.infor_writer()
        return

    def hkl_to_angles(self, aimed_hkl, rotation_axis, given_motor_values, limitations=None):

        aimed_hkl = np.array(aimed_hkl, dtype=float)
        motor_names = self.infor.get_para_value('motor_names', self.section_ar[0])
        B_matrix = np.array(self.infor.get_para_value('B_matrix', self.section_ar[2]), dtype=float)
        U_matrix = np.array(self.infor.get_para_value('U_matrix', self.section_ar[4]), dtype=float)

        expected_q = np.flip(np.dot(U_matrix, np.dot(B_matrix, aimed_hkl)))
        position_parameters = np.zeros(6)
        para_selected = []
        for i, element in enumerate(motor_names):
            if element in rotation_axis:
                para_selected.append(1)
            else:
                para_selected.append(0)
        para_selected = np.array(para_selected, dtype='bool')
        if limitations is None:
            limitations = ([-180, -180, -180], [180, 180, 180])
        offsets = least_squares(self.q_error_single_peak, [10.0, 10.0, -10.0], bounds=limitations, args=([0.0, 0.0], position_parameters, para_selected, expected_q, given_motor_values))
        given_motor_values[para_selected] = offsets.x
        for i in range(6):
            print('%s = %.3f' % (motor_names[i], given_motor_values[i]))
        return
