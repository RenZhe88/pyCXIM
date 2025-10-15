# -*- coding: utf-8 -*-
"""
The typical calibration process for the six circle diffractometer.

Created on Thu Jan 11 16:11:59 2024

@author: Ren Zhe
@email: renzhe@ihep.ac.cn
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares

from ..Common.Information_file_generator import InformationFileIO
from ..scan_reader.BSRF.pilatus_reader import BSRFPilatusImporter
from ..scan_reader.Desy.eiger_reader import DesyEigerImporter
from ..scan_reader.ESRF.h5_reader import ESRFH5Importer
from .RC2RSM_6C import cal_q_pos


class Calibration(object):
    """
    Record the calibration process of a sample measured with six cirlce diffractometer.

    The calibration contains four types of information.
    1. The detector informaiton, including detector distance, pixel size, detector rotation, and direct beam position.
    2. The crystal information, which is defined by the lattice parameters and the expected crystal orientaion.
    3. The measured crystal orientation, which is determined by the two Bragg peaks.

    Now the code is implenmented for p10, p08 and 1w1a.
    If you have a six circle diffractometer at your beamline and want to use this calibration process.
    Please contact the author renzhe@ihep.ac.cn, renzhetu001@gmail.com

    Parameters
    ----------
    pathsave : str
        The folder to save the calibration results.
    filename : str, optional
        The filename to save the calibration results. The default is "calibration.txt".

    Returns
    -------
    None.

    """

    def __init__(self, pathsave, filename="calibration.txt"):
        self.section_ar = ['General Information', 'Detector calibration', 'Crystal_information', 'Calibration scan %s_%d', 'Calculated UB matrix']
        pathinfor = os.path.join(pathsave, filename)
        self.infor = InformationFileIO(pathinfor)
        self.infor.infor_reader()
        return

    def init_beamtime(self, beamline, path, detector, **kwargs):
        """
        Define the basic information of the beamtime.

        Parameters
        ----------
        beamline : str
            The name of the beamline. For now, please select between 'p10', 'p08', '1w1a' and 'id01'.
        path : str
            The path of the beamtime folder.
        detector : str
            The detector used.
        **kwargs: Additional information provided as keyword arguments
        Keyword Arguments:
        pathmask : str
            The path of the detector mask. Required for DESY, BSRF beamlines.
        beamtimeID : str, optional
            The beamtime ID. Required for ESRF beamlines.
        experimental_method : str, optional
            The experimental method. Required for ESRF beamlines.

        Raises
        ------
        KeyError
            For beamline other than 'p10' and '1w1a', raise this error.

        Returns
        -------
        None.

        """
        if beamline == 'p10':
            motor_names = ['om', 'del', 'chi', 'phi', 'gam', 'mu', 'fmbenergy']
        elif beamline == 'p08':
            motor_names = ['om', 'tt', 'chi', 'phis', 'tth', 'omh', 'energyfmb']
        elif beamline == '1w1a':
            motor_names = ['eta', 'del', 'chi', 'phi', 'nu', 'mu', 'energy']
        elif beamline == 'id01':
            motor_names = ['eta', 'delta', 'chi', 'phi', 'nu', 'mu', 'nrj']
        else:
            raise KeyError('Now the code has only been implemented for the six circle diffractometer at P10 beamline, Desy and 1w1a beamline, BSRF! For other beamlines, please contact the author! renzhe@ihep.ac.cn')

        self.infor.add_para('beamline', self.section_ar[0], beamline)
        self.infor.add_para('path', self.section_ar[0], path)
        self.infor.add_para('detector', self.section_ar[0], detector)
        self.infor.add_para('motor_names', self.section_ar[0], motor_names)
        if beamline in ['p10', 'p08', '1w1a']:
            if 'pathmask' in kwargs.keys():
                self.infor.add_para('pathmask', self.section_ar[0], kwargs['pathmask'])
            else:
                raise ValueError('The path of the mask file is missing!')
        elif beamline in ['id01']:
            if 'beamtimeID' in kwargs.keys() and 'experimental_method' in kwargs.keys():
                self.infor.add_para('beamtimeID', self.section_ar[0], kwargs['beamtimeID'])
                self.infor.add_para('experimental_method', self.section_ar[0], kwargs['experimental_method'])
            else:
                raise ValueError('The path of the mask file is missing!')
        self.infor.infor_writer()
        return

    def init_scanner(self, sample_name, scan_num):
        """
        Initiate scanner. Sample name and scan number will be needed, other parameters should be given in the initiate beamtime.
        If new beamline is added, this part should be changed.

        Parameters
        ----------
        sample_name : str
            The name of the specimen.
        scan_num : int
            the scan number for the data treatment.

        Returns
        -------
        scan : object
            The scan method.

        """
        beamline, path, detector = self.load_parameters(['beamline', 'path', 'detector'], section=self.section_ar[0])
        if beamline == 'p10':
            pathmask = self.infor.get_para_value('pathmask', section=self.section_ar[0])
            scan = DesyEigerImporter(beamline, path, sample_name, scan_num, detector, pathmask=pathmask, creat_save_folder=False)
            energy = scan.get_motor_pos('fmbenergy')
        elif beamline == 'p08':
            pathmask = self.infor.get_para_value('pathmask', section=self.section_ar[0])
            scan = DesyEigerImporter(beamline, path, sample_name, scan_num, detector, pathmask=pathmask, creat_save_folder=False)
            energy = scan.get_motor_pos('energyfmb')
        elif beamline == '1w1a':
            pathmask = self.infor.get_para_value('pathmask', section=self.section_ar[0])
            scan = BSRFPilatusImporter(beamline, path, sample_name, scan_num, detector, pathmask=pathmask, creat_save_folder=False)
            energy = scan.get_motor_pos('energy')
        elif beamline == 'id01':
            beamtimeID = self.infor.get_para_value('beamtimeID', section=self.section_ar[0])
            experimental_method = self.infor.get_para_value('experimental_method', section=self.section_ar[0])
            scan = ESRFH5Importer(beamline, path, beamtimeID, sample_name, experimental_method, scan_num, detector, creat_save_folder=False)
            energy = scan.get_motor_pos('nrj')
        self.infor.add_para('energy', self.section_ar[0], energy)
        return scan

    def load_parameters(self, parameter_names, section=''):
        """
        Load the desired parameter values from the calibration information file.

        Parameters
        ----------
        parameter_names : list
            The list of the parameter names.
        section : str, optional
            The name of the section. The default is ''.

        Returns
        -------
        paralist : list
            The corresponding list of the parameter values.

        """
        paralist = []
        for para_name in parameter_names:
            paralist.append(self.infor.get_para_value(para_name, section=section))
        return paralist

    def section_exists(self, section_name):
        """
        Check if the section exists in the calibration information file.

        Parameters
        ----------
        section_name : str
            The name of the section.

        Returns
        -------
        bool
            Whether the section exists in the information file.

        """
        return self.infor.section_exists(section_name)

    def detector_calib(self, sample_name, scan_num):
        """
        Read the detector calibration scans and the calculate the detector parameteres.

        By moving delta arm, the detector calibration scan measures direct beam position with respect to different two theta angles.

        Parameters
        ----------
        sample_name : str
            The name of the specimen, which is defined by the p10_newfile name or spec_newfile name.
        scan_num : int
            The scan number.

        Returns
        -------
        None.

        """
        scan = self.init_scanner(sample_name, scan_num)

        X_pos, Y_pos, int_ar = scan.image_peak_pos_per_frame()
        print(scan)
        motor = scan.get_scan_motor()
        angle_ar = scan.get_scan_data(motor)
        pixelsize = scan.get_detector_pixelsize()

        cch = np.zeros(2, dtype=int)
        angle_ar = angle_ar[int_ar > np.amax(int_ar) * 0.5]
        X_pos = X_pos[int_ar > np.amax(int_ar) * 0.5]
        Y_pos = Y_pos[int_ar > np.amax(int_ar) * 0.5]
        fit_x = np.polyfit(np.tan(np.deg2rad(angle_ar)), X_pos, 1)
        fit_y = np.polyfit(np.tan(np.deg2rad(angle_ar)), Y_pos, 1)
        distance = np.sqrt(fit_x[0]**2.0 + fit_y[0]**2.0) * pixelsize
        cch[0] = np.round(fit_y[1])
        cch[1] = np.round(fit_x[1])
        det_rot = -np.degrees(fit_x[0] / fit_y[0])

        print('fitted direct beam position: ' + str(cch))
        print('detector distance:           %f' % distance)
        print('detector rotation angle:     %f' % (-np.degrees(fit_x[0] / fit_y[0])))

        plt.plot(angle_ar, X_pos, 'r+', label='X position')
        plt.plot(angle_ar, np.poly1d(fit_x)(np.tan(np.deg2rad(angle_ar))), 'r-', label='X fit')
        plt.plot(angle_ar, Y_pos, 'b+', label='Y position')
        plt.plot(angle_ar, np.poly1d(fit_y)(np.tan(np.deg2rad(angle_ar))), 'b-', label='Y fit')
        plt.xlabel("angle (degree)")
        plt.ylabel('Beam position (pixel)')
        plt.legend()
        plt.show()

        self.infor.del_para_section(self.section_ar[1])
        self.infor.add_para('sample_name', self.section_ar[1], sample_name)
        self.infor.add_para('scan_number', self.section_ar[1], scan_num)
        self.infor.add_para('direct_beam_position', self.section_ar[1], cch)
        self.infor.add_para('detector_distance', self.section_ar[1], distance)
        self.infor.add_para('pixelsize', self.section_ar[1], pixelsize)
        self.infor.add_para('detector_rotation', self.section_ar[1], det_rot)
        self.infor.infor_writer()
        return

    def B_matrix_cal(self, lattice_constants):
        """
        Calculate the B matrix according to the given lattice parameter.

        Parameters
        ----------
        lattice_constants : list
            The lattice constant writen in the form of [a, b, c, alpha, beta, gamma].

        Returns
        -------
        None.

        """
        a, b, c = lattice_constants[:3]
        alpha, beta, gamma = np.deg2rad(lattice_constants[3:])

        f = np.sqrt(1 - np.square(np.cos(alpha)) - np.square(np.cos(beta)) - np.square(np.cos(gamma)) + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma))
        A_matrix = np.array([[a, b * np.cos(gamma), c * np.cos(beta)],
                             [0, b * np.sin(gamma), c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)],
                             [0, 0, c * f / np.sin(gamma)]])
        B_matrix = 2.0 * np.pi * np.linalg.inv(A_matrix).T
        self.infor.add_para('lattice_constants', self.section_ar[2], lattice_constants)
        self.infor.add_para('B_matrix', self.section_ar[2], B_matrix)
        return

    def expected_U_matrix_cal(self, surface_hkl, inplane_hkl):
        """
        Calculate the expected U matrix from the given surface direction and inplane direction.

        Parameters
        ----------
        surface_hkl : list
            The mill index of the specimen surface direction.
        inplane_hkl : list
            The inplane direction along the X-ray beam.

        Returns
        -------
        None.

        """
        B_matrix = np.array(self.infor.get_para_value('B_matrix', self.section_ar[2]), dtype=float)
        surface_dir = np.dot(B_matrix, surface_hkl)
        surface_dir = surface_dir / np.linalg.norm(surface_dir)
        inplane_dir1 = np.dot(B_matrix, inplane_hkl)
        inplane_dir1 = inplane_dir1 - np.dot(inplane_dir1, surface_dir) * surface_dir
        inplane_dir1 = inplane_dir1 / np.linalg.norm(inplane_dir1)
        inplane_dir2 = np.cross(surface_dir, inplane_dir1)
        inplane_dir2 = inplane_dir2 / np.linalg.norm(inplane_dir2)

        inv_U = np.array([inplane_dir1, inplane_dir2, surface_dir]).T
        expected_U_matrix = np.linalg.inv(inv_U)
        self.infor.add_para('surface_direction', self.section_ar[2], surface_hkl)
        self.infor.add_para('inplane_direction', self.section_ar[2], inplane_hkl)
        self.infor.add_para('expected_U_matrix', self.section_ar[2], expected_U_matrix)
        return

    def crystal_information(self, lattice_constants, surface_dir, inplane_dir):
        """
        Generate the crystal information section in the calibration file.

        Parameters
        ----------
        lattice_constants : list
            The lattice constant writen in the form of [a, b, c, alpha, beta, gamma].
        surface_dir : list
            The mill index of the specimen surface direction.
        inplane_dir : list
            The inplane direction along the X-ray beam.

        Returns
        -------
        None.

        """
        self.infor.del_para_section(self.section_ar[2])
        self.B_matrix_cal(lattice_constants)
        self.expected_U_matrix_cal(surface_dir, inplane_dir)
        self.infor.infor_writer()
        return

    def cal_possible_Bragg_angles(self, hkl_range=[6, 6, 6]):
        """
        Calculate possible angles for the Bragg peak positions.

        Important: The extinguish rules for the lattice structure is not considered in this case.

        Parameters
        ----------
        hkl_range : list, optional
            The maximum h k l indexes, where the diffraction peaks should be searched. The default is [6, 6, 6].

        Raises
        ------
        RuntimeError
            If the detector calibration section is not performed, then the energy parameter will be missing in the information file. In this case, raise RuntimeError.

        Returns
        -------
        None.

        """
        B_matrix = np.array(self.infor.get_para_value('B_matrix', self.section_ar[2]), dtype=float)
        surface_dir = np.array(self.infor.get_para_value('surface_direction', self.section_ar[2]), dtype=float)
        q_surf = np.dot(B_matrix, surface_dir)

        energy = self.infor.get_para_value('energy', self.section_ar[0])
        if energy is not None:
            hc = 1.23984 * 10000.0
            wavelength = hc / energy
        else:
            raise RuntimeError('Could not find the energy in the information file, please perform the detector calibration first!')

        print('mill_index,    angle2surf,    Bragg angle')
        for h_index in range(hkl_range[0]):
            for k_index in range(hkl_range[1]):
                for l_index in range(hkl_range[2]):
                    if h_index * h_index + k_index * k_index + l_index * l_index != 0:
                        mill_index = np.array([h_index, k_index, l_index])
                        q = np.dot(B_matrix, mill_index)
                        if (np.linalg.norm(q) * wavelength / 4 / np.pi) < 1.0:
                            Bragg_angle = np.rad2deg(np.arcsin(np.linalg.norm(q) * wavelength / 4 / np.pi))
                            angle2surf = np.rad2deg(np.arccos(np.dot(q, q_surf) / (np.linalg.norm(q) + np.finfo(np.float64).eps) / (np.linalg.norm(q_surf) + np.finfo(np.float64).eps)))
                            print('%s,        %7.2f,        %7.2f' % (str(mill_index), angle2surf, Bragg_angle))
        return

    def load_peak_infor(self, sample_name, scan_num):
        """
        Load the basic information of a scan from the substrate diffraction peak.

        Parameters
        ----------
        sample_name : str
            The name of the specimen, which is defined by the p10_newfile name or spec_newfile name.
        scan_num : int
            The scan number.

        Returns
        -------
        pixel_position : list
            The peak position on the detector in the form of [Y, X].
        motor_position : list
            The motor position of the diffractometer in the order of ['om', 'del', 'chi', 'phi', 'gam', 'mu', 'energy'] or ['eta', 'del', 'chi', 'phi', 'nu', 'mu', 'energy'].
        detector_para : list
            The detector parameter from the calibration scan in the order of ['detector_distance', 'pixelsize', 'detector_rotation', 'direct_beam_position'].

        """
        beamline, path, detector, pathmask = self.load_parameters(['beamline', 'path', 'detector', 'pathmask'], section=self.section_ar[0])
        detector_para = self.load_parameters(['detector_distance', 'pixelsize', 'detector_rotation', 'direct_beam_position'], section=self.section_ar[1])

        scan = self.init_scanner(sample_name, scan_num)
        pixel_position, motor_position = scan.load_6C_peak_infor()
        print(scan)

        q_vector = cal_q_pos(pixel_position, motor_position, detector_para)
        self.infor.add_para('sample_name', self.section_ar[3] % (sample_name, scan_num), sample_name)
        self.infor.add_para('scan_number', self.section_ar[3] % (sample_name, scan_num), scan_num)
        self.infor.add_para('motor_position', self.section_ar[3] % (sample_name, scan_num), motor_position)
        self.infor.add_para('pixel_position', self.section_ar[3] % (sample_name, scan_num), pixel_position)
        self.infor.add_para('q_vector', self.section_ar[3] % (sample_name, scan_num), q_vector)
        self.infor.infor_writer()
        return pixel_position, motor_position, detector_para

    def q_error_single_peak(self, offsets, pixel_position, motor_position, detector_para,
                            para_selected, expected_q, full_offset_values):
        """
        Calculate the q vector of diffraction peak based on certain angular offsets and compare it to the expected value.

        Parameters
        ----------
        offsets : list
            The offsets values of the measured angles for the diffraction peak.
        pixel_position : list
            The peak position on the detector in the form of [Y, X].
        motor_position : list
            The motor position of the diffractometer in the order of
            ['om', 'del', 'chi', 'phi', 'gam', 'energy']
            or ['eta', 'del', 'chi', 'phi', 'nu', 'energy'].
        detector_para : list
            The detector parameter from the calibration scan in the order of ['detector_distance', 'pixelsize', 'detector_rotation', 'direct_beam_position'].
        para_selected : list
            The list of bool values indicating which three motors should be considered with offset values.
        expected_q : list
            The expect q vector value in inverse angstroms in the order of [qz, qy, qx].
        full_offset_values : list
            The offset values of other motors, the motor order is ['om', 'del', 'chi', 'phi', 'gam', 'energy'].

        Returns
        -------
        error_ar : ndarray
            The difference between q vector calculated and q vector expected.

        """
        assert len(offsets) == 3, 'For single peak only three parameter error could be fitted!'

        full_offset_values[para_selected] = np.array(offsets, dtype=float)
        motor_position = motor_position + full_offset_values
        q_vector = cal_q_pos(pixel_position, motor_position, detector_para)
        error_ar = q_vector - expected_q
        return error_ar

    def single_Bragg_peak_tilt_cal(self, sample_name, scan_num, peak_index, error_source,
                                   known_error_values=np.zeros(7)):
        """
        Calculate offsets of certain motors according to a given Bragg peak.

        Parameters
        ----------
        sample_name : str
            The name of the specimen, which is defined by the p10_newfile name or spec_newfile name.
        scan_num : int
            The scan number.
        peak_index : list
            The index of the Bragg peak, e.g. [2, 2, 0].
        error_source : list
            Three motors where the offset should be considered. For symmetric diffraction peak, this usually is ['om', 'del', 'chi'], and for asymmetric diffraction peak, this usually is ['om', 'del', 'phi'].
        known_error_values : list, optional
            The known offsets values of the diffractometers in the order of ['om', 'del', 'chi', 'phi', 'gam', 'mu', 'energy']. The default is np.zeros(6).

        Returns
        -------
        None.

        """
        self.infor.del_para_section(self.section_ar[3] % (sample_name, scan_num))
        pixel_position, motor_position, detector_para = self.load_peak_infor(sample_name, scan_num)

        paranames = self.infor.get_para_value('motor_names', section=self.section_ar[0])
        para_selected = []
        for element in paranames:
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
        self.infor.add_para('peak_index', self.section_ar[3] % (sample_name, scan_num), peak_index)
        self.infor.add_para('error_source', self.section_ar[3] % (sample_name, scan_num), error_source)
        self.infor.add_para('offsets_parameters', self.section_ar[3] % (sample_name, scan_num), full_offsets_ar)
        for i, paraname in enumerate(paranames):
            if paraname in error_source:
                print('%s_offset=%.2f' % (paraname, known_error_values[i]))
        self.infor.infor_writer()
        return

    def q_error_multiple_peak(self, euler_angles, hkl_ar, q_ar):
        """
        Calculate the q vectors of diffraction peaks of a crystal orientation and compare it to the meausred ones.

        Parameters
        ----------
        euler_angles : list
            Three angles defining the .
        hkl_ar : list
            The mill index of the diffraction peaks, e.g. [[2, 2, 0], [3, 3, 2]].
        q_ar : list
            The measured q values of the diffraction peaks.

        Returns
        -------
        error_ar : ndarray
            The difference of the q position between the expected and meausred diffraction peaks.

        """
        hkl_ar = np.array(hkl_ar, dtype=float)
        B_matrix = np.array(self.infor.get_para_value('B_matrix', self.section_ar[2]), dtype=float)
        rotation_matrix = R.from_euler('yxz', euler_angles, degrees=True)
        U_matrix = rotation_matrix.as_matrix()
        error_ar = np.array([])
        for i in np.arange(hkl_ar.shape[0]):
            error = np.dot(U_matrix, np.dot(B_matrix, hkl_ar[i, :])) - np.flip(q_ar[i, :])
            error_ar = np.append(error_ar, error)
        return error_ar

    def multi_Bragg_peak_U_matrix_cal(self, sample_name_ar, scan_num_ar, peak_index_ar):
        """
        Calculate the U matrix based on at least two none-parallel Bragg peaks.

        Parameters
        ----------
        sample_name_ar : str or list
            The name of the specimen or specimens, which is defined by the p10_newfile name or spec_newfile name.
        scan_num_ar : list
            The list of the scan numbers.
        peak_index_ar : list
            The mill index of the diffraction peaks, e.g. [[2, 2, 0], [3, 3, 2]].

        Returns
        -------
        None.

        """
        q_ar = np.zeros_like(peak_index_ar, dtype=float)
        for i, scan_num in enumerate(scan_num_ar):
            if (type(sample_name_ar) == list) and (len(scan_num_ar) == len(sample_name_ar)):
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
        additional_rotation_matrix = np.flip(np.dot(expected_U_matrix, np.linalg.inv(U_matrix)))
        additional_rotation = R.from_matrix(additional_rotation_matrix)
        print('Angular deviations')
        angular_deviations = additional_rotation.as_euler('yxz', degrees=True)
        print(angular_deviations)
        self.infor.del_para_section(self.section_ar[4])
        self.infor.add_para('sample_names', self.section_ar[4], sample_name_ar)
        self.infor.add_para('scan_num_list', self.section_ar[4], scan_num_ar)
        self.infor.add_para('peak_index_list', self.section_ar[4], peak_index_ar)
        self.infor.add_para('find_U_matrix', self.section_ar[4], leastsq_solution.success)
        self.infor.add_para('U_matrix', self.section_ar[4], U_matrix)
        self.infor.add_para('additional_rotation_matrix', self.section_ar[4], additional_rotation_matrix)
        self.infor.add_para('angular_deviations', self.section_ar[4], angular_deviations)
        self.infor.infor_writer()
        return

    def possible_hkl_index(self, sample_name, scan_num):
        """
        Calculate possible hkl indexes of a diffraction peak.

        Parameters
        ----------
        sample_name : str
            The name of the specimen, which is defined by the p10_newfile name or spec_newfile name.
        scan_num : int
            The scan number.

        Returns
        -------
        None.

        """
        assert self.infor.section_exists(self.section_ar[4]), 'The U matrix must be calculated before determining the index of unknown diffraction peaks!'
        B_matrix = np.array(self.infor.get_para_value('B_matrix', self.section_ar[2]), dtype=float)
        U_matrix = np.array(self.infor.get_para_value('U_matrix', self.section_ar[4]), dtype=float)
        if not self.infor.section_exists(self.section_ar[3] % (sample_name, scan_num)):
            self.load_peak_infor(sample_name, scan_num)
        q_vector = np.array(self.infor.get_para_value('q_vector', self.section_ar[3] % (sample_name, scan_num)), dtype=float)
        q_vector = np.flip(q_vector)
        hkl_index = np.dot(np.linalg.inv(B_matrix), np.dot(np.linalg.inv(U_matrix), q_vector))
        print('Possible index of the diffraction peak is ' + str(np.around(hkl_index, decimals=2)))
        return

    def hkl_to_angles(self, aimed_hkl, rotation_axis, given_motor_values, limitations=None):
        """
        Calculate the possible angles for the aimed hkl index.

        Parameters
        ----------
        aimed_hkl : list
            The hkl index to be measured.
        rotation_axis : list
            The rotation axis to be used, e.g. ['om', 'del', 'phi'].
        given_motor_values : list
            The given values of other motors.
        limitations : tuple, optional
            The limitations of the three rotation _axis to be searched. The default is None.

        Returns
        -------
        None.

        """
        aimed_hkl = np.array(aimed_hkl, dtype=float)
        detector_para = self.load_parameters(['detector_distance', 'pixelsize', 'detector_rotation', 'direct_beam_position'], section=self.section_ar[1])
        motor_names = self.infor.get_para_value('motor_names', self.section_ar[0])
        B_matrix = np.array(self.infor.get_para_value('B_matrix', self.section_ar[2]), dtype=float)
        U_matrix = np.array(self.infor.get_para_value('U_matrix', self.section_ar[4]), dtype=float)
        cch = detector_para[3]

        expected_q = np.flip(np.dot(U_matrix, np.dot(B_matrix, aimed_hkl)))
        position_parameters = np.zeros(7)
        para_selected = []
        for i, element in enumerate(motor_names):
            if element in rotation_axis:
                para_selected.append(1)
            else:
                para_selected.append(0)
        para_selected = np.array(para_selected, dtype='bool')
        if limitations is None:
            limitations = ([-180, -180, -180], [180, 180, 180])
        offsets = least_squares(self.q_error_single_peak, [10.0, 10.0, 10.0], bounds=limitations, args=(cch, position_parameters, detector_para, para_selected, expected_q, given_motor_values))
        given_motor_values[para_selected] = offsets.x
        for i in range(len(motor_names)):
            print('%s = %.3f' % (motor_names[i], given_motor_values[i]))
        return
