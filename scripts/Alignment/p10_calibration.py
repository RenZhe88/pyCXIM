# -*- coding: utf-8 -*-
"""
Cablirate the detector distances and the offsets of the 6c diffractometer
Created on Fri May 12 14:39:49 2023

@author: renzhe
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import sys
sys.path.append(r'F:\Work place 3\testprog\pyCXIM_master')
from pyCXIM.Common.Information_file_generator import InformationFileIO
from pyCXIM.scan_reader.Desy.eiger_reader import DesyEigerImporter


def B_matrix_cal(lattice_constants):
    a, b, c = lattice_constants[:3]
    alpha, beta, gamma = np.deg2rad(lattice_constants[3:])

    f = np.sqrt(1 - np.square(np.cos(alpha)) - np.square(np.cos(beta)) - np.square(np.cos(gamma)) + 2 * np.cos(alpha) * np.cos(beta) * np.cos(gamma))
    A_matrix = np.array([[a, b * np.cos(gamma), c * np.cos(beta)],
                         [0, b * np.sin(gamma), c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)],
                         [0, 0, c * f / np.sin(gamma)]])
    B_matrix = 2.0 * np.pi * np.linalg.inv(A_matrix).T
    return B_matrix


def aim_U_matrix_cal(surface_dir, inplane_dir, B_matrix):
    surface_dir = np.dot(B_matrix, surface_dir)
    surface_dir = surface_dir / np.linalg.norm(surface_dir)
    inplane_dir = np.dot(B_matrix, inplane_dir)
    inplane_dir1 = inplane_dir - np.dot(inplane_dir, surface_dir) * surface_dir
    inplane_dir1 = inplane_dir1 / np.linalg.norm(inplane_dir1)
    inplane_dir2 = np.cross(surface_dir, inplane_dir1)
    inplane_dir2 = inplane_dir2 / np.linalg.norm(inplane_dir2)

    inv_U = np.array([inplane_dir1, inplane_dir2, surface_dir])
    inv_U = inv_U.T
    aimed_U_matrix = np.linalg.inv(inv_U)
    return aimed_U_matrix


def load_peak_infor(path, p10_file, scan_num, pathinfor, pathmask,
                    geometry='out_of_plane', detector='e4m'):
    # Import the basic informaiton from p10 calibration scan
    infor = InformationFileIO(pathinfor)
    infor.infor_reader()
    distance = infor.get_para_value('detector_distance')
    pixelsize = infor.get_para_value('pixelsize')
    cch = infor.get_para_value('direct_beam_position')
    det_rot = infor.get_para_value('detector_rotation')

    # Read the scan files and convert the peak position to the q_vector
    scan = DesyEigerImporter('p10', path, p10_file, scan_num, detector, pathmask=pathmask)
    print(scan)
    pch = scan.eiger_find_peak_position(cut_width=[50, 50])
    if geometry == 'out_of_plane':
        scan_motor_ar = scan.get_scan_data('om')
        omega = scan_motor_ar[int(pch[0])]
        phi = scan.get_motor_pos('phi')
    elif geometry == 'in_plane':
        scan_motor_ar = scan.get_scan_data('phi')
        omega = scan.get_motor_pos('om')
        phi = scan_motor_ar[int(pch[0])]
    delta = scan.get_motor_pos('del')
    chi = scan.get_motor_pos('chi') - 90.0
    gamma = scan.get_motor_pos('gam')
    energy = scan.get_motor_pos('fmbenergy')

    position_parameters = np.array([omega, delta, chi, phi, gamma, energy], dtype=float)
    calibration_parameters = np.array([det_rot, distance, pixelsize], dtype=float)
    pixel_position = np.array([cch[0] - pch[1], cch[1] - pch[2]])
    return pixel_position, position_parameters, calibration_parameters

def cal_abs_q_pos(pixel_position, position_parameters, calibration_parameters,
                  offset_values=np.zeros(6)):
    offset_values = np.array(offset_values, dtype=float)
    position_parameters = position_parameters + offset_values
    omega = np.deg2rad(position_parameters[0])
    delta = np.deg2rad(position_parameters[1])
    chi = np.deg2rad(position_parameters[2])
    phi = np.deg2rad(position_parameters[3])
    gamma = np.deg2rad(position_parameters[4])
    energy = position_parameters[5]
    det_rot = np.deg2rad(calibration_parameters[0])
    distance = calibration_parameters[1]
    pixelsize = calibration_parameters[2]

    pixel_distance = np.linalg.norm([distance, (pixel_position[0]) * pixelsize, (pixel_position[1]) * pixelsize])
    q_vector = np.array([pixel_position[0], pixel_position[1], distance / pixelsize])
    det_rot_transform = np.array([[np.cos(det_rot), -np.sin(det_rot), 0],
                                  [np.sin(det_rot), np.cos(det_rot), 0],
                                  [0, 0, 1.0]])
    q_vector = np.dot(det_rot_transform, q_vector)
    delta_transform = np.array([[np.cos(delta), 0, np.sin(delta)],
                                [0, 1, 0],
                                [-np.sin(delta), 0, np.cos(delta)]])
    q_vector = np.dot(delta_transform, q_vector)
    gamma_transform = np.array([[1, 0, 0],
                                [0, np.cos(gamma), np.sin(gamma)],
                                [0, -np.sin(gamma), np.cos(gamma)]])
    q_vector = np.dot(gamma_transform, q_vector)
    q_vector = q_vector - np.array([0, 0, pixel_distance / pixelsize])
    omega_transform = np.array([[np.cos(omega), 0, -np.sin(omega)],
                                [0, 1, 0],
                                [np.sin(omega), 0, np.cos(omega)]])
    q_vector = np.dot(omega_transform, q_vector)
    chi_transform = np.array([[np.cos(chi), -np.sin(chi), 0],
                              [np.sin(chi), np.cos(chi), 0],
                              [0, 0, 1]])
    q_vector = np.dot(chi_transform, q_vector)
    phi_transform = np.array([[1, 0, 0],
                              [0, np.cos(phi), np.sin(phi)],
                              [0, -np.sin(phi), np.cos(phi)]])
    q_vector = np.dot(phi_transform, q_vector)
    hc = 1.23984 * 10000.0
    wavelength = hc / energy
    units = 2 * np.pi * pixelsize / wavelength / pixel_distance
    q_vector = q_vector * units
    return q_vector


def cal_q_error_single_peak(offsets, position_parameters, para_selected,
                            pixel_position, calibration_parameters,
                            full_offset_values, expected_q):
    assert sum(para_selected) == 3, 'For single peak only three parameter error could be fitted!'
    assert len(offsets) == 3, 'For single peak only three parameter error could be fitted!'

    full_offset_values[para_selected] = np.array(offsets, dtype=float)
    q_vector = cal_abs_q_pos(pixel_position, position_parameters, calibration_parameters, full_offset_values)
    error_ar = q_vector - expected_q
    return error_ar

def cal_q_error_multiple_peak(euler_angles, B_matrix, hkl_ar, q_ar):
    rotation_matrix = R.from_euler('yxz', euler_angles, degrees=True)
    additional_U_matrix = rotation_matrix.as_matrix()
    error_ar = np.array([])
    for i in np.arange(hkl_ar.shape[0]):
        error = np.flip(np.dot(additional_U_matrix, np.dot(B_matrix, hkl_ar[i, :]))) - q_ar[i, :]
        error_ar = np.append(error_ar, error)
    return error_ar


def calibration():
    # Inputs: general information
    year = 2023
    beamtimeID = 11017662
    pixelsize = 0.075
    detector = 'e4m'
    Calibration_type = 'detector'
    # Calibration_type = 'single Bragg 6C'
    # Calibration_type = 'multiple Bragg 6C'

    # Inputs: paths
    path = r"F:\Raw Data\20230925_P10_BFO_Pt_LiNiMnO2_AlScN\raw"
    pathsave = r"F:\Work place 3\sample\XRD"
    pathmask = r'F:\Work place 3\testprog\pyCXIM_master\detector_mask\p10_e4m_mask.npy'

    # Inputs:Detector parameters
    if Calibration_type == 'detector':
        p10_file = r"det_cal"
        scan_num = 2

    # Inputs:Simple calibration with symmetric diffraction peak
    elif Calibration_type == 'single Bragg 6C':
        p10_file = r"PTO_STO_DSO_730"
        scan_num = 1
        geometry = 'out_of_plane'
        peak = np.array([2, 2, 0], dtype=float)
        surface_dir = np.array([1, 1, 0], dtype=float)
        inplane_dir = np.array([0, 0, 1], dtype=float)
        lattice_constants = [5.44, 5.71, 7.89, 90, 90, 90]
        # omega, delta, chi, phi, gamma, energy
        error_source = ['omega', 'delta', 'chi']
        known_error_values = np.array([0, 0, 0, 0, 0, 0], dtype=float)

    # Inputs:Simple calibration with symmetric diffraction peak
    elif Calibration_type == 'multiple Bragg 6C':
        p10_file = r"PTO_STO_DSO_730"
        scan_num_ar = [1, 7, 17, 19]
        geometry = 'out_of_plane'
        peak_index_ar = np.array([[2, 2, 0], [3, 3, 2], [2, 2, 0], [4, 2, 0]], dtype=float)
        surface_dir = np.array([1, 1, 0], dtype=float)
        inplane_dir = np.array([0, 0, 1], dtype=float)
        lattice_constants = [5.44, 5.71, 7.89, 90, 90, 90]
        aimed_hkl = [4.0, 2., 0]
        rotation_source = ['omega', 'delta', 'phi']
        fixed_values = np.array([0, 0, 0.20000000000000284, 0, 0, 13100.08922750442], dtype=float)

    pathinfor = os.path.join(pathsave, "calibration.txt")
    section_ar = ['General Information', 'Detector calibration', 'Bragg peak %s calibration scan %d', 'Calculated UB matrix']
    infor = InformationFileIO(pathinfor)
    infor.add_para('year', section_ar[0], year)
    infor.add_para('beamtimeID', section_ar[0], beamtimeID)
    infor.add_para('pathinfor', section_ar[0], pathinfor)
    infor.add_para('pathmask', section_ar[0], pathmask)
    infor.add_para('pathsave', section_ar[0], pathsave)

    if Calibration_type == 'detector':
        scan = DesyEigerImporter('p10', path, p10_file, scan_num, detector, pathmask=pathmask)
        print(scan)
        command = scan.get_command()
        motor = str(command.split()[1])
        angle_ar = scan.get_scan_data(motor)

        cch = np.zeros(2, dtype=int)
        X_pos, Y_pos, int_ar = scan.eiger_peak_pos_per_frame()
        angle_ar = angle_ar[int_ar > np.amax(int_ar) * 0.5]
        X_pos = X_pos[int_ar > np.amax(int_ar) * 0.5]
        Y_pos = Y_pos[int_ar > np.amax(int_ar) * 0.5]
        fit_x = np.polyfit(np.radians(angle_ar), X_pos, 1)
        fit_y = np.polyfit(np.radians(angle_ar), Y_pos, 1)
        distance = np.sqrt(fit_x[0]**2.0 + fit_y[0]**2.0) * pixelsize
        cch[0] = np.round(fit_y[1])
        cch[1] = np.round(fit_x[1])

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

        infor.add_para('path', section_ar[1], path)
        infor.add_para('p10_newfile', section_ar[1], p10_file)
        infor.add_para('scan_number', section_ar[1], scan_num)
        infor.add_para('direct_beam_position', section_ar[1], list(cch))
        infor.add_para('detector_distance', section_ar[1], distance)
        infor.add_para('pixelsize', section_ar[1], pixelsize)
        infor.add_para('detector_rotation', section_ar[1], -np.degrees(fit_x[0] / fit_y[0]))
        infor.add_para('shift_x_per_radians', section_ar[1], fit_x[0])
        infor.add_para('shift_y_per_radians', section_ar[1], fit_y[0])
        infor.infor_writer()

    elif Calibration_type == 'single Bragg 6C':
        # calculate the unit of the grid intensity
        B_matrix = B_matrix_cal(lattice_constants)
        expected_U_matrix = aim_U_matrix_cal(surface_dir, inplane_dir, B_matrix)
        expected_q = np.dot(expected_U_matrix, np.dot(B_matrix, peak))
        expected_q = np.flip(expected_q)

        infor = InformationFileIO(pathinfor)
        infor.infor_reader()
        infor.del_para_section(section_ar[2] % (str(peak), scan_num))
        infor.add_para('path', section_ar[2] % (str(peak), scan_num), path)
        infor.add_para('p10_newfile', section_ar[2] % (str(peak), scan_num), p10_file)
        infor.add_para('scan_number', section_ar[2] % (str(peak), scan_num), scan_num)
        infor.add_para('geometry', section_ar[2] % (str(peak), scan_num), geometry)
        infor.add_para('peak', section_ar[2] % (str(peak), scan_num), list(peak))
        infor.add_para('surface_direction', section_ar[2] % (str(peak), scan_num), list(surface_dir))
        infor.add_para('inplane_direction', section_ar[2] % (str(peak), scan_num), list(inplane_dir))
        infor.add_para('lattice_constants', section_ar[2] % (str(peak), scan_num), list(lattice_constants))
        infor.add_para('expected_q', section_ar[2] % (str(peak), scan_num), list(np.flip((expected_q))))
        infor.add_para('error_source', section_ar[2] % (str(peak), scan_num), error_source)

        pixel_position, position_parameters, calibration_parameters = load_peak_infor(path, p10_file, scan_num, pathinfor, pathmask, geometry='out_of_plane', detector='e4m')
        infor.add_para('omega', section_ar[2] % (str(peak), scan_num), position_parameters[0])
        infor.add_para('delta', section_ar[2] % (str(peak), scan_num), position_parameters[1])
        infor.add_para('chi', section_ar[2] % (str(peak), scan_num), position_parameters[2])
        infor.add_para('phi', section_ar[2] % (str(peak), scan_num), position_parameters[3])
        infor.add_para('gamma', section_ar[2] % (str(peak), scan_num), position_parameters[4])
        infor.add_para('energy', section_ar[2] % (str(peak), scan_num), position_parameters[5])

        paranames = ['omega', 'delta', 'chi', 'phi', 'gamma', 'energy']
        para_selected = []
        for i, element in enumerate(paranames):
            if element in error_source:
                para_selected.append(1)
            else:
                para_selected.append(0)
        para_selected = np.array(para_selected, dtype='bool')

        offsets = fsolve(cal_q_error_single_peak, [0.1, 0.1, 0.1], args=(position_parameters, para_selected, pixel_position, calibration_parameters, known_error_values, expected_q))
        print('remaining error is' + str(cal_q_error_single_peak(offsets, position_parameters, para_selected, pixel_position, calibration_parameters, known_error_values, expected_q)))
        known_error_values[para_selected] = np.array(offsets)
        for i, paraname in enumerate(paranames):
            infor.add_para('%s_error' % paraname, section_ar[2] % (str(peak), scan_num), known_error_values[i])
            if paraname in error_source:
                print('%s_offset=%.2f' % (paraname, known_error_values[i]))

        infor.infor_writer()

    elif Calibration_type == 'multiple Bragg 6C':
        # calculate the unit of the grid intensity
        B_matrix = B_matrix_cal(lattice_constants)

        q_ar = np.zeros_like(peak_index_ar, dtype=float)
        for i in range(len(scan_num_ar)):
            pixel_position, position_parameters, calibration_parameters = load_peak_infor(path, p10_file, scan_num_ar[i], pathinfor, pathmask, geometry='out_of_plane', detector='e4m')
            q_ar[i, :] = cal_abs_q_pos(pixel_position, position_parameters, calibration_parameters)

        U_matrix_angles = least_squares(cal_q_error_multiple_peak, np.array([10.0, 10.0, 10.0]), args=(B_matrix, peak_index_ar, q_ar))
        print('Find UB matrix?')
        remaining_error = cal_q_error_multiple_peak(U_matrix_angles.x, B_matrix, peak_index_ar, q_ar)
        find_U_Matrix_indicator = np.allclose(remaining_error, np.zeros(len(remaining_error)), atol=0.01)
        print(find_U_Matrix_indicator)
        print('Remaining error for the fitting:')
        print(remaining_error)

        rotation_matrix = R.from_euler('yxz', U_matrix_angles.x, degrees=True)
        U_matrix = rotation_matrix.as_matrix()
        print('measured UB matrix')
        print(np.around(U_matrix, 2))

        expected_U_matrix = aim_U_matrix_cal(surface_dir, inplane_dir, B_matrix)
        print('expected_UB')
        print(np.around(expected_U_matrix, 2))
        additional_rotation = np.dot(U_matrix, np.linalg.inv(expected_U_matrix))
        additional_rotation = R.from_matrix(additional_rotation)
        print('Angular deviations')
        angular_deviations = additional_rotation.as_euler('yxz', degrees=True)
        print(angular_deviations)

        infor = InformationFileIO(pathinfor)
        infor.infor_reader()
        infor.del_para_section(section_ar[3])
        infor.add_para('path', section_ar[3], path)
        infor.add_para('p10_newfile', section_ar[3], p10_file)
        infor.add_para('scan_num_list', section_ar[3], list(scan_num_ar))
        infor.add_para('peak_index_list', section_ar[3], (peak_index_ar.tolist()))
        infor.add_para('surface_direction', section_ar[3], list(surface_dir))
        infor.add_para('inplane_direction', section_ar[3], list(inplane_dir))
        infor.add_para('lattice_constants', section_ar[3], list(lattice_constants))
        infor.add_para('peak_position_list', section_ar[3], q_ar.tolist())
        infor.add_para('find_U_matrix', section_ar[3], find_U_Matrix_indicator)
        infor.add_para('U_matrix', section_ar[3], U_matrix.tolist())
        infor.add_para('expected_U_matrix', section_ar[3], expected_U_matrix.tolist())
        infor.add_para('additional_rotation_matrix', section_ar[3], additional_rotation.as_matrix().tolist())
        infor.add_para('angular_deviations_omega', section_ar[3], angular_deviations[0])
        infor.add_para('angular_deviations_chi', section_ar[3], angular_deviations[1])
        infor.add_para('angular_deviations_phi', section_ar[3], angular_deviations[2])
        infor.infor_writer()

        aimed_hkl = np.array(aimed_hkl, dtype=float)
        expected_q = np.flip(np.dot(U_matrix, np.dot(B_matrix, aimed_hkl)))

        paranames = ['omega', 'delta', 'chi', 'phi', 'gamma', 'energy']
        position_parameters = fixed_values
        para_selected = []
        for i, element in enumerate(paranames):
            if element in rotation_source:
                para_selected.append(1)
                position_parameters[i] = 0
            else:
                para_selected.append(0)
        para_selected = np.array(para_selected, dtype='bool')

        offsets = least_squares(cal_q_error_single_peak, [10.0, 10.0, -10.0], bounds=([-4, 0, -180], [180, 180, 180]), args=(position_parameters, para_selected, [0.0, 0.0], calibration_parameters, np.zeros(6), expected_q))
        # offsets = offsets - (offsets / 360).astype(int) * 360.0
        position_parameters[para_selected] = offsets.x
        for i in range(6):
            print('%s = %.3f' % (paranames[i], position_parameters[i]))
    return


if __name__ == '__main__':
    calibration()
