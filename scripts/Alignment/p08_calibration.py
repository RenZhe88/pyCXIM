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
import sys
sys.path.append(r'E:\Work place 3\testprog\pyCXIM_master')
from pyCXIM.Common.Information_file_generator import InformationFileIO
from pyCXIM.p08_scan_reader.p08_eiger_reader import P08EigerScan
from pyCXIM.RSM.RC2RSM import RC2RSM_6C

def calibration():
    # Inputs: general information
    year = "2023"
    beamtimeID = "11016147"
    pixelsize = 0.075
    detector = 'eiger1m'
    Calibration_type = 'detector'
    # Calibration_type = 'single Bragg 6C'

    # Inputs:Detector parameters
    if Calibration_type == 'detector':
        p08_file = r"det_cal"
        scan_num = 2

    # Inputs:Simple calibration with symmetric diffraction peak
    if Calibration_type == 'single Bragg 6C':
        p08_file = r"B12SYNS1P1"
        scan_num = 13
        geometry = 'out_of_plane'
        peak = np.array([0, 0, 12])
        lattice_constants = [4.7597, 4.7597, 12.993]
        # omega, delta, chi, phi, det_rot, energy
        error_source = ['omega', 'delta', 'chi']

    # Inputs: paths
    path = r"U:\2023\data\11016147\raw"
    pathsave = r"E:\Work place 3\sample\XRD\20230615 PTO_insitu"
    pathmask = r''
    pathinfor = os.path.join(pathsave, "calibration.txt")
    section_ar = ['General Information', 'Detector calibration', 'Bragg peak calibration %s']

    infor = InformationFileIO(pathinfor)
    infor.add_para('year', section_ar[0], year)
    infor.add_para('beamtimeID', section_ar[0], beamtimeID)
    infor.add_para('pathinfor', section_ar[0], pathinfor)
    infor.add_para('pathmask', section_ar[0], pathmask)
    infor.add_para('pathsave', section_ar[0], pathsave)

    if Calibration_type == 'detector':
        scan = P08EigerScan(path, p08_file, scan_num, pathmask=pathmask)
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
        infor.add_para('p08_newfile', section_ar[1], p08_file)
        infor.add_para('scan_number', section_ar[1], scan_num)
        infor.add_para('direct_beam_position', section_ar[1], list(cch))
        infor.add_para('detector_distance', section_ar[1], distance)
        infor.add_para('pixelsize', section_ar[1], pixelsize)
        infor.add_para('detector_rotation', section_ar[1], -np.degrees(fit_x[0] / fit_y[0]))
        infor.add_para('shift_x_per_radians', section_ar[1], fit_x[0])
        infor.add_para('shift_y_per_radians', section_ar[1], fit_y[0])
        infor.infor_writer()

    elif Calibration_type == 'single Bragg 6C':
        infor = InformationFileIO(pathinfor)
        infor.infor_reader()
        distance = infor.get_para_value('detector_distance')
        pixelsize = infor.get_para_value('pixelsize')
        cch = infor.get_para_value('direct_beam_position')
        det_rot = infor.get_para_value('detector_rotation')
        print('distance=%.1f' % distance)
        print('pixelsize=%f' % pixelsize)
        print('cch=' + str(cch))
        print('detector_inplane_rotation=%f' % det_rot)

        # calculate the unit of the grid intensity
        qx = 2 * np.pi * peak[0] / lattice_constants[0]
        qy = 2 * np.pi * peak[1] / lattice_constants[1]
        qz = 2 * np.pi * peak[2] / lattice_constants[2]
        expected_q = np.array([qz, qy, qx])

        infor.del_para_section(section_ar[2] % (str(peak)))
        infor.add_para('path', section_ar[2] % (str(peak)), path)
        infor.add_para('p08_newfile', section_ar[2] % (str(peak)), p08_file)
        infor.add_para('scan_number', section_ar[2] % (str(peak)), scan_num)
        infor.add_para('peak', section_ar[2] % (str(peak)), list(peak))
        infor.add_para('lattice_constants', section_ar[2] % (str(peak)), list(lattice_constants))
        infor.add_para('expected_q', section_ar[2] % (str(peak)), list(np.flip((expected_q))))
        infor.add_para('error_source', section_ar[2] % (str(peak)), error_source)
        infor.add_para('geometry', section_ar[2] % (str(peak)), geometry)

        scan = P08EigerScan(path, p08_file, scan_num, detector, pathmask=pathmask)
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
        print('om=%f, delta=%.2f, chi=%.2f, phi=%.2f, gamma=%.2f' % (omega, delta, chi, phi, gamma))
        print('energy=%d' % energy)

        infor.add_para('omega', section_ar[2] % (str(peak)), omega)
        infor.add_para('delta', section_ar[2] % (str(peak)), delta)
        infor.add_para('chi', section_ar[2] % (str(peak)), chi)
        infor.add_para('phi', section_ar[2] % (str(peak)), phi)
        infor.add_para('gamma', section_ar[2] % (str(peak)), gamma)
        infor.add_para('energy', section_ar[2] % (str(peak)), energy)

        parameters = [omega, delta, chi, phi, gamma, det_rot, energy, distance]
        paranames = ['omega', 'delta', 'chi', 'phi', 'gamma', 'det_rot', 'energy', 'distance']
        para_offsets_full = np.zeros_like(paranames, dtype=float)
        para_selected = []
        for i, element in enumerate(paranames):
            if element in error_source:
                para_selected.append(1)
            else:
                para_selected.append(0)
                if infor.get_para_value('%s_error' % element) is not None:
                    parameters[i] = parameters[i] + infor.get_para_value('%s_error' % element)
                    para_offsets_full[i] = infor.get_para_value('%s_error' % element)
        para_selected = np.array(para_selected, dtype='bool')

        offsets = fsolve(cal_q_error_single_peak, [0.1, 0.1, 0.1], args=(scan_motor_ar, geometry, para_selected, parameters, pixelsize, pch, cch, expected_q))
        print('remaining error is' + str(cal_q_error_single_peak(offsets, scan_motor_ar, geometry, para_selected, parameters, pixelsize, pch, cch, expected_q)))
        para_offsets_full[para_selected] = np.array(offsets)
        for i, paraname in enumerate(paranames):
            infor.add_para('%s_error' % paraname, section_ar[2] % (str(peak)), para_offsets_full[i])
            if paraname in error_source:
                print('%s_offset=%.2f' % (paraname, para_offsets_full[i]))

        infor.infor_writer()
    return

def cal_q_error_single_peak(offsets, scan_motor_ar, geometry, para_selected, parameters, pixelsize, pch, cch, expected_q):
    assert sum(para_selected) == 3, 'For single peak only three parameter error could be fitted!'
    assert len(offsets) == 3, 'For single peak only three parameter error could be fitted!'
    assert len(para_selected) == len(parameters), 'The parameter selected must have the same dimension as the parameters!'
    if geometry == 'out_of_plane' and para_selected[0]:
        scan_motor_ar = scan_motor_ar + np.array(offsets)[0]
    elif geometry == 'in_plane' and para_selected[3]:
        scan_motor_ar = scan_motor_ar + np.array(offsets)[3]
    else:
        scan_motor_ar = scan_motor_ar
    parameters = np.array(parameters)
    parameters[para_selected] = parameters[para_selected] + np.array(offsets)
    parameters = list(parameters)
    RSM_convertor = RC2RSM_6C(scan_motor_ar, geometry, *parameters, pixelsize)
    q_vector = RSM_convertor.cal_abs_q_pos(pch, cch)
    error_ar = q_vector - expected_q
    return error_ar

if __name__ == '__main__':
    calibration()
