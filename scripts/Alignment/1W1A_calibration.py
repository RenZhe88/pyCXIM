# -*- coding: utf-8 -*-
"""
Cablirate the detector distances and the offsets of the 6c diffractometer.
Created on Fri May 12 14:39:49 2023

@author: Ren Zhe
@email: renzhe@ihep.ac.cn
"""

import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append(r'F:\Work place 3\testprog\pyCXIM_master')
from pyCXIM.RSM.Calibration_6C import Calibration


def calibration():
    # Inputs: general information
    # Calibration_type = 'detector'
    Calibration_type = 'crystal infor'
    # Calibration_type = 'single Bragg 6C'
    # Calibration_type = 'multiple Bragg 6C'
    # Calibration_type = 'hkl_to_angles'

    # Inputs: paths
    path = r"F:\Work place 4\sample\XRD\Additional Task\20240131 1W1A test data\rsm"
    pathsave = r"F:\Work place 4\sample\XRD"
    pathmask = r'F:\Work place 3\testprog\pyCXIM_master\detector_mask\badpix_mask.tif'
    detector = '300K-A'

    # Inputs:Detector parameters
    if Calibration_type == 'detector':
        file_1w1a = r"sample_1"
        scan_num = 1

    elif Calibration_type == 'crystal infor':
        surface_dir = np.array([0, 0, 1], dtype=float)
        inplane_dir = np.array([1, 0, 0], dtype=float)
        lattice_constants = [3.9050, 3.9050, 3.9050, 90, 90, 90]

    # Inputs:Simple calibration with symmetric diffraction peak
    elif Calibration_type == 'single Bragg 6C':
        file_1w1a = r"sample_1"
        scan_num = 9
        peak = np.array([-1.0, 0, 3.0], dtype=float)
        # 'eta', 'del', 'chi', 'phi', 'nu', 'energy'
        error_source = ['eta', 'del', 'phi']
        known_error_values = np.array([0, 0, 0.3217048221225212, 0, 0, 0], dtype=float)

    # Inputs:Simple calibration with symmetric diffraction peak
    elif Calibration_type == 'multiple Bragg 6C':
        file_1w1a = r"sample_1"
        scan_num_ar = [8, 9]
        peak_index_ar = np.array([[0, 0, 2], [-1, 0, 3]], dtype=float)

    elif Calibration_type == 'hkl_to_angles':
        # 'eta', 'del', 'chi', 'phi', 'nu', 'energy'
        aimed_hkl = [0, 0, 2.0]
        rotation_source = ['eta', 'del', 'chi']
        fixed_values = np.array([0, 0, 0, 0, 0, 8016.564], dtype=float)
        limitations = None

    Calibinfor = Calibration(pathsave)

    if not Calibinfor.section_exists('General Information'):
        Calibinfor.init_beamtime('1w1a', path, detector, pathmask)

    if Calibration_type == 'detector':
        Calibinfor.detector_calib(file_1w1a, scan_num)

    elif Calibration_type == 'crystal infor':
        Calibinfor.crystal_information(lattice_constants, surface_dir, inplane_dir)
        Calibinfor.cal_possible_Bragg_angles()

    elif Calibration_type == 'single Bragg 6C':
        Calibinfor.single_Bragg_peak_tilt_cal(file_1w1a, scan_num, peak, error_source, known_error_values)

    elif Calibration_type == 'multiple Bragg 6C':
        Calibinfor.multi_Bragg_peak_U_matrix_cal(file_1w1a, scan_num_ar, peak_index_ar)

    elif Calibration_type == 'hkl_to_angles':
        Calibinfor.hkl_to_angles(aimed_hkl, rotation_source, fixed_values, limitations)

    return


if __name__ == '__main__':
    calibration()
