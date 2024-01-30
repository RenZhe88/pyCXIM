# -*- coding: utf-8 -*-
"""
Cablirate the detector distances and the offsets of the 6c diffractometer
Created on Fri May 12 14:39:49 2023

@author: Ren Zhe
@email: renzhe@ihep.ac.cn
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.append(r'F:\Work place 3\testprog\pyCXIM_master')
from pyCXIM.RSM.Calibration_6C import Calibration


def calibration():
    # Inputs: general information
    # Calibration_type = 'detector'
    # Calibration_type = 'crystal infor'
    # Calibration_type = 'single Bragg 6C'
    Calibration_type = 'multiple Bragg 6C'
    # Calibration_type = 'Calculate peak index'

    # Inputs: paths
    path = r"F:\Raw Data\20220608 P10 PTO BFO\raw"
    pathsave = r"F:\Work place 3\sample\XRD\20220608 Inhouse PTO film BFO islands\PTO_STO\PTO_STO_DSO_730"
    pathmask = r'F:\Work place 3\testprog\pyCXIM_master\detector_mask\p10_e4m_mask.npy'
    detector = 'e4m'

    # Inputs:Detector parameters
    if Calibration_type == 'detector':
        p10_file = r"det_cal"
        scan_num = 2

    elif Calibration_type == 'crystal infor':
        surface_dir = np.array([1, 1, 0], dtype=float)
        inplane_dir = np.array([0, 0, 1], dtype=float)
        lattice_constants = [5.44, 5.71, 7.89, 90, 90, 90]

    # Inputs:Simple calibration with symmetric diffraction peak
    elif Calibration_type == 'single Bragg 6C':
        p10_file = r"PTO_STO_DSO_730"
        scan_num = 7
        peak = np.array([3, 3, 2], dtype=float)
        # om, del, chi, phi, gamma, energy
        error_source = ['om', 'del', 'phi']
        known_error_values = np.array([0, 0, 0.7465478933840769, 0, 0, 0], dtype=float)

    # Inputs:Determine the U matrix based on measured Bragg peaks and their mill indexes
    elif Calibration_type == 'multiple Bragg 6C':
        p10_file = r"PTO_STO_DSO_730"
        scan_num_ar = [1, 7, 17, 19]
        peak_index_ar = np.array([[2, 2, 0], [3, 3, 2], [2, 2, 0], [4, 2, 0]], dtype=float)

    # Inputs:Determine the peak indexes based on the previously calculated U_matrix
    elif Calibration_type == 'Calculate peak index':
        p10_file = r"PTO_STO_DSO_730"
        scan_num = 19

        # aimed_hkl = [4.0, 2., 0]
        # rotation_source = ['omega', 'delta', 'phi']
        # fixed_values = np.array([0, 0, 0.20000000000000284, 0, 0, 13100.08922750442], dtype=float)

    Calibinfor = Calibration(pathsave)

    if not Calibinfor.section_exists('General Information'):
        Calibinfor.init_beamtime('p10', path, detector, pathmask)

    if Calibration_type == 'detector':
        Calibinfor.detector_calib(p10_file, scan_num)

    elif Calibration_type == 'crystal infor':
        Calibinfor.crystal_information(lattice_constants, surface_dir, inplane_dir)
        Calibinfor.cal_possible_Bragg_angles()

    elif Calibration_type == 'single Bragg 6C':
        Calibinfor.single_Bragg_peak_tilt_cal(p10_file, scan_num, peak, error_source, known_error_values)

    elif Calibration_type == 'multiple Bragg 6C':
        Calibinfor.multi_Bragg_peak_U_matrix_cal(p10_file, scan_num_ar, peak_index_ar)

    elif Calibration_type == 'Calculate peak index':
        Calibinfor.possible_hkl_index(p10_file, scan_num)
    return


if __name__ == '__main__':
    calibration()
