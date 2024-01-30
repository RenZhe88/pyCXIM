# -*- coding: utf-8 -*-
"""
Cablirate the detector distances and the offsets of the 6c diffractometer
Created on Fri May 12 14:39:49 2023

@author: Ren Zhe
@email: renzhe@ihep.ac.cn
"""

import sys

sys.path.append(r'F:\Work place 3\testprog\pyCXIM_master')
from pyCXIM.scan_reader.Desy.eiger_reader import DesyEigerImporter
from pyCXIM.RSM.Calibration_6C import Calibration


def calibration():
    # Inputs: general information
    Calibration_type = 'detector'
    # Calibration_type = 'crystal infor'
    # Calibration_type = 'single Bragg 6C'
    # Calibration_type = 'multiple Bragg 6C'
    # Calibration_type = 'Calculate peak index'

    # Inputs: paths
    path = r"F:\Raw Data\20230615_P08_PTO_STO_in_situ\raw"
    pathsave = r"F:\Work place 4"
    pathmask = r''
    detector = 'eiger1m'

    # Inputs:Detector parameters
    if Calibration_type == 'detector':
        p08_file = r"det_cal"
        scan_num = 2

    Calibinfor = Calibration(pathsave)

    if not Calibinfor.section_exists('General Information'):
        Calibinfor.init_beamtime('p08', path, detector, pathmask)

    if Calibration_type == 'detector':
        Calibinfor.detector_calib(p08_file, scan_num)

    return

if __name__ == '__main__':
    calibration()
