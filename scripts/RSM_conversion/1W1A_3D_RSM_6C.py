# -*- coding: utf-8 -*-
"""
Calculate the reciprocal space map measured with six circle diffractometer at P10.

To get the correct RSM data, calibration has to be first performed generating the calibration file with script calibration_p10.py
In case of questions, please contact me. Detailed explaination file of the code could be sent upon requist.

Author: Ren Zhe
Date: 2020/12/04
Email: zhe.ren@desy.de or renzhetu001@gmail.com
"""

import os
import numpy as np
import sys
import time
sys.path.append(r'E:\Work place 3\testprog\pyCXIM_master')
from pyCXIM.Common.Information_file_generator import InformationFileIO
from pyCXIM.scan_reader.BSRF.pilatus_reader import BSRFPilatusImporter
from pyCXIM.RSM.RC2RSM_6C import RC2RSM_6C
import pyCXIM.RSM.RSM_post_processing as RSM_post_processing


def RSM_6C():
    start_time = time.time()
    # %%Inputs: general information
    year = "2023"
    beamtimeID = "1698819146"
    p10_file = r"sample_1"
    scan_num = 14
    detector = '300K-A'
    geometry = 'out_of_plane'

    # Roi on the detector [Ymin, Ymax, Xmin, Xmax]
    roi = None

    # Inputs: reciprocal space box size in pixels
    save_full_3D_RSM = True
    generating_3D_vtk_file = True

    # Inputs: paths
    path = r"F:\Work place 4\sample\XRD\Additional Task\20240131 1W1A test data\rsm"
    pathsave = r"F:\Work place 4\Temp"
    pathmask = r'F:\Work place 3\testprog\pyCXIM_master\detector_mask\1w1a_pilatus_mask.npy'
    pathcalib = r'F:\Work place 4\sample\XRD\Additional Task\20240131 1W1A test data\result\calibration.txt'

    # %% Generate the RSM
    print("#################")
    print("Basic information")
    print("#################")

    # Read images and fio files
    scan = BSRFPilatusImporter('1w1a', path, p10_file, scan_num, detector, pathsave, pathmask)
    print(scan)
    if geometry == 'out_of_plane':
        scan_motor_ar = scan.get_scan_data('eta')
    elif geometry == 'in_plane':
        scan_motor_ar = scan.get_scan_data('phi')
    scan_step = (scan_motor_ar[-1] - scan_motor_ar[0]) / (len(scan_motor_ar) - 1)
    eta = scan.get_motor_pos('eta')
    delta = scan.get_motor_pos('del')
    chi = scan.get_motor_pos('chi')
    phi = scan.get_motor_pos('phi')
    nu = scan.get_motor_pos('nu')
    mu = scan.get_motor_pos('mu')
    energy = scan.get_motor_pos('energy')
    scan.write_scan()

    # Generate the paths for saving the data
    pathsave = scan.get_pathsave()
    pathinfor = os.path.join(pathsave, "scan_%04d_information.txt" % scan_num)
    path3dintensity = os.path.join(pathsave, "scan%04d.npz" % scan_num)
    path3dmask = os.path.join(pathsave, "scan%04d_mask.npz" % scan_num)

    calibinfor = InformationFileIO(pathcalib)
    calibinfor.infor_reader()
    cch = calibinfor.get_para_value('direct_beam_position', section='Detector calibration')
    distance = calibinfor.get_para_value('detector_distance', section='Detector calibration')
    pixelsize = calibinfor.get_para_value('pixelsize', section='Detector calibration')
    det_rot = calibinfor.get_para_value('detector_rotation', section='Detector calibration')
    additional_rotation_matrix = np.array(calibinfor.get_para_value('additional_rotation_matrix', section='Calculated UB matrix'), dtype=float)

    dataset, mask3D, pch, roi = scan.load_rois(roi=roi, show_cen_image=(not os.path.exists(pathinfor)), normalize_signal='Monitor', correction_mode='constant')

    RSM_converter = RC2RSM_6C(scan_motor_ar, geometry,
                              eta, delta, chi, phi, nu, mu, energy,
                              distance, pixelsize, det_rot, cch,
                              additional_rotation_matrix)

    # determining the rebin parameter
    rebinfactor = RSM_converter.cal_rebinfactor()

    # Finding the maximum peak position
    print("peak at eta = %.2f, delta = %.2f, chi = %.2f, phi = %.2f, nu = %.2f" % (eta, delta, chi, phi, nu))

    # writing the scan information to the aimed file
    section_ar = ['General Information', 'Paths', 'Scan Information', 'Routine1: Reciprocal space map']
    infor = InformationFileIO(pathinfor)
    infor.add_para('command', section_ar[0], scan.get_command())
    infor.add_para('year', section_ar[0], year)
    infor.add_para('beamtimeID', section_ar[0], beamtimeID)
    infor.add_para('p10_newfile', section_ar[0], p10_file)
    infor.add_para('scan_number', section_ar[0], scan_num)
    infor.add_para('detector', section_ar[0], detector)

    infor.add_para('path', section_ar[1], path)
    infor.add_para('pathsave', section_ar[1], pathsave)

    infor.add_para('pathinfor', section_ar[1], pathinfor)
    infor.add_para('path3dintensity', section_ar[1], path3dintensity)
    infor.add_para('pathmask', section_ar[1], pathmask)
    infor.add_para('path3dmask', section_ar[1], path3dmask)

    infor.add_para('geometry', section_ar[2], geometry)
    infor.add_para('roi', section_ar[2], list(roi))
    infor.add_para('peak_position', section_ar[2], list(pch))
    infor.add_para('scan_step', section_ar[2], scan_step)
    infor.add_para('eta', section_ar[2], eta)
    infor.add_para('delta', section_ar[2], delta)
    infor.add_para('chi', section_ar[2], chi)
    infor.add_para('phi', section_ar[2], phi)
    infor.add_para('nu', section_ar[2], nu)
    infor.add_para('mu', section_ar[2], mu)

    infor.add_para('additional_rotation_matrix', section_ar[2], additional_rotation_matrix.tolist())
    infor.add_para('direct_beam_position', section_ar[2], list(cch))
    infor.add_para('detector_distance', section_ar[2], distance)
    infor.add_para('pixelsize', section_ar[2], pixelsize)
    infor.add_para('det_rot', section_ar[2], det_rot)

    infor.infor_writer()

    print("")
    print("##################")
    print("Generating the RSM")
    print("##################")

    # calculate the qx, qy, qz ranges of the scan
    q_origin, new_shape, RSM_unit = RSM_converter.cal_q_range(roi, rebinfactor=rebinfactor)

    # generate the 3D reciprocal space map
    print('Calculating intensity...')
    RSM_int = RSM_converter.RSM_conversion(dataset, new_shape, rebinfactor=rebinfactor, cval=0, prefilter=True)
    del dataset
    qmax = np.array([np.argmax(np.sum(RSM_int, axis=(1, 2))), np.argmax(np.sum(RSM_int, axis=(0, 2))), np.argmax(np.sum(RSM_int, axis=(0, 1)))], dtype=int)

    print('Calculating mask...')
    RSM_mask = RSM_converter.RSM_conversion(mask3D, new_shape, rebinfactor=rebinfactor, cval=1, prefilter=True)
    del mask3D

    if save_full_3D_RSM:
        print('saving 3D RSM and the corresponding mask')
        filename = "%s_%05d_RSM.npz" % (p10_file, scan_num)
        pathsaveRSM = os.path.join(pathsave, filename)
        np.savez_compressed(pathsaveRSM, data=RSM_int)
        infor.add_para('pathRSM', section_ar[1], pathsaveRSM)

        filename = "%s_%05d_RSM_mask.npz" % (p10_file, scan_num)
        pathsaveRSMmask = os.path.join(pathsave, filename)
        np.savez_compressed(pathsaveRSMmask, data=RSM_mask)
        infor.add_para('pathRSMmask', section_ar[1], pathsaveRSMmask)

    if generating_3D_vtk_file:
        print('saving VTI file for the visualization')
        filename = "scan%04d_diffraction_pattern.vti" % scan_num
        RSM_post_processing.RSM2vti(pathsave, RSM_int, filename, RSM_unit, q_origin)

    # Generate the images of the reciprocal space map
    print('Generating the images of the RSM')
    pathsavetmp = os.path.join(pathsave, 'scan%04d_integrate' % scan_num + '_%s.png')
    RSM_post_processing.plot_with_units(RSM_int, q_origin, RSM_unit, pathsavetmp)
    pathsavetmp = os.path.join(pathsave, 'scan%04d' % scan_num + '_%s.png')
    RSM_post_processing.plot_with_units(RSM_int, q_origin, RSM_unit, pathsavetmp, qmax=qmax)

    # save the information
    scan.write_scan()
    infor.add_para('RSM_shape', section_ar[3], list(new_shape))
    infor.add_para('rebinfactor', section_ar[3], rebinfactor)
    infor.add_para('RSM_unit', section_ar[3], RSM_unit)
    infor.add_para('q_origin', section_ar[3], q_origin)
    end_time = time.time()
    infor.add_para('total_time', section_ar[3], end_time - start_time)
    # infor.add_para('qmax', section_ar[3], qmax)
    infor.infor_writer()
    return

if __name__ == '__main__':
    RSM_6C()
