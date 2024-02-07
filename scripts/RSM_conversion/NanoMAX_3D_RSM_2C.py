#!/usr/local/bin/python2.7.3 -tttt
import numpy as np
import os
import sys
import time

sys.path.append(r'E:\Work place 3\testprog\pyCXIM_master')
from pyCXIM.Common.Information_file_generator import InformationFileIO
from pyCXIM.scan_reader.MAXIV.nanomax_merlin_reader import NanoMaxMerlinScan
from pyCXIM.RSM.RC2RSM_2C import RC2RSM_2C
import pyCXIM.RSM.RSM_post_processing as RSM_post_processing

def BCDI_preparation():
    start_time = time.time()
    # %% Inputs: select the functions of the code from the following mode

    # Inputs: general information
    year = "2023"
    nanomax_newfile = r"PTO_STO_DSO_730"
    scan_num = 60
    detector = 'merlin'

    qz_direction = 'surface direction'
    # qz_direction = 'diffraction vector direction'

    # Inputs: Detector parameters
    detector_distance = 1000.0
    pixelsize = 0.055
    # Direct beam position on the detector Y, X
    cch = [258, 258]
    # The half width of the detector roi in the order of [Y, X]
    wxy = [200, 200, 200, 200]
    # Roi on the detector [Ymin, Ymax, Xmin, Xmax]
    roi = [100, 400, 100, 400]
    # Method to find the centeral position for the cut, please select from 'maximum intensity', 'maximum integration',  'weight_center'
    cut_central_pos = 'maximum integration'

    # Half width of reciprocal space box size in pixels
    generating_3D_vtk_file = False

    # Inputs: Paths
    # the folder that stores the raw data of the beamtime
    path = r"F:\Raw Data\20230623_NanoMax_PTO_STO\raw"
    # the aimed saving folder
    pathsavefolder = r"F:\Work place 4\Temp"
    # the path for the mask file for the detector
    pathmask = r'F:\Work place 3\testprog\X-ray diffraction\Common functions\nanomax_merlin_mask.npy'

    # %% Read the information and detector images of the scan
    print("#################")
    print("Basic information")
    print("#################")
    # reading the fio file
    scan = NanoMaxMerlinScan('NanoMax', path, nanomax_newfile, scan_num, detector, pathsavefolder, pathmask)
    print(scan)
    energy = scan.get_motor_pos('energy')

    # Generate the paths for saving the data
    pathsave = scan.get_pathsave()
    pathinfor = os.path.join(pathsave, "scan_%04d_information.txt" % scan_num)

    # Load the detector images
    dataset, mask3D, pch, wxy = scan.merlin_load_images(roi, wxy, show_cen_image=(not os.path.exists(pathinfor)))
    wxy = np.array(wxy, dtype=int)
    dataset = np.flip(np.swapaxes(dataset, 1, 2), axis=1)
    mask3D = np.flip(np.swapaxes(mask3D, 1, 2), axis=1)
    pch[[1, 2]] = pch[[2, 1]]
    pch[1] = 515 - pch[1]
    wxy[[0, 1, 2, 3]] = wxy[[3, 2, 0, 1]]

    # load the scan motors
    # read the phi values
    scan_motor_ar = scan.get_scan_data('gonphi')
    # read the delta value, which is gamma in the horizontal direction
    two_theta = scan.get_motor_pos('gamma')
    if qz_direction == 'surface direction':
        scan_motor_offset = 0
    elif qz_direction == 'diffraction vector direction':
        scan_motor_offset = -two_theta / 2.0 - scan_motor_ar[int(pch[0])]

    scan_motor_ar = scan_motor_ar + scan_motor_offset
    # Finding the maximum peak position
    omega = scan_motor_ar[pch[0]]
    om_step = (scan_motor_ar[-1] - scan_motor_ar[0]) / (len(scan_motor_ar) - 1)
    print("peak at omega = %f" % (omega))

    RSM_converter = RC2RSM_2C(scan_motor_ar, two_theta, energy, detector_distance, pixelsize, cch)

    # writing the scan information to the aimed file
    section_ar = ['General Information', 'Paths', 'Scan Information', 'Routine1: Reciprocal space map', 'Routine2: direct cutting']
    infor = InformationFileIO(pathinfor)
    infor.add_para('command', section_ar[0], scan.get_command())
    infor.add_para('year', section_ar[0], year)
    infor.add_para('nanomax_newfile', section_ar[0], nanomax_newfile)
    infor.add_para('scan_number', section_ar[0], scan_num)

    infor.add_para('path', section_ar[1], path)
    infor.add_para('pathsave', section_ar[1], pathsave)
    infor.add_para('pathinfor', section_ar[1], pathinfor)
    infor.add_para('pathmask', section_ar[1], pathmask)

    infor.add_para('roi', section_ar[2], list(roi))
    infor.add_para('peak_position', section_ar[2], list(pch))
    infor.add_para('omega', section_ar[2], omega)
    infor.add_para('delta', section_ar[2], two_theta)
    infor.add_para('omegastep', section_ar[2], om_step)
    infor.add_para('omega_error', section_ar[2], scan_motor_offset)
    infor.add_para('direct_beam_position', section_ar[2], list(cch))
    infor.add_para('detector_distance', section_ar[2], detector_distance)
    infor.add_para('energy', section_ar[2], scan.get_motor_pos('energy'))
    infor.add_para('pixelsize', section_ar[2], pixelsize)
    infor.add_para('detector', section_ar[2], detector)

    infor.infor_writer()

    # %% Perform the corresponding operations

    print("")
    print("##################")
    print("Generating the RSM")
    print("##################")

    # Creating the aimed folder
    pathtmp = os.path.join(pathsave, "pynxpre")
    if not os.path.exists(pathtmp):
        os.mkdir(pathtmp)
    pathtmp = os.path.join(pathtmp, "reciprocal_space_map")
    if not os.path.exists(pathtmp):
        os.mkdir(pathtmp)

    # determining the rebin parameter
    rebinfactor = RSM_converter.cal_rebinfactor()

    # calculate the qx, qy, qz ranges of the scan
    if len(wxy) == 2:
        q_center, new_shape, RSM_unit = RSM_converter.cal_q_range([(pch[1] - wxy[0]), (pch[1] + wxy[0]), (pch[2] - wxy[1]), (pch[2] + wxy[1])], rebinfactor=rebinfactor)
    if len(wxy) == 4:
        q_center, new_shape, RSM_unit = RSM_converter.cal_q_range([(pch[1] - wxy[0]), (pch[1] + wxy[1]), (pch[2] - wxy[2]), (pch[2] + wxy[3])], rebinfactor=rebinfactor)

    # generate the 3D reciprocal space map
    print('Calculating intensity...')
    RSM_int = RSM_converter.RSM_conversion(dataset, new_shape, rebinfactor, cval=0, prefilter=False)
    del dataset

    # load the mask and generate the new mask for the 3D reciprocal space map
    print('Calculating the mask...')
    RSM_mask = RSM_converter.RSM_conversion(mask3D, new_shape, rebinfactor, cval=1, prefilter=False)
    del mask3D

    RSM_mask[RSM_mask >= 0.1] = 1
    RSM_mask[RSM_mask < 0.1] = 0

    print('saving the full 3D RSM')
    filename = "%s_%05d_RSM.npz" % (nanomax_newfile, scan_num)
    path3dRSM = os.path.join(pathsave, filename)
    np.savez_compressed(path3dRSM, data=RSM_int)
    filename = "%s_%05d_RSM_mask.npz" % (nanomax_newfile, scan_num)
    path3dmask = os.path.join(pathsave, filename)
    np.savez_compressed(path3dmask, data=RSM_mask)

    if generating_3D_vtk_file:
        filename = "scan%04d_diffraction_pattern.vti" % scan_num
        origin = q_center - np.array(RSM_int.shape, dtype=float) / 2.0 * RSM_unit
        RSM_post_processing.RSM2vti(pathtmp, RSM_int, filename, RSM_unit, origin=origin)

    # Generate the images of the reciprocal space map
    print('Generating the images of the RSM')
    qmax = np.array([np.argmax(np.sum(RSM_int, axis=(1, 2))), np.argmax(np.sum(RSM_int, axis=(0, 2))), np.argmax(np.sum(RSM_int, axis=(0, 1)))], dtype=int)
    pathsavetmp = os.path.join(pathsave, 'scan%04d_integrate' % scan_num + '_%s.png')
    RSM_post_processing.plot_with_units(RSM_int, q_center, RSM_unit, pathsavetmp)
    pathsavetmp = os.path.join(pathsave, 'scan%04d_cut' % scan_num + '_%s.png')
    RSM_post_processing.plot_with_units(RSM_int, q_center, RSM_unit, pathsavetmp, qmax=qmax)
    del RSM_int, RSM_mask

    # save the information
    infor.add_para('path3DRSM', section_ar[1], path3dRSM)
    infor.add_para('path3Dmask', section_ar[1], path3dmask)
    infor.add_para('roi_width', section_ar[3], list(wxy))
    infor.add_para('RSM_unit', section_ar[3], RSM_unit)
    infor.add_para('RSM_shape', section_ar[3], list(new_shape))
    infor.add_para('rebinfactor', section_ar[3], rebinfactor)
    infor.add_para('q_center', section_ar[3], list(q_center))
    infor.add_para('RSM_cut_central_mode', section_ar[3], cut_central_pos)
    infor.add_para('qcen', section_ar[3], list(qmax))
    infor.infor_writer()

    end_time = time.time()
    infor.add_para('total_calculation_time', section_ar[0], end_time - start_time)
    infor.infor_writer()
    return


if __name__ == '__main__':
    BCDI_preparation()
