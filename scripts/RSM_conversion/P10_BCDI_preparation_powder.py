# -*- coding: utf-8 -*-
"""
Prepare the BCDI data for the phase retrieval processes.
Special cases designed for powder diffraction.
Created on Wed Jan 24 15:32:26 2024

@author: Ren Zhe
@email: renzhe@ihep.ac.cn
"""
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
import os
import sys
import time
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares

sys.path.append(r'E:\Work place 3\testprog\pyCXIM_master')
from pyCXIM.Common.Information_file_generator import InformationFileIO
from pyCXIM.scan_reader.Desy.eiger_reader import DesyEigerImporter
from pyCXIM.RSM.RC2RSM_6C import RC2RSM_6C
from pyCXIM.RSM.RC2RSM_6C import cal_q_pos
import pyCXIM.RSM.RSM_post_processing as RSM_post_processing


def q_error_symmetric_peak(two_angles, q_vector):
    euler_angles = [two_angles[0], 0, two_angles[1]]
    rotation_matrix = R.from_euler('yxz', euler_angles, degrees=True)
    additional_rotation_matrix = rotation_matrix.as_matrix()
    error = np.dot(additional_rotation_matrix, q_vector)
    return error[1:]

def BCDI_preparation():
    start_time = time.time()
    # %% Inputs: select the functions of the code from the following mode
    # ['Gif', 'Reciprocal_space_map', '2D_cuts']
    Functions_selected = ['Gif', 'Reciprocal_space_map', '2D_cuts']

    # Inputs: general information
    year = "2024"
    beamtimeID = "11018562"
    p10_newfile = 'WTY04'
    scan_num = 53
    detector = 'e4m'
    geometry = 'out_of_plane'
    # geometry = 'in_plane'

    # Inputs: Detector parameters
    # The half width of the detector roi in the order of [Y, X]
    wxy = [180, 180]
    # Roi on the detector [Ymin, Ymax, Xmin, Xmax]
    roi = [839 - 100, 839 + 100, 375 - 100, 375 + 100]
    # Method to find the centeral position for the cut, please select from 'maximum intensity', 'maximum integration',  'weight center'
    cut_central_pos = 'weight center'

    # Half width of reciprocal space box size in pixels
    RSM_bs = [80, 80, 50]
    use_prefilter = False
    save_full_3D_RSM = False
    generating_3D_vtk_file = False

    # Inputs: Paths
    # the folder that stores the raw data of the beamtime
    path = r"F:\Raw Data\20240601_P10_BFO_LiNiMnO2\raw"
    # the aimed saving folder
    pathsavefolder = r"F:\Work place 4\Temp"
    # the path for the mask file for the detector
    pathmask = r'F:\Work place 3\testprog\pyCXIM_master\detector_mask\p10_e4m_mask.npy'
    pathcalib = r'F:\Work place 4\sample\XRD\20240602_BFO_chiral_P10_Desy\Battery_cathode\calibration.txt'

    # %% Read the information and detector images of the scan
    print("#################")
    print("Basic information")
    print("#################")

    # reading the fio file
    scan = DesyEigerImporter('p10', path, p10_newfile, scan_num, detector, pathsavefolder, pathmask)
    print(scan)

    # Generate the paths for saving the data
    pathsave = scan.get_pathsave()
    pathinfor = os.path.join(pathsave, "scan_%04d_information.txt" % scan_num)
    dataset, mask3D, pch, wxy = scan.eiger_load_images(roi, wxy, show_cen_image=(not os.path.exists(pathinfor)))

    if geometry == 'out_of_plane':
        scan_motor_ar = scan.get_scan_data('om')
        omega = scan_motor_ar[pch[0]]
        phi = scan.get_motor_pos('phi')
        print("peak at omega = %f" % (omega))
    elif geometry == 'in_plane':
        scan_motor_ar = scan.get_scan_data('phi')
        phi = scan_motor_ar[pch[0]]
        omega = scan.get_motor_pos('om')
        print("peak at phi = %f" % (phi))
    scan_step = (scan_motor_ar[-1] - scan_motor_ar[0]) / (len(scan_motor_ar) - 1)
    delta = scan.get_motor_pos('del')
    chi = scan.get_motor_pos('chi')
    gamma = scan.get_motor_pos('gam')
    mu = scan.get_motor_pos('mu')
    energy = scan.get_motor_pos('fmbenergy')

    calibinfor = InformationFileIO(pathcalib)
    calibinfor.infor_reader()
    cch = calibinfor.get_para_value('direct_beam_position', section='Detector calibration')
    distance = calibinfor.get_para_value('detector_distance', section='Detector calibration')
    pixelsize = calibinfor.get_para_value('pixelsize', section='Detector calibration')
    det_rot = calibinfor.get_para_value('detector_rotation', section='Detector calibration')

    # Generate the paths for saving the data
    pathsave = scan.get_pathsave()
    pathinfor = os.path.join(pathsave, "scan_%04d_information.txt" % scan_num)

    # Load the detector images
    scan.write_fio()

    q_vector = cal_q_pos(pch[1:], [omega, delta, chi, phi, gamma, mu, energy], [distance, pixelsize, det_rot, cch])
    print(q_vector)

    leastsq_solution = least_squares(q_error_symmetric_peak, np.array([10.0, 10.0]), args=(q_vector, ))
    print('Find UB matrix?')
    print(leastsq_solution.success)

    rotation_matrix = R.from_euler('yxz', [leastsq_solution.x[0], 0, leastsq_solution.x[1]], degrees=True)
    additional_rotation_matrix = rotation_matrix.as_matrix()

    RSM_converter = RC2RSM_6C(scan_motor_ar, geometry,
                              omega, delta, chi, phi, gamma, mu, energy,
                              distance, pixelsize, det_rot, cch,
                              additional_rotation_matrix)

    # writing the scan information to the aimed file
    section_ar = ['General Information', 'Paths', 'Scan Information', 'Routine1: Reciprocal space map', 'Routine2: direct cutting']
    infor = InformationFileIO(pathinfor)
    infor.add_para('command', section_ar[0], scan.get_command())
    infor.add_para('year', section_ar[0], year)
    infor.add_para('beamtimeID', section_ar[0], beamtimeID)
    infor.add_para('p10_newfile', section_ar[0], p10_newfile)
    infor.add_para('scan_number', section_ar[0], scan_num)

    infor.add_para('path', section_ar[1], path)
    infor.add_para('pathsave', section_ar[1], pathsave)
    infor.add_para('pathinfor', section_ar[1], pathinfor)
    infor.add_para('pathmask', section_ar[1], pathmask)

    infor.add_para('geometry', section_ar[2], geometry)
    infor.add_para('roi', section_ar[2], list(roi))
    infor.add_para('peak_position', section_ar[2], list(pch))
    infor.add_para('scan_step', section_ar[2], scan_step)
    infor.add_para('omega', section_ar[2], omega)
    infor.add_para('delta', section_ar[2], delta)
    infor.add_para('chi', section_ar[2], chi)
    infor.add_para('phi', section_ar[2], phi)
    infor.add_para('gamma', section_ar[2], gamma)
    infor.add_para('mu', section_ar[2], mu)
    infor.add_para('energy', section_ar[2], scan.get_motor_pos('fmbenergy'))

    infor.add_para('direct_beam_position', section_ar[2], list(cch))
    infor.add_para('detector_distance', section_ar[2], distance)
    infor.add_para('pixelsize', section_ar[2], pixelsize)
    infor.add_para('det_rot', section_ar[2], det_rot)
    infor.add_para('detector', section_ar[2], detector)

    infor.infor_writer()

    # %% Perform the corresponding operations
    if 'Gif' in Functions_selected:
        print("")
        print("########################")
        print("Generating the gif image")
        print("########################")
        # save images into gif
        fig = plt.figure(figsize=(6, 6))
        plt.axis("off")
        img_frames = []
        for i in range(len(scan_motor_ar)):
            if i % 5 == 0:
                img = np.array(dataset[i, :, :], dtype=float)
                plt_im = plt.imshow(np.log10(img + 1.0), cmap="hot")
                img_frames.append([plt_im])
        gif_img = anim.ArtistAnimation(fig, img_frames)
        pathsavegif = os.path.join(pathsave, "scan%04d.gif" % scan_num)
        gif_img.save(pathsavegif, writer='pillow', fps=10)
        print('GIF image saved')
        plt.close()

    if 'Reciprocal_space_map' in Functions_selected:
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
        q_origin, new_shape, RSM_unit = RSM_converter.cal_q_range([(pch[1] - wxy[0]), (pch[1] + wxy[0]), (pch[2] - wxy[1]), (pch[2] + wxy[1])], rebinfactor=rebinfactor)

        # generate the 3D reciprocal space map
        print('Calculating intensity...')
        RSM_int = RSM_converter.RSM_conversion(dataset, new_shape, rebinfactor, cval=0, prefilter=use_prefilter)
        del dataset

        if save_full_3D_RSM:
            print('saving the full 3D RSM')
            filename = "%s_%05d_RSM.npz" % (p10_newfile, scan_num)
            pathsaveRSM = os.path.join(pathsave, filename)
            np.savez_compressed(pathsaveRSM, data=RSM_int)
            infor.add_para('pathRSM', section_ar[1], pathsaveRSM)

        # Cutting the three dimensional data with the center of mass in the center of cut intensity
        RSM_cut, qcen, RSM_bs = RSM_post_processing.Cut_central(RSM_int, RSM_bs, cut_mode=cut_central_pos)

        print("saving the RSM cut for pynx...")
        path3dRSM = os.path.join(pathtmp, "scan%04d.npz" % scan_num)
        np.savez_compressed(path3dRSM, data=RSM_cut)

        # load the mask and generate the new mask for the 3D reciprocal space map
        print('Calculating the mask...')
        RSM_mask = RSM_converter.RSM_conversion(mask3D, new_shape, rebinfactor, cval=1, prefilter=use_prefilter)
        del mask3D

        RSM_mask[RSM_mask >= 0.1] = 1
        RSM_mask[RSM_mask < 0.1] = 0
        RSM_cut_mask, qcen, RSM_bs = RSM_post_processing.Cut_central(RSM_mask, RSM_bs, cut_mode='given', peak_pos=qcen)

        print("saving the mask...")
        path3dmask = os.path.join(pathtmp, "scan%04d_mask.npz" % scan_num)
        np.savez_compressed(path3dmask, data=RSM_cut_mask)

        if generating_3D_vtk_file:
            filename = "scan%04d_diffraction_pattern.vti" % scan_num
            RSM_post_processing.RSM2vti(pathsave, RSM_int, filename, RSM_unit, origin=q_origin)

        # Generate the images of the reciprocal space map
        print('Generating the images of the RSM')
        pathsavetmp = os.path.join(pathsave, 'scan%04d_integrate' % scan_num + '_%s.png')
        RSM_post_processing.plot_with_units(RSM_int, q_origin, RSM_unit, pathsavetmp)
        pathsavetmp = os.path.join(pathsave, 'scan%04d_cut' % scan_num + '_%s.png')
        RSM_post_processing.plot_with_units(RSM_int, q_origin, RSM_unit, pathsavetmp, qmax=qcen)
        pathsavetmp = os.path.join(pathtmp, 'scan%04d' % scan_num + '_%s.png')
        RSM_post_processing.plot_without_units(RSM_cut, RSM_cut_mask, pathsavetmp)
        del RSM_int, RSM_mask

        q_cen_rsm = qcen * RSM_unit + q_origin

        # save the information
        infor.add_para('path3DRSM', section_ar[1], path3dRSM)
        infor.add_para('path3Dmask', section_ar[1], path3dmask)
        infor.add_para('use_prefilter', section_ar[1], use_prefilter)
        infor.add_para('roi_width', section_ar[3], list(wxy))
        infor.add_para('RSM_unit', section_ar[3], RSM_unit)
        infor.add_para('RSM_shape', section_ar[3], list(new_shape))
        infor.add_para('rebinfactor', section_ar[3], rebinfactor)
        infor.add_para('q_origin', section_ar[3], list(q_origin))
        infor.add_para('RSM_cut_central_mode', section_ar[3], cut_central_pos)
        infor.add_para('pynx_box_size', section_ar[3], RSM_bs)
        infor.add_para('RSM_q_center', section_ar[3], list(q_cen_rsm))
        infor.add_para('q_centeral_pixel', section_ar[3], list(qcen))
        infor.infor_writer()

    if ("2D_cuts" in Functions_selected) and ('Reciprocal_space_map' in Functions_selected):
        print("")
        print("##################")
        print("Generating the 2D cuts")
        print("##################")

        # Saving the intensity cuts for the 2D phase retrieval
        pathtmp = os.path.join(pathsave, "cutqz")
        if not os.path.exists(pathtmp):
            os.mkdir(pathtmp)
        pathtmp2 = os.path.join(pathtmp, "cutqz.npy")
        np.save(pathtmp2, RSM_cut[RSM_bs[0], :, :])
        infor.add_para('path_cutqz', '2D cuts', pathtmp)
        pathtmp2 = os.path.join(pathtmp, "cutqz_mask.npy")
        np.save(pathtmp2, RSM_cut_mask[RSM_bs[0], :, :])

        pathtmp = os.path.join(pathsave, "cutqy")
        if not os.path.exists(pathtmp):
            os.mkdir(pathtmp)
        pathtmp2 = os.path.join(pathtmp, "cutqy.npy")
        infor.add_para('path_cutqy', '2D cuts', pathtmp)
        np.save(pathtmp2, RSM_cut[:, RSM_bs[1], :])
        pathtmp2 = os.path.join(pathtmp, "cutqy_mask.npy")
        np.save(pathtmp2, RSM_cut_mask[:, RSM_bs[1], :])

        pathtmp = os.path.join(pathsave, "cutqx")
        if not os.path.exists(pathtmp):
            os.mkdir(pathtmp)
        pathtmp2 = os.path.join(pathtmp, "cutqx.npy")
        np.save(pathtmp2, RSM_cut[:, :, RSM_bs[2]])
        infor.add_para('path_cutqx', '2D cuts', pathtmp)
        pathtmp2 = os.path.join(pathtmp, "cutqx_mask.npy")
        np.save(pathtmp2, RSM_cut_mask[:, :, RSM_bs[2]])

    if 'Reciprocal_space_map' in Functions_selected:
        del RSM_cut
        del RSM_cut_mask

    end_time = time.time()
    infor.add_para('total_calculation_time', section_ar[0], end_time - start_time)
    infor.infor_writer()
    return


if __name__ == '__main__':
    BCDI_preparation()
