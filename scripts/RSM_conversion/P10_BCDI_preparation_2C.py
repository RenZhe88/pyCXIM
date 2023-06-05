#!/usr/local/bin/python2.7.3 -tttt

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import sys
import time
sys.path.append(r'E:\Work place 3\testprog\pyCXIM_master')
from pyCXIM.Common.Information_file_generator import InformationFileIO
from pyCXIM.p10_scan_reader.p10_eiger_reader import P10EigerScan
from pyCXIM.RSM.RC2RSM import RC2RSM_2C
import pyCXIM.RSM.RSM_post_processing as RSM_post_processing

def BCDI_preparation():
    start_time = time.time()
    # %% Inputs: select the functions of the code from the following mode
    # ['Gif', 'Direct_cut', 'Reciprocal_space_map', '2D_cuts']
    Functions_selected = ['Gif', 'Direct_cut', 'Reciprocal_space_map', '2D_cuts']

    # Inputs: general information
    year = "2021"
    beamtimeID = "11013318"
    p10_newfile = r"B12SYNS1P1"
    scan_num = 144
    detector = 'e4m'
    geometry = 'out_of_plane'
    # geometry = 'in_plane'

    qz_direction = 'surface direction'
    # qz_direction = 'diffraction vector direction'

    # Inputs: Detector parameters
    detector_distance = 1826.7286125322419
    pixelsize = 0.075
    # Direct beam position on the detector Y, X
    cch = [1039, 1341]
    # The half width of the detector roi in the order of [Y, X]
    wxy = [400, 400]
    # Roi on the detector [Ymin, Ymax, Xmin, Xmax]
    roi = [400, 1200, 1000, 1700]
    # Method to find the centeral position for the cut, please select from 'maximum intensity', 'maximum integration',  'weight_center'
    cut_central_pos = 'maximum integration'
    # Half size for the direct cut in pixels
    DC_bs = [120, 120, 120]

    # Half width of reciprocal space box size in pixels
    RSM_bs = [60, 60, 60]
    save_full_3D_RSM = False
    generating_3D_vtk_file = False

    # Inputs: Paths
    # the folder that stores the raw data of the beamtime
    path = r"E:\Data2\XRD raw\20211004 P10 BFO Pt\raw"
    # the aimed saving folder
    pathsavefolder = r"E:\Work place 3\sample\XRD\20211004 Inhouse PTO BFO Pt\Pt_islands"
    # the path for the mask file for the detector
    pathmask = r'E:\Work place 3\testprog\X-ray diffraction\Common functions\e4m_mask.npy'

    # %% Read the information and detector images of the scan
    print("#################")
    print("Basic information")
    print("#################")
    # reading the fio file
    scan = P10EigerScan(path, p10_newfile, scan_num, detector, pathsavefolder, pathmask)
    print(scan)
    energy = scan.get_motor_pos('fmbenergy')

    # Generate the paths for saving the data
    pathsave = scan.get_pathsave()
    pathinfor = os.path.join(pathsave, "scan_%04d_information.txt" % scan_num)

    # Load the detector images
    dataset, mask3D, pch, wxy = scan.eiger_load_images(roi, wxy, show_cen_image=(not os.path.exists(pathinfor)))
    scan.write_fio()

    # load the scan motors
    if geometry == 'out_of_plane':
        # read the omega values for each step in the rocking curve
        scan_motor_ar = scan.get_scan_data('om')
        # read the delta value
        two_theta = scan.get_motor_pos('del')
        if qz_direction == 'surface direction':
            scan_motor_offset = -0.5869580019812211
        elif qz_direction == 'diffraction vector direction':
            scan_motor_offset = two_theta / 2.0 - scan_motor_ar[int(pch[0])]
    elif geometry == 'in_plane':
        # read the phi values
        scan_motor_ar = scan.get_scan_data('phi')
        # read the delta value, which is gamma in the horizontal direction
        two_theta = scan.get_motor_pos('gam')
        if qz_direction == 'surface direction':
            scan_motor_offset = 0
        elif qz_direction == 'diffraction vector direction':
            scan_motor_offset = -two_theta / 2.0 - scan_motor_ar[int(pch[0])]

    scan_motor_ar = scan_motor_ar + scan_motor_offset
    # Finding the maximum peak position
    omega = scan_motor_ar[pch[0]]
    om_step = (scan_motor_ar[-1] - scan_motor_ar[0]) / (len(scan_motor_ar) - 1)
    print("peak at omega = %f" % (omega))

    RSM_converter = RC2RSM_2C(scan_motor_ar, two_theta, energy, detector_distance, pixelsize, geometry)

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

    infor.add_para('roi', section_ar[2], list(roi))
    infor.add_para('peak_position', section_ar[2], list(pch))
    infor.add_para('omega', section_ar[2], omega)
    infor.add_para('delta', section_ar[2], two_theta)
    infor.add_para('omegastep', section_ar[2], om_step)
    infor.add_para('omega_error', section_ar[2], scan_motor_offset)
    infor.add_para('direct_beam_position', section_ar[2], list(cch))
    infor.add_para('detector_distance', section_ar[2], detector_distance)
    infor.add_para('energy', section_ar[2], scan.get_motor_pos('fmbenergy'))
    infor.add_para('pixelsize', section_ar[2], pixelsize)
    infor.add_para('geometry', section_ar[2], geometry)
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

    if 'Direct_cut' in Functions_selected:
        print("")
        print("##########################################")
        print("Generating the stack of the detector image")
        print("##########################################")

        # Save the data
        pathtmp = os.path.join(pathsave, "pynxpre")
        if not os.path.exists(pathtmp):
            os.mkdir(pathtmp)
        pathtmp = os.path.join(pathtmp, "stacked_detector_images")
        if not os.path.exists(pathtmp):
            os.mkdir(pathtmp)

        if geometry == 'in_plane':
            DC_bs = np.array(DC_bs, dtype=int)
            DC_bs[[1, 2]] = DC_bs[[2, 1]]

        # cutting the stacked detector images
        Direct_cut, npch, DC_bs = RSM_post_processing.Cut_central(dataset, DC_bs, cut_mode=cut_central_pos)
        Direct_mask, npch, DC_bs = RSM_post_processing.Cut_central(mask3D, DC_bs, cut_mode='given', peak_pos=npch)

        if geometry == 'in_plane':
            Direct_cut = np.swapaxes(Direct_cut, 1, 2)
            Direct_mask = np.swapaxes(Direct_mask, 1, 2)
            DC_bs[[1, 2]] = DC_bs[[2, 1]]

        print("saving the data...")
        path_stacked = os.path.join(pathtmp, "scan%04d.npz" % scan_num)
        np.savez_compressed(path_stacked, data=Direct_cut)
        print('saving mask')
        path_stacked_mask = os.path.join(pathtmp, "scan%04d_mask.npz" % scan_num)
        np.savez_compressed(path_stacked_mask, data=Direct_mask)

        npch = npch + np.array([0, pch[1] - wxy[0], pch[2] - wxy[1]])
        print('Cutting at position' + str(npch))

        # ploting the ycut of the stacking images to estimate the quality of the image
        ycut = Direct_cut[:, :, DC_bs[2]]
        maskycut = Direct_mask[:, :, DC_bs[2]]
        plt.imshow(np.log10(ycut.T + 1.0), cmap="Blues")
        plt.imshow(np.ma.masked_where(maskycut == 0, maskycut).T, cmap="Reds", alpha=0.8, vmin=0.1, vmax=0.5)
        plt.xlabel('img_num')
        plt.ylabel('detector_Y')
        plt.savefig(os.path.join(pathtmp, 'ycut.png'))
        plt.show()
        plt.close()

        infor.add_para('path_stacked_detector_images', section_ar[1], path_stacked)
        infor.add_para('path_stacked_mask', section_ar[1], path_stacked_mask)
        infor.add_para('direct_cut_center_mode', section_ar[4], cut_central_pos)
        infor.add_para('direct_cut_box_size', section_ar[4], DC_bs)
        infor.add_para('direct_cut_centeral_pixel', section_ar[4], list(npch))
        infor.add_para('DC_unit', section_ar[4], RSM_converter.get_RSM_unit())
        infor.infor_writer()

    if ("2D_cuts" in Functions_selected) and ('Direct_cut' in Functions_selected):
        print("")
        print("##################")
        print("Generating the 2D cuts of the direct cut")
        print("##################")
        pathtmp = os.path.join(pathsave, "cuty")
        if not os.path.exists(pathtmp):
            os.mkdir(pathtmp)
        pathtmp2 = os.path.join(pathtmp, "cuty.npy")
        np.save(pathtmp2, ycut)
        infor.add_para('path_cuty', '2D cuts', pathtmp)
        pathtmp2 = os.path.join(pathtmp, "cuty_mask.npy")
        np.save(pathtmp2, maskycut)

    if 'Direct_cut' in Functions_selected:
        del Direct_cut
        del Direct_mask

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
        q_origin, new_shape, RSM_unit = RSM_converter.cal_q_range([(pch[1] - wxy[0]), (pch[1] + wxy[0]), (pch[2] - wxy[1]), (pch[2] + wxy[1])], cch, rebinfactor=rebinfactor)

        # generate the 3D reciprocal space map
        print('Calculating intensity...')
        RSM_int = RSM_converter.RSM_conversion(dataset, new_shape, rebinfactor, cval=0, prefilter=False)
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
        RSM_mask = RSM_converter.RSM_conversion(mask3D, new_shape, rebinfactor, cval=1, prefilter=False)
        del mask3D

        RSM_mask[RSM_mask >= 0.1] = 1
        RSM_mask[RSM_mask < 0.1] = 0
        RSM_cut_mask, qcen, RSM_bs = RSM_post_processing.Cut_central(RSM_mask, RSM_bs, cut_mode='given', peak_pos=qcen)

        print("saving the mask...")
        path3dmask = os.path.join(pathtmp, "scan%04d_mask.npz" % scan_num)
        np.savez_compressed(path3dmask, data=RSM_cut_mask)

        if generating_3D_vtk_file:
            filename = "scan%04d_diffraction_pattern.vti" % scan_num
            RSM_post_processing.RSM2vti(pathtmp, RSM_int, filename, RSM_unit, origin=q_origin)

        # Generate the images of the reciprocal space map
        print('Generating the images of the RSM')
        pathsavetmp = os.path.join(pathsave, 'scan%04d_integrate' % scan_num + '_%s.png')
        RSM_post_processing.plot_with_units(RSM_int, q_origin, RSM_unit, pathsavetmp)
        pathsavetmp = os.path.join(pathsave, 'scan%04d_cut' % scan_num + '_%s.png')
        RSM_post_processing.plot_with_units(RSM_int, q_origin, RSM_unit, pathsavetmp, qmax=qcen)
        pathsavetmp = os.path.join(pathtmp, 'scan%04d' % scan_num + '_%s.png')
        RSM_post_processing.plot_without_units(RSM_cut, RSM_cut_mask, pathsavetmp)
        del RSM_int, RSM_mask

        # save the information
        infor.add_para('path3DRSM', section_ar[1], path3dRSM)
        infor.add_para('path3Dmask', section_ar[1], path3dmask)
        infor.add_para('roi_width', section_ar[3], list(wxy))
        infor.add_para('RSM_unit', section_ar[3], RSM_unit)
        infor.add_para('RSM_shape', section_ar[3], list(new_shape))
        infor.add_para('rebinfactor', section_ar[3], rebinfactor)
        infor.add_para('q_origin', section_ar[3], list(q_origin))
        infor.add_para('RSM_cut_central_mode', section_ar[3], cut_central_pos)
        infor.add_para('pynx_box_size', section_ar[3], RSM_bs)
        infor.add_para('qcen', section_ar[3], list(qcen))
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
