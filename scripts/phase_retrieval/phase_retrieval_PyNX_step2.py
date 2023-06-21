# -*- coding: utf-8 -*-
"""
3D Phase retrieval performed with the schrink wrap method.

Input:
    path_scan_infor: the path of the information file generated for by the P10_BCDI_programs
    SeedNum: The number of random starting points, which would be averaged
    cut_selected: selected cuts used for phase retrieval
    algorithm: Similar algorithms as PyNX, here the method can be chosen from DIF, HIO, RAAR, ER, Sup
    HIO: hybrid input output format
    RAAR: Relaxed averaged alternating reflections
    ER: Error reduction
    DIF: difference map
    One typical algorithm: (DIF**50)**2*(HIO**50*Sup)**20*(DIF**50)**2*(RAAR**80*ER**10*Sup)**40
    3. paths:
        path defines the folder to read the images
        pathsave defines the folder to save the generated images and the infomation file
        path mask defines the path to load the prefined mask for the detector
    4. Detector parameters including the distance from the sample to the detector (distance), pixel_size of the detector (pixel_size), the direct beam position (cch), and the half width of the region of interest (wxy) 
    5. box size for the direct cut in pixels: The half width for the direct cut of the stacked detector images
    6. reciprocal space box size in pixels: The half width for the reciprocal space map

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import h5py
import time
sys.path.append(r'E:\Work place 3\testprog\pyCXIM_master')
from pyCXIM.Common.Information_file_generator import InformationFileIO
from pyCXIM.phase_retrieval.phase_retrieval_widget import PhaseRetrievalWidget

def plot_pynx_results():
    # %%Inputs
    pathsave = r'E:\Work place 3\sample\XRD\20201128 Longfei\B15_syn_S1_2_00024\pynxpre\reciprocal_space_map'
    data_description = 'reciprocal_space_map'
    # data_description='stacked_detector_images'
    trial_num = 5
    path_scan_infor = r"E:\Work place 3\sample\XRD\20211004 Inhouse PTO BFO Pt\Pt islands\B12SYNS1P1_00043\scan_0043_information.txt"
    pathPyNXfolder = r'E:\Work place 3\sample\XRD\20211004 Inhouse PTO BFO Pt\Pt islands\B12SYNS1P1_00043\pynxpre\reciprocal_space_map\PyNX Trial04'
    intensity_file = 'scan0024.npz'
    mask_file = 'scan0024_mask.npz'

    flip_condition = 'Support'
    # flip_condition = 'Phase'
    # flip_condition = 'Modulus'
    first_seed_flip = False
    further_analysis_selected = 10

    display_range = [400, 400, 400]
    # %%Load the information file
    print("Loading the information file...")
    if data_description == 'reciprocal_space_map':
        para_name_list = [
            'year', 'beamtimeID', 'scan_number', 'p10_newfile',
            'detector_distance', 'energy', 'pixelsize', 'RSM_unit']
    elif data_description == 'stacked_detector_images':
        para_name_list = [
            'year', 'beamtimeID', 'scan_number', 'p10_newfile', 'omega', 'delta',
            'omegastep', 'detector_distance', 'energy', 'pixelsize', 'DC_unit']

    # Loading the information file
    path_retrieval_infor = os.path.join(pathsave, "Phase_retrieval_information.txt")
    pr_infor = InformationFileIO(path_retrieval_infor)
    if not os.path.exists(path_retrieval_infor):
        if os.path.exists(path_scan_infor):
            scan_infor = InformationFileIO(path_scan_infor)
            pr_infor.copy_para_file(scan_infor, para_name_list, 'General Information')
        else:
            print('Could not find the desired scan parameter file! Generate the file with desired parameters!')
            pr_infor.gen_empty_para_file(para_name_list, 'General Information')
            assert False, 'Please fill the parameter file first, which is stored in the pathsave folder.'
    else:
        pr_infor.infor_reader()

    pr_file = PhaseRetrievalWidget(pathsave, trial_num, data_description, mode='w')
    print('Loading the measured intensity')
    
    pr_file.load_para_from_infor_file(path_retrieval_infor, para_name_list)

    zd, yd, xd = pr_file.get_para('data_shape')
    if data_description == 'reciprocal_space_map':
        unit = float(pr_infor.get_para_value('RSM_unit'))
    elif data_description == 'stacked_detector_images':
        unit = float(pr_infor.get_para_value('DC_unit'))


    filenames = []
    for filename in os.listdir(pathPyNXfolder):
        if filename[-4:] == '.cxi':
            filenames.append(filename)
    pathPRresult = os.path.join(pathPyNXfolder, filenames[0])
    tempimgfile = h5py.File(pathPRresult, "r")
    Para_group = tempimgfile['entry_last/image_1/process_1/configuration']
    pr_file.add_para('nb_run', len(filenames))
    para_name_list = [
        'rebin', 'nb_er', 'nb_hio', 'nb_raar',
        'roi_final', 'psf', 'support', 'RSM_unit']
    roi_final = pr_file.get_para('roi_final')
    pr_file.load_image_data(intensity_file, mask_file, roi_final)
    use_mask = True
    support_type = np.array(Para_group['support'])
    if np.array(Para_group['support']) == b'auto':
        support_smooth_width_begin = np.array(Para_group['support_smooth_width_begin'])
        support_smooth_width_end = np.array(Para_group['support_smooth_width_end'])
        support_smooth_width_relax_n = np.array(Para_group['support_smooth_width_end'])
        support_threshold_method = np.array(Para_group['support_smooth_width_relax_n'])
        support_update_period = np.array(Para_group['support_size'])
        thrpara = np.array(Para_group['support_threshold'])
        support_threshold_method = str(Para_group['support_threshold_method'])
        support_update = True
    else:
        support_update = False
    tempimgfile.close()





    imgfile.attrs['flip_condition'] = flip_condition
    imgfile.attrs['first_seed_flip'] = first_seed_flip
    imgfile.create_dataset("Input/intensity", data=image, dtype='f', chunks=(1, yd, xd), compression="gzip")
    imgfile.attrs['use_mask'] = use_mask
    imgfile.create_dataset("Input/mask", data=MaskFFT, dtype='f', chunks=(1, yd, xd), compression="gzip")
    imgfile.close()

    # %% Start the retrieval process
    imgfile = h5py.File(pathsaveimg, "r+")
    Solution_group = imgfile.create_group('Solutions')

    Modulus_sum = np.zeros_like(image)
    Phase_sum = np.zeros_like(image)
    Img_sum = np.zeros_like(image, dtype=complex)
    Support_sum = np.zeros_like(image)
    intensity_sum = np.zeros_like(image)
    err_ar = np.zeros((SeedNum, 6))

    for Seed in range(SeedNum):
        print('Seed %d' % Seed)
        pathPRresult = os.path.join(pathPyNXfolder, filenames[Seed])
        tempimgfile = h5py.File(pathPRresult, "r")
        img = np.array(tempimgfile['entry_last/image_1/data'], dtype=complex)
        support = np.array(tempimgfile['entry_last/image_1/support'], dtype=float)
        support, img = pp.CenterSup(support, img)
        support, img = pp.CenterSup(support, img)
        support, img = pp.CenterSup(support, img)
        err_ar[Seed, 0] = Seed
        err_ar[Seed, 1] = np.sum(support)
        err_ar[Seed, 2] = pp.get_Fourier_space_error(img, support, image, MaskFFT)
        err_ar[Seed, 3] = np.array(tempimgfile['/entry_last/image_1/process_1/results/llk_poisson'])
        err_ar[Seed, 4] = np.array(tempimgfile['/entry_last/image_1/process_1/results/free_llk_poisson'])
        intensity_sum = intensity_sum + np.square(np.abs(np.fft.fftshift(np.fft.fftn(img * support))))

        # removing the symmetrical cases by fliping the images
        Modulus_3D = np.abs(img)
        Phase_3D = pp.phase_corrector(np.angle(img), support)
        if flip_condition == 'Modulus':
            flip_con = (np.sum(Modulus_sum * Modulus_3D) < np.sum(Modulus_sum * np.flip(Modulus_3D)))
        elif flip_condition == 'Support':
            flip_con = (np.sum(Support_sum * support) < np.sum(Support_sum * np.flip(support)))
        elif flip_condition == 'Phase':
            flip_con = (np.sum(Phase_sum * Phase_3D) < np.sum(Phase_sum * -1.0 * np.flip(Phase_3D)))

        if Seed == 0 and first_seed_flip:
            flip_con = True

        if flip_con:
            err_ar[Seed, 5] = 1
            img = np.flip(img)
            img = np.conjugate(img)
            Modulus_3D = np.flip(Modulus_3D)
            Phase_3D = -1.0 * np.flip(Phase_3D)
            support = np.flip(support)
        else:
            err_ar[Seed, 5] = 0
        Seed_group = Solution_group.create_group("Seed%03d" % Seed)
        Seed_group.create_dataset("image", data=img, dtype='complex128', chunks=(1, yd, xd), compression="gzip")
        Seed_group.create_dataset("support", data=support, dtype='f', chunks=(1, yd, xd), compression="gzip")

        Support_sum = Support_sum + support
        Modulus_sum = Modulus_sum + Modulus_3D
        Phase_sum = Phase_sum + Phase_3D
        Img_sum = Img_sum + np.multiply(Modulus_3D, np.exp(1j * Phase_3D))
        if Seed < 10:
            plot_3D_result((support, Modulus_3D, Phase_3D), ('Support', 'Modulus', 'Phase'))

    Modulus_sum = Modulus_sum / SeedNum
    Phase_sum = Phase_sum / SeedNum
    Img_sum = Img_sum / SeedNum
    intensity_sum = intensity_sum / SeedNum
    Support_sum = Support_sum / SeedNum

    imgfile.create_dataset("Error/error", data=err_ar, dtype='float')
    imgfile['Error'].attrs['column_names'] = ['Seed', 'support_size', 'Fourier space error', 'Poisson Likelihood', 'Free_Likelihood', 'Flip']
    imgfile.create_dataset("Average_All/Support_sum", data=Support_sum, dtype='float', chunks=(1, yd, xd), compression="gzip")
    imgfile.create_dataset("Average_All/Modulus_sum", data=Modulus_sum, dtype='float', chunks=(1, yd, xd), compression="gzip")
    imgfile.create_dataset("Average_All/Phase_sum", data=Phase_sum, dtype='float', chunks=(1, yd, xd), compression="gzip")
    imgfile.create_dataset("Average_All/intensity_sum", data=intensity_sum, dtype='float', chunks=(1, yd, xd), compression="gzip")
    imgfile.create_dataset("Average_All/phase_retrieval_transfer_function", data=pp.cal_PRTF(image, Img_sum, MaskFFT), dtype='float')
    imgfile.close()

    # %% plot and save the final results
    unitz = np.arange(-zd / 2 + 0.5, zd / 2 + 0.5) * 2.0 * np.pi / zd / unit / 10.0
    unity = np.arange(-yd / 2 + 0.5, yd / 2 + 0.5) * 2.0 * np.pi / yd / unit / 10.0
    unitx = np.arange(-xd / 2 + 0.5, xd / 2 + 0.5) * 2.0 * np.pi / xd / unit / 10.0
    arrays_to_plot = (Support_sum, Modulus_sum / np.amax(Modulus_sum), Phase_sum)
    array_names = ('Average support', 'Average Amplitude', 'Average Phase')
    plot_3D_result2(arrays_to_plot, array_names, unitx, unity, unitz, display_range, pathsave, "Trial%d.png" % (trial_num))

    fig, axs = plt.subplots(3, 3, figsize=(24, 24))
    im = axs[0, 0].imshow(np.log10(intensity_sum[int(zd / 2), :, :] + 1.0), cmap="jet")
    plt.colorbar(im, ax=axs[0, 0], shrink=0.6)
    axs[0, 0].axis('off')
    axs[0, 0].set_title('Retrieved intensity Qz', fontsize=24)
    im = axs[0, 1].imshow(np.log10(intensity_sum[:, int(yd / 2), :] + 1.0), cmap="jet")
    plt.colorbar(im, ax=axs[0, 1], shrink=0.6)
    axs[0, 1].axis('off')
    axs[0, 1].set_title('Retrieved intensity Qy', fontsize=24)
    im = axs[0, 2].imshow(np.log10(intensity_sum[:, :, int(xd / 2)] + 1.0), cmap="jet")
    plt.colorbar(im, ax=axs[0, 2], shrink=0.6)
    axs[0, 2].axis('off')
    axs[0, 2].set_title('Retrieved intensity Qx', fontsize=24)
    im = axs[1, 0].imshow(np.log10(image[int(zd / 2), :, :] + 1.0), cmap="jet")
    plt.colorbar(im, ax=axs[1, 0], shrink=0.6)
    axs[1, 0].axis('off')
    axs[1, 0].set_title('Measured intensity Qz', fontsize=24)
    im = axs[1, 1].imshow(np.log10(image[:, int(yd / 2), :] + 1.0), cmap="jet")
    plt.colorbar(im, ax=axs[1, 1], shrink=0.6)
    axs[1, 1].axis('off')
    axs[1, 1].set_title('Measured intensity Qy', fontsize=24)
    im = axs[1, 2].imshow(np.log10(image[:, :, int(xd / 2)] + 1.0), cmap="jet")
    plt.colorbar(im, ax=axs[1, 2], shrink=0.6)
    axs[1, 2].axis('off')
    axs[1, 2].set_title('Measured intensity Qx', fontsize=24)
    im = axs[2, 0].imshow((intensity_sum - image)[int(zd / 2), :, :], cmap="jet", vmax=1.0e3, vmin=-1.0e3)
    plt.colorbar(im, ax=axs[2, 0], shrink=0.6)
    axs[2, 0].axis('off')
    axs[2, 0].set_title('Intensity difference Qz', fontsize=24)
    im = axs[2, 1].imshow((intensity_sum - image)[:, int(yd / 2), :], cmap="jet", vmax=1.0e3, vmin=-1.0e3)
    plt.colorbar(im, ax=axs[2, 1], shrink=0.6)
    axs[2, 1].axis('off')
    axs[2, 1].set_title('Intensity difference Qy', fontsize=24)
    im = axs[2, 2].imshow((intensity_sum - image)[:, :, int(xd / 2)], cmap="jet", vmax=1.0e3, vmin=-1.0e3)
    plt.colorbar(im, ax=axs[2, 2], shrink=0.6)
    axs[2, 2].axis('off')
    axs[2, 2].set_title('Intensity difference Qx', fontsize=24)
    fig.tight_layout()
    plt.savefig(os.path.join(pathsave, "Intensity_difference_Trial%d.png" % (trial_num)))
    plt.close()

    if support_update:
        fig, axs = plt.subplots(2, 2, figsize=(16, 16))
        axs[0, 0].plot(err_ar[:, 1], err_ar[:, 2], 'r.')
        axs[0, 0].set_title('Fourier space error', fontsize=24)
        axs[0, 0].set_xlabel('total support pixel', fontsize=24)
        axs[0, 0].set_ylabel('Fourier space error', fontsize=24)
        axs[0, 1].plot(err_ar[:, 1], err_ar[:, 3], 'r.')
        axs[0, 1].set_title('Log likelihood', fontsize=24)
        axs[0, 1].set_xlabel('total support pixel', fontsize=24)
        axs[0, 1].set_ylabel('Log likelihood', fontsize=24)
        axs[1, 0].plot(err_ar[:, 1], err_ar[:, 4], 'r.')
        axs[1, 0].set_title('Free Log likelihood', fontsize=24)
        axs[1, 0].set_xlabel('total support pixel', fontsize=24)
        axs[1, 0].set_ylabel('Free Log likelihood', fontsize=24)
        axs[1, 1].plot(pp.cal_PRTF(image, Img_sum, MaskFFT), 'r.')
        axs[1, 1].set_title('Phase retrieval transfer function', fontsize=24)
        axs[1, 1].set_xlabel('pixel', fontsize=24)
        axs[1, 1].set_ylabel('PRTF', fontsize=24)
        fig.tight_layout()
        plt.savefig(os.path.join(pathsave, "Error_Trial%d.png" % (trial_num)))
        plt.close()
    else:
        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        axs[0, 0].hist(err_ar[:, 2], bins=20, histtype='step')
        axs[0, 0].set_title('Fourier space error', fontsize=24)
        axs[0, 0].set_xlabel('Fourier space error', fontsize=24)
        axs[0, 0].set_ylabel('Num of solutions', fontsize=24)
        axs[0, 1].hist(err_ar[:, 3], bins=20, histtype='step')
        axs[0, 1].set_title('Log likelihood', fontsize=24)
        axs[0, 1].set_xlabel('Log likelihood', fontsize=24)
        axs[0, 1].set_ylabel('Num of solutions', fontsize=24)
        axs[1, 0].hist(err_ar[:, 4], bins=5, histtype='step')
        axs[1, 0].set_title('Free Log likelihood', fontsize=24)
        axs[1, 0].set_xlabel('Free Log likelihood', fontsize=24)
        axs[1, 0].set_ylabel('Num of solutions', fontsize=24)
        axs[1, 1].plot(pp.cal_PRTF(image, Img_sum, MaskFFT), 'r.')
        axs[1, 1].set_title('Phase retrieval transfer function', fontsize=24)
        axs[1, 1].set_xlabel('pixel', fontsize=24)
        axs[1, 1].set_ylabel('PRTF', fontsize=24)
        fig.tight_layout()
        plt.savefig(os.path.join(pathsave, "Error_Trial%d.png" % (trial_num)))
        plt.close()

    pathsavevti = os.path.join(pathsave, "Trial%02d.vti" % (trial_num))
    voxel_size = ((2.0 * np.pi / zd / unit / 10.0), (2.0 * np.pi / yd / unit / 10.0), (2.0 * np.pi / xd / unit / 10.0))
    pp.save_to_vti(pathsavevti, (Modulus_sum / np.amax(Modulus_sum), Phase_sum, Support_sum), ('Modulus', 'Phase', 'Support'), voxel_size=voxel_size)

    # %% Transforming into Orthoganol coordinates
    if method_selected == 'stacked_detector_images':
        Ortho_support_sum, Ortho_unit = pp.Orth3D(Support_sum, omega, omegastep, delta, distance, pixelsize, energy)
        Ortho_modulus_sum, Ortho_unit = pp.Orth3D(Modulus_sum, omega, omegastep, delta, distance, pixelsize, energy)
        Ortho_phase_sum, Ortho_unit = pp.Orth3D(Phase_sum, omega, omegastep, delta, distance, pixelsize, energy)

        nz, ny, nx = Ortho_support_sum.shape
        unitz = np.arange(-nz / 2 + 0.5, nz / 2 + 0.5) * Ortho_unit
        unity = np.arange(-ny / 2 + 0.5, ny / 2 + 0.5) * Ortho_unit
        unitx = np.arange(-nx / 2 + 0.5, nx / 2 + 0.5) * Ortho_unit
        arrays_to_plot = (Ortho_support_sum, Ortho_modulus_sum / np.amax(Ortho_modulus_sum), Ortho_phase_sum)
        array_names = ('Average support', 'Average Amplitude', 'Average Phase')
        plot_3D_result2(arrays_to_plot, array_names, unitx, unity, unitz, display_range, pathsave, "Trial%d_orthonormalized.png" % (trial_num))

        pathsavevti = os.path.join(pathsave, "Trial%02d_ortho.vti" % trial_num)
        voxel_size = (Ortho_unit, Ortho_unit, Ortho_unit)
        pp.save_to_vti(pathsavevti, (Ortho_modulus_sum / np.amax(Ortho_modulus_sum), Ortho_phase_sum, Ortho_support_sum), ('Modulus', 'Phase', 'Support'), voxel_size=voxel_size)
        imgfile = h5py.File(pathsaveimg, "r+")
        imgfile.create_dataset("Ortho/Ortho_support_sum", data=Ortho_support_sum, dtype='f', chunks=(1, ny, nx), compression="gzip")
        imgfile.create_dataset("Ortho/Ortho_modulus_sum", data=Ortho_modulus_sum, dtype='f', chunks=(1, ny, nx), compression="gzip")
        imgfile.create_dataset("Ortho/Ortho_phase_sum", data=Ortho_phase_sum, dtype='f', chunks=(1, ny, nx), compression="gzip")
        imgfile.close()

    # %% Cleaning up the memory
    del Support_sum
    del Modulus_sum
    del Phase_sum
    del Img_sum
    del intensity_sum
    if method_selected == 'stacked_detector_images':
        del Ortho_support_sum
        del Ortho_modulus_sum
        del Ortho_phase_sum

    # %% select results for SVD analysis or averaging
    selected_image_num = int(further_analysis_selected / 100.0 * SeedNum)
    if (not support_update) and selected_image_num >= 3:
        further_analysis_method = 'SVD'
        if first_seed_flip:
            support = np.flip(support)

        print("%d images selected for SVD analysis" % selected_image_num)
        select_Modulus_sum, select_Phase_sum, Mode1_Modulus, Mode1_Phase, Mode2_Modulus, Mode2_Phase, Mode3_Modulus, Mode3_Phase, evalue = pp.svd_analysis(pathsaveimg, support, selected_image_num, err_ar[:, 2])
        unitz = np.arange(-zd / 2 + 0.5, zd / 2 + 0.5) * 2.0 * np.pi / zd / unit / 10.0
        unity = np.arange(-yd / 2 + 0.5, yd / 2 + 0.5) * 2.0 * np.pi / yd / unit / 10.0
        unitx = np.arange(-xd / 2 + 0.5, xd / 2 + 0.5) * 2.0 * np.pi / xd / unit / 10.0
        arrays_to_plot = (select_Modulus_sum / np.amax(select_Modulus_sum), select_Phase_sum, Mode1_Modulus, Mode1_Phase)
        array_names = ('Average Modulus', 'Average Phase', 'Mode1 %.2f%% Modulus' % (evalue[0] * 100.0), 'Mode1 %.2f%% Phase' % (evalue[0] * 100.0))
        plot_3D_result2(arrays_to_plot, array_names, unitx, unity, unitz, display_range, pathsave, "Trial%02d_svd.png" % trial_num)

        imgfile = h5py.File(pathsaveimg, "r+")
        imgfile.attrs['further_analysis_selected'] = further_analysis_selected
        imgfile.attrs['selected_image_num'] = selected_image_num
        imgfile.attrs['further_analysis_method'] = further_analysis_method
        imgfile.create_dataset("SVD_analysis/select_Modulus_sum", data=select_Modulus_sum, dtype='float', chunks=(1, ny, nx), compression="gzip")
        imgfile.create_dataset("SVD_analysis/select_Phase_sum", data=select_Phase_sum, dtype='float', chunks=(1, ny, nx), compression="gzip")
        imgfile.create_dataset("SVD_analysis/Mode1_Modulus", data=Mode1_Modulus, dtype='float', chunks=(1, ny, nx), compression="gzip")
        imgfile.create_dataset("SVD_analysis/Mode1_Phase", data=Mode1_Phase, dtype='float', chunks=(1, ny, nx), compression="gzip")
        imgfile.create_dataset("SVD_analysis/Mode2_Modulus", data=Mode2_Modulus, dtype='float', chunks=(1, ny, nx), compression="gzip")
        imgfile.create_dataset("SVD_analysis/Mode2_Phase", data=Mode2_Phase, dtype='float', chunks=(1, ny, nx), compression="gzip")
        imgfile.create_dataset("SVD_analysis/Mode3_Modulus", data=Mode3_Modulus, dtype='float', chunks=(1, ny, nx), compression="gzip")
        imgfile.create_dataset("SVD_analysis/Mode3_Phase", data=Mode3_Phase, dtype='float', chunks=(1, ny, nx), compression="gzip")
        imgfile.create_dataset("SVD_analysis/evalue", data=evalue, dtype='float', compression="gzip")
        imgfile.close()

        pathsaveresult = os.path.join(pathsave, "Trial%02d_svd_average.vti" % trial_num)
        pp.save_to_vti(pathsaveresult, (select_Modulus_sum / np.amax(select_Modulus_sum), select_Phase_sum, support), ('Modulus', 'Phase', 'Support'), voxel_size=voxel_size)
        pathsaveresult = os.path.join(pathsave, "Trial%02d_mode1.vti" % trial_num)
        pp.save_to_vti(pathsaveresult, (Mode1_Modulus / np.amax(Mode1_Modulus), Mode1_Phase, support), ('Modulus', 'Phase', 'Support'), voxel_size=voxel_size)
    else:
        further_analysis_method = 'Average'
        further_analysis_method = 'Average'
        select_Modulus_sum, select_Phase_sum, select_Support_sum = pp.selected_average(pathsaveimg, selected_image_num, err_ar[:, 2])

        unitz = np.arange(-zd / 2 + 0.5, zd / 2 + 0.5) * 2.0 * np.pi / zd / unit / 10.0
        unity = np.arange(-yd / 2 + 0.5, yd / 2 + 0.5) * 2.0 * np.pi / yd / unit / 10.0
        unitx = np.arange(-xd / 2 + 0.5, xd / 2 + 0.5) * 2.0 * np.pi / xd / unit / 10.0
        arrays_to_plot = (select_Support_sum, select_Modulus_sum / np.amax(select_Modulus_sum), select_Phase_sum)
        array_names = ('Average Support', 'Average Modulus', 'Average Phase')
        plot_3D_result2(arrays_to_plot, array_names, unitx, unity, unitz, display_range, pathsave, "Trial%02d_selected_average.png" % trial_num)

        imgfile = h5py.File(pathsaveimg, "r+")
        imgfile.attrs['further_analysis_selected'] = further_analysis_selected
        imgfile.attrs['selected_image_num'] = selected_image_num
        imgfile.attrs['further_analysis_method'] = further_analysis_method
        imgfile.create_dataset("Selected_average/select_Modulus_sum", data=select_Modulus_sum, dtype='float', compression="gzip")
        imgfile.create_dataset("Selected_average/select_Phase_sum", data=select_Phase_sum, dtype='float', compression="gzip")
        imgfile.create_dataset("Selected_average/select_Support_sum", data=select_Support_sum, dtype='float', chunks=(1, yd, xd), compression="gzip")
        imgfile.close()

        pathsaveresult = os.path.join(pathsave, "Trial%02d_selected_average.vti" % trial_num)
        pp.save_to_vti(pathsaveresult, (select_Modulus_sum / np.amax(select_Modulus_sum), select_Phase_sum, select_Support_sum), ('Modulus', 'Phase', 'Support'), voxel_size=voxel_size)

    # %% Transforming into Orthoganol coordinates
    if method_selected == 'stacked_detector_images':
        if further_analysis_method == 'SVD':
            Ortho_support, Ortho_unit = pp.Orth3D(support, omega, omegastep, delta, distance, pixelsize, energy)
            Ortho_modulus_sum, Ortho_unit = pp.Orth3D(select_Modulus_sum, omega, omegastep, delta, distance, pixelsize, energy)
            Ortho_phase_sum, Ortho_unit = pp.Orth3D(select_Phase_sum, omega, omegastep, delta, distance, pixelsize, energy)

            Ortho_Mode1_Modulus, Ortho_unit = pp.Orth3D(Mode1_Modulus, omega, omegastep, delta, distance, pixelsize, energy)
            Ortho_Mode1_Phase, Ortho_unit = pp.Orth3D(Mode1_Phase, omega, omegastep, delta, distance, pixelsize, energy)
            Ortho_Mode2_Modulus, Ortho_unit = pp.Orth3D(Mode2_Modulus, omega, omegastep, delta, distance, pixelsize, energy)
            Ortho_Mode2_Phase, Ortho_unit = pp.Orth3D(Mode2_Phase, omega, omegastep, delta, distance, pixelsize, energy)
            Ortho_Mode3_Modulus, Ortho_unit = pp.Orth3D(Mode3_Modulus, omega, omegastep, delta, distance, pixelsize, energy)
            Ortho_Mode3_Phase, Ortho_unit = pp.Orth3D(Mode3_Phase, omega, omegastep, delta, distance, pixelsize, energy)

            nz, ny, nx = Ortho_support.shape
            unitz = np.arange(-nz / 2 + 0.5, nz / 2 + 0.5) * Ortho_unit
            unity = np.arange(-ny / 2 + 0.5, ny / 2 + 0.5) * Ortho_unit
            unitx = np.arange(-nx / 2 + 0.5, nx / 2 + 0.5) * Ortho_unit
            arrays_to_plot = (Ortho_support, Ortho_modulus_sum / np.amax(Ortho_modulus_sum), Ortho_phase_sum)
            array_names = ('Support', 'Average Amplitude', 'Average Phase')
            plot_3D_result2(arrays_to_plot, array_names, unitx, unity, unitz, display_range, pathsave, "Trial%d_ortho_svd_average.png" % (trial_num))
            arrays_to_plot = (Ortho_support, Ortho_Mode1_Modulus / np.amax(Ortho_Mode1_Modulus), Ortho_Mode1_Phase)
            array_names = ('Support', 'Mode1 %.2f%% Modulus' % (evalue[0] * 100.0), 'Mode1 %.2f%% Phase' % (evalue[0] * 100.0))
            plot_3D_result2(arrays_to_plot, array_names, unitx, unity, unitz, display_range, pathsave, "Trial%d_ortho_svd_mode1.png" % (trial_num))

            imgfile = h5py.File(pathsaveimg, "r+")
            imgfile.create_dataset("Ortho/Ortho_select_Modulus_sum", data=Ortho_modulus_sum, dtype='f', chunks=(1, ny, nx), compression="gzip")
            imgfile.create_dataset("Ortho/Ortho_select_Phase_sum", data=Ortho_phase_sum, dtype='f', chunks=(1, ny, nx), compression="gzip")
            imgfile.create_dataset("Ortho/Ortho_Mode1_Modulus", data=Ortho_Mode1_Modulus, dtype='f', chunks=(1, ny, nx), compression="gzip")
            imgfile.create_dataset("Ortho/Ortho_Mode1_Phase", data=Ortho_Mode1_Phase, dtype='f', chunks=(1, ny, nx), compression="gzip")
            imgfile.create_dataset("Ortho/Ortho_Mode2_Modulus", data=Ortho_Mode2_Modulus, dtype='f', chunks=(1, ny, nx), compression="gzip")
            imgfile.create_dataset("Ortho/Ortho_Mode2_Phase", data=Ortho_Mode2_Phase, dtype='f', chunks=(1, ny, nx), compression="gzip")
            imgfile.create_dataset("Ortho/Ortho_Mode3_Modulus", data=Ortho_Mode3_Modulus, dtype='f', chunks=(1, ny, nx), compression="gzip")
            imgfile.create_dataset("Ortho/Ortho_Mode3_Phase", data=Ortho_Mode3_Phase, dtype='f', chunks=(1, ny, nx), compression="gzip")
            imgfile.close()

            voxel_size = (Ortho_unit, Ortho_unit, Ortho_unit)
            pathsaveresult = os.path.join(pathsave, "Trial%02d_svd_ortho_average.vti" % trial_num)
            pp.save_to_vti(pathsaveresult, (Ortho_modulus_sum / np.amax(Ortho_modulus_sum), Ortho_phase_sum, Ortho_support), ('Modulus', 'Phase', 'Support'), voxel_size=voxel_size)
            pathsaveresult = os.path.join(pathsave, "Trial%02d_ortho_mode1.vti" % trial_num)
            pp.save_to_vti(pathsaveresult, (Ortho_Mode1_Modulus / np.amax(Ortho_Mode1_Modulus), Ortho_Mode1_Phase, Ortho_support), ('Modulus', 'Phase', 'Support'), voxel_size=voxel_size)
        elif further_analysis_method == 'Average':
            Ortho_support, Ortho_unit = pp.Orth3D(support, omega, omegastep, delta, distance, pixelsize, energy)
            Ortho_modulus_sum, Ortho_unit = pp.Orth3D(select_Modulus_sum, omega, omegastep, delta, distance, pixelsize, energy)
            Ortho_phase_sum, Ortho_unit = pp.Orth3D(select_Phase_sum, omega, omegastep, delta, distance, pixelsize, energy)

            nz, ny, nx = Ortho_support.shape
            unitz = np.arange(-nz / 2 + 0.5, nz / 2 + 0.5) * Ortho_unit
            unity = np.arange(-ny / 2 + 0.5, ny / 2 + 0.5) * Ortho_unit
            unitx = np.arange(-nx / 2 + 0.5, nx / 2 + 0.5) * Ortho_unit
            arrays_to_plot = (Ortho_support, Ortho_modulus_sum / np.amax(Ortho_modulus_sum), Ortho_phase_sum)
            array_names = ('Support', 'Average Amplitude', 'Average Phase')
            plot_3D_result2(arrays_to_plot, array_names, unitx, unity, unitz, display_range, pathsave, "Trial%d_ortho_selected_average.png" % (trial_num))

            imgfile = h5py.File(pathsaveimg, "r+")
            imgfile.create_dataset("Ortho/Ortho_select_Modulus_sum", data=Ortho_modulus_sum, dtype='f', compression="gzip")
            imgfile.create_dataset("Ortho/Ortho_select_Phase_sum", data=Ortho_phase_sum, dtype='f', compression="gzip")
            imgfile.close()

            voxel_size = (Ortho_unit, Ortho_unit, Ortho_unit)
            pathsaveresult = os.path.join(pathsave, "Trial%02d_selected_ortho_average.vti" % trial_num)
            pp.save_to_vti(pathsaveresult, (Ortho_modulus_sum / np.amax(Ortho_modulus_sum), Ortho_phase_sum, Ortho_support), ('Modulus', 'Phase', 'Support'), voxel_size=voxel_size)

    # %% Cleaning up the memory
    del support
    del select_Modulus_sum
    del select_Phase_sum
    if further_analysis_method == 'SVD':
        del Mode1_Modulus
        del Mode1_Phase
        del Mode2_Modulus
        del Mode2_Phase
        del Mode3_Modulus
        del Mode3_Phase
    if method_selected == 'stacked_detector_images':
        del Ortho_modulus_sum
        del Ortho_phase_sum
        del Ortho_support
        if further_analysis_method == 'SVD':
            del Ortho_Mode1_Modulus
            del Ortho_Mode1_Phase
            del Ortho_Mode2_Modulus
            del Ortho_Mode2_Phase
            del Ortho_Mode3_Modulus
            del Ortho_Mode3_Phase

    # %% Writing the information
    # save the Information for the Phase retrieval
    section = 'General Information'
    pr_infor.add_para('year', section, year)
    pr_infor.add_para('beamtimeID', section, beamtimeID)
    pr_infor.add_para('scan', section, scan)
    pr_infor.add_para('p10_newfile', section, p10_newfile)
    pr_infor.add_para('method_selected', section, method_selected)
    pr_infor.add_para('omega', section, omega)
    pr_infor.add_para('delta', section, delta)
    pr_infor.add_para('omegastep', section, omegastep)
    pr_infor.add_para('unit', section, unit)
    pr_infor.add_para('distance', section, distance)
    pr_infor.add_para('energy', section, energy)
    pr_infor.add_para('pixelsize', section, pixelsize)
    pr_infor.add_para('pathread', section, pathread)
    pr_infor.add_para('pathsave', section, pathsave)
    pr_infor.add_para('pathmask', section, pathmask)
    pr_infor.add_para('total_trial_num', section, trial_num)

    section = 'Trial %02d' % trial_num
    pr_infor.add_para('pathresult', section, pathsaveimg)
    pr_infor.add_para('data_shape', section, image.shape)
    pr_infor.add_para('use_mask', section, os.path.exists(pathmask))
    pr_infor.add_para('nb_run', section, SeedNum)
    pr_infor.add_para('voxel_size', section, list(voxel_size))

    pr_infor.add_para('rebin', section, rebin)
    pr_infor.add_para('roi_final', section, roi_final)
    pr_infor.add_para('nb_er', section, nb_er)
    pr_infor.add_para('nb_hio', section, nb_hio)
    pr_infor.add_para('nb_raar', section, nb_raar)
    pr_infor.add_para('support_type', section, support_type)
    pr_infor.add_para('nb_raar', section, nb_raar)

    pr_infor.add_para('flip_condition', section, flip_condition)
    pr_infor.add_para('first_seed_flip', section, first_seed_flip)

    if support_update:
        pr_infor.add_para('support_type', section, support_type)
        pr_infor.add_para('support_smooth_width_begin', section, support_smooth_width_begin)
        pr_infor.add_para('support_smooth_width_end', section, support_smooth_width_end)
        pr_infor.add_para('support_threshold_method', section, support_threshold_method)
        pr_infor.add_para('support_threshold', section, thrpara)
    pr_infor.infor_writer()
    return


if __name__ == '__main__':
    plot_pynx_results()
