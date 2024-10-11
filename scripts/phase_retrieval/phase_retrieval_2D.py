# -*- coding: utf-8 -*-
"""
2D Phase retrieval.

Input:
    path_scan_infor: the path of the information file generated for by the P10_BCDI_programs
    SeedNum: The number of random starting points, which would be averaged
    cut_selected: selected cuts used for phase retrieval
    algorithm: Similar algorithms as PyNX, here the method can be chosen from DIF, HIO, RAAR, ER, Sup
    HIO: hybrid input output format
    RAAR: Relaxed averaged alternating reflections
    ER: Error reduction
    DIF: difference map
    One typical algorithm: (HIO**50*Sup)**60*(RAAR**80*Sup)**10
    3. paths:
        path defines the folder to read the images
        pathsave defines the folder to save the generated images and the infomation file
        path mask defines the path to load the prefined mask for the detector
    4. Detector parameters including the distance from the sample to the detector (distance), pixel_size of the detector (pixel_size), the direct beam position (cch), and the half width of the region of interest (wxy) 
    5. box size for the direct cut in pixels: The half width for the direct cut of the stacked detector images
    6. reciprocal space box size in pixels: The half width for the reciprocal space map
Created on Fri May 12 14:39:49 2023

@author: Ren Zhe
@email: renzhe@ihep.ac.cn
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import h5py
import time
sys.path.append(r'F:\Work place 3\testprog\pyCXIM_master')
from pyCXIM.Common.Information_file_generator import InformationFileIO
from pyCXIM.phase_retrieval.phase_retrieval_widget import PhaseRetrievalWidget

def phase_retrieval_2D(scan_num):
    # %%Input
    starting_time = time.time()
    path_scan_infor = r"F:\Work place 4\sample\XRD\20240602_BFO_chiral_P10_Desy\Polarized_BFO2\polarized_BFO_%05d\scan_%04d_information.txt" % (scan_num, scan_num)
    SeedNum = 100
    # For 2D images the data description can be 'cutqz', 'cutqy', 'cutqx', 'cuty'
    data_description = 'cutqz'
    pathsave = r'F:\Work place 4\sample\XRD\20240602_BFO_chiral_P10_Desy\Polarized_BFO2\polarized_BFO_%05d\cutqz' % scan_num
    intensity_file = "%s.npy" % data_description
    mask_file = "%s_mask.npy" % data_description

    precision = '64'
    algorithm = "(HIO**50*Sup)**20*(DIF**50)**2*(RAAR**80*ER**10*Sup)**40"
    # algorithm = "DIF**200*DETWIN*(RAAR**50*ER**10)**40"

    # Input: parameters for creating the initial suppport.
    # Please chose from 'auto_correlation', 'import', 'average', 'support_selected', or 'modulus_selected'
    support_type = 'auto_correlation'
    support_from_trial = 0

    # If support_type is 'auto_correlation'
    auto_corr_thrpara = 0.008
    # If support_type is 'average', 'support_selected', or'modulus_selected'
    Initial_support_threshold = 0.8
    # If support_type is 'support_selected' or 'modulus_selected'
    percent_selected = 10
    # If support_type is 'modulus_selected'
    modulus_smooth_width = 0.5
    # If support_type is 'import'
    path_import_initial_support = r'F:\Work place 4\sample\XRD\High strain test\20211004_Pt_islands_Stephane\B12SYNS1P1_00144\cutqz\support.npz'

    # Input: starting image inherented from trial
    start_trial_num = 0

    # Input: parameters for the free Log likelihood
    Free_LLK = False
    FLLK_percentage = 0.01
    FLLK_radius = 3

    # Input: parameters for the shrink wrap loop
    # threhold_update_method = 'random'
    threhold_update_method = 'exp_increase'
    # threhold_update_method = 'lin_increase'
    support_para_update_precent = 0.8
    thrpara_min = 0.08
    thrpara_max = 0.10
    support_smooth_width_begin = 3.5
    support_smooth_width_end = 1.0
    hybrid_para_begin = 0.0
    hybrid_para_end = 0.0

    # Input: parameters for the detwin operation
    detwin_axis = 0

    # Input: parameters for flipping the images to remove the trival solutions.
    # flip_condition = 'Support'
    flip_condition = 'Phase'
    # flip_condition ='Modulus'
    first_seed_flip = True
    phase_unwrap_method = 0

    # Input: Parameters for further analysis like SVD and average
    further_analysis_selected = 10
    error_type_for_selection = 'Fourier space error'

    # Input: Parameters determining the display of the images
    display_range = [500, 500]
    display_image_num = 10
    # %% Load the image data and the mask

    # Loading the intensity and the mask defining the dead pixels
    print("Loading the information file...")

    if data_description in ['cutqx', 'cutqy', 'cutqz']:
        para_name_list = [
            'year', 'beamtimeID', 'scan_number', 'p10_newfile',
            'detector_distance', 'energy', 'pixelsize', 'unit']
    elif data_description == 'cuty':
        para_name_list = [
            'year', 'beamtimeID', 'scan_number', 'p10_newfile', 'omega', 'delta',
            'omegastep', 'detector_distance', 'energy', 'pixelsize', 'unit',
            'direct_cut_box_size']

    path_retrieval_infor = os.path.join(pathsave, "Phase_retrieval_information.txt")
    pr_infor = InformationFileIO(path_retrieval_infor)
    if not os.path.exists(path_retrieval_infor):
        trial_num = 1
        start_trial_num = 0
        support_type = 'auto_correlation'
        if os.path.exists(path_scan_infor):
            scan_infor = InformationFileIO(path_scan_infor)
            pr_infor.add_para('total_trial_num', 'General Information', 0)
            pr_infor.copy_para_values(scan_infor, para_name_list, 'General Information')
            if data_description in ['cutqx', 'cutqy', 'cutqz']:
                pr_infor.copy_para_values(scan_infor, ['RSM_unit'], 'General Information', ['unit'])
            elif data_description == 'cuty':
                pr_infor.copy_para_values(scan_infor, ['DC_unit'], 'General Information', ['unit'])
        else:
            print('Could not find the desired scan parameter file! Generate the file with desired parameters!')
            pr_infor.gen_empty_para_file(para_name_list, 'General Information')
            assert False, 'Please fill the parameter file first, which is stored in the pathsave folder.'
    else:
        pr_infor.infor_reader()
        trial_num = pr_infor.get_para_value('total_trial_num') + 1

    pr_file = PhaseRetrievalWidget(pathsave, trial_num, data_description, mode='w')
    pr_file.load_image_data(intensity_file, mask_file)
    pr_file.load_para_from_infor_file(path_retrieval_infor, para_name_list)
    yd, xd = pr_file.get_para('data_shape')

    # %%Load information file and support
    pr_file.create_initial_support(support_type, auto_corr_thrpara, support_from_trial,
                                   Initial_support_threshold, percent_selected, modulus_smooth_width,
                                   path_import_initial_support)

    # %% Start the retrieval process
    pr_file.phase_retrieval_main(algorithm, SeedNum, start_trial_num, precision, Free_LLK,
                                 FLLK_percentage, FLLK_radius, threhold_update_method,
                                 support_para_update_precent, thrpara_min, thrpara_max,
                                 support_smooth_width_begin, support_smooth_width_end,
                                 hybrid_para_begin, hybrid_para_end, detwin_axis,
                                 flip_condition, first_seed_flip, phase_unwrap_method,
                                 display_image_num)

    # %% plot and save the final result
    array_names = ('Modulus_sum', 'Phase_sum', 'Support_sum')
    pr_file.analysis_and_plot_2D('Average_All', array_names,
                                 title='Average results of %d runs' % pr_file.get_para('nb_run'),
                                 filename="Trial%d" % (trial_num), save_image=True,
                                 subplot_config=None, display_range=display_range)
    pr_file.plot_2D_intensity(array_group='Average_All', save_image=True, filename="Intensity_difference_Trial%d.png" % (trial_num))

    # %% select results for SVD analysis or averaging
    pr_file.further_analysis(further_analysis_selected, error_type=error_type_for_selection)
    array_names = ('Modulus_sum', 'Phase_sum', 'Support_sum')
    pr_file.analysis_and_plot_2D('Selected_average', array_names,
                                 title='Average results of %d runs with minimum error' % pr_file.get_para('further_analysis_selected'),
                                 filename="Trial%02d_selected_average" % trial_num, save_image=True,
                                 subplot_config=None, display_range=display_range)
    if pr_file.get_para('further_analysis_method') == 'SVD':
        evalue = pr_file.get_dataset("SVD_analysis/evalue")
        array_names = ('Mode1_Modulus', 'Mode1_Phase', 'Mode2_Modulus', 'Mode2_Phase', 'Mode3_Modulus', 'Mode3_Phase')
        pr_file.analysis_and_plot_2D('SVD_analysis', array_names,
                                     title='SVD Mode1 %.2f%%, Mode2 %.2f%%, Mode3 %.2f%%' % (evalue[0] * 100, evalue[1] * 100, evalue[2] * 100),
                                     filename="Trial%02d_svd" % trial_num, save_image=True,
                                     subplot_config=(3, 2), display_range=display_range)

    pr_file.plot_error_matrix(save_image=True, filename="Error_Trial%d.png" % (trial_num))
    pr_file.plot_2D_intensity(array_group='Selected_average', save_image=True, filename="Selected_intensity_difference_Trial%d.png" % (trial_num))

    # %% save the Information for the Phase retrieval
    ending_time = time.time()
    pr_file.add_para('total_calculation_time', ending_time - starting_time)
    pr_file.save_para_list()
    section = 'General Information'
    para_name_list = [
        'year', 'beamtimeID', 'scan_number', 'p10_newfile', 'data_description', 'omega',
        'delta', 'omegastep', 'detector_distance', 'energy', 'pixelsize', 'intensity_file',
        'mask_file', 'pathsave', 'unit', 'box_size']
    pr_file.save_para_to_infor_file(path_retrieval_infor, section, para_name_list)
    pr_infor.add_para('total_trial_num', section, trial_num)
    pr_infor.infor_writer()

    section = 'Trial %02d' % trial_num
    para_name_list = [
        'pathresult', 'data_shape', 'use_mask', 'start_trial_num', 'nb_run',
        'voxel_size', 'Ortho_voxel_size', 'algorithm', 'precision', 'flip_condition',
        'first_seed_flip', 'total_calculation_time', 'support_type',
        'support_from_trial', 'start_trial_num', 'auto_corr_thrpara',
        'Initial_support_threshold', 'percent_selected',
        'modulus_smooth_width', 'path_import_initial_support', 'Free_LLK',
        'FLLK_percentage', 'FLLK_radius', 'support_update', 'threhold_update_method',
        'support_update_loops', 'support_threshold_min', 'support_threshold_max',
        'support_smooth_width_begin', 'support_smooth_width_end', 'threhold_increase_rate',
        'hybrid_para_begin', 'hybrid_para_end', 'detwin_axis',
        'further_analysis_selected', 'further_analysis_method',
        'phase_unwrap_method', 'error_for_further_analysis_selection']
    pr_file.save_para_to_infor_file(path_retrieval_infor, section, para_name_list)
    return

if __name__ == '__main__':
    phase_retrieval_2D(83)