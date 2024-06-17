# -*- coding: utf-8 -*-
"""
3D Phase retrieval.

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
sys.path.append(r'E:\Work place 3\testprog\pyCXIM_master')
from pyCXIM.Common.Information_file_generator import InformationFileIO
from pyCXIM.phase_retrieval.phase_retrieval_widget import PhaseRetrievalWidget

# %%Input
starting_time = time.time()
pathsave = r'F:\Work place 4\sample\XRD\20240602_BFO_chiral_P10_Desy\Battery_cathode\WTY01\WTY01_00054\pynxpre\reciprocal_space_map'
intensity_file = 'scan0054.npz'
mask_file = 'scan0054_mask.npz'
path_scan_infor = r"F:\Work place 4\sample\XRD\20240602_BFO_chiral_P10_Desy\Battery_cathode\WTY01\WTY01_00054\scan_0054_information.txt"
# data_description = 'reciprocal_space_map_CDI'
data_description = 'reciprocal_space_map_BCDI'
# data_description = 'stacked_detector_images_BCDI'

# Input: parameters for creating the initial suppport.
# Please chose from 'auto_correlation', 'import', 'average', 'support_selected', or 'modulus_selected'
support_type = 'auto_correlation'
support_from_trial = 0

# If support_type is 'auto_correlation'
auto_corr_thrpara = 0.004
# If support_type is 'average', 'support_selected', or'modulus_selected'
Initial_support_threshold = 0.7
# If support_type is 'support_selected' or 'modulus_selected'
percent_selected = 10
# If support_type is 'modulus_selected'
modulus_smooth_width = 0.3
# If support_type is 'import'
path_import_initial_support = r'E:\Work place 3\sample\XRD\20221103 BFO islands\BFO_LAO_4_7_00087\cutqz\Trial02.npz'

# Input: starting image inherented from trial
start_trial_num = 0
SeedNum = 100
algorithm = '(HIO**40*Sup)**10*(RAAR**50*ER**10*Sup)**40'
# algorithm = "DIF**200*DETWIN*(RAAR**50*ER**10)**25"

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
thrpara_max = 0.11
support_smooth_width_begin = 3.5
support_smooth_width_end = 1.0

# Input: parameters for the detwin operation
detwin_axis = 0

# Input: parameters for flipping the images to remove the trival solutions.
flip_condition = 'Support'
# flip_condition = 'Phase'
# flip_condition = 'Modulus'
first_seed_flip = True
phase_unwrap_method = 6

# Input: The number of images selected for further analysis like SVD and average
further_analysis_selected = 10
error_type_for_selection = 'Fourier space error'

# Input: Parameters determining the display of the images
display_range = [300, 300, 300]
display_image_num = 10
# %% Load information file, image data and the mask

# Loading the intensity and the mask defining the dead pixels
print("Loading the information file...")
if data_description == 'reciprocal_space_map_CDI':
    para_name_list = [
        'year', 'beamtimeID', 'scan_number', 'p10_newfile',
        'detector_distance', 'energy', 'pixelsize', 'unit']
elif data_description == 'reciprocal_space_map_BCDI':
    para_name_list = [
        'year', 'beamtimeID', 'scan_number', 'p10_newfile',
        'detector_distance', 'energy', 'pixelsize', 'q_vector', 'unit']
elif data_description == 'stacked_detector_images_BCDI':
    para_name_list = [
        'year', 'beamtimeID', 'scan_number', 'p10_newfile', 'omega', 'delta',
        'omegastep', 'detector_distance', 'energy', 'pixelsize', 'q_vector', 'unit']

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
        if data_description == 'reciprocal_space_map_BCDI':
            pr_infor.copy_para_values(scan_infor, ['RSM_q_center', 'RSM_unit'], 'General Information', ['q_vector', 'unit'])
        elif data_description == 'stacked_detector_images_BCDI':
            pr_infor.copy_para_values(scan_infor, ['direct_cut_q_center', 'DC_unit'], 'General Information', ['q_vector', 'unit'])
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


# %% create the initial support for the phase retrieval process
pr_file.create_initial_support(support_type, auto_corr_thrpara, support_from_trial,
                               Initial_support_threshold, percent_selected, modulus_smooth_width,
                               path_import_initial_support)

# %% Start the retrieval process
pr_file.phase_retrieval_main(algorithm, SeedNum, start_trial_num, Free_LLK,
                             FLLK_percentage, FLLK_radius, threhold_update_method,
                             support_para_update_precent, thrpara_min, thrpara_max,
                             support_smooth_width_begin, support_smooth_width_end,
                             detwin_axis, flip_condition, first_seed_flip,
                             phase_unwrap_method, display_image_num)

# %% Analysis, plot and save the final results
pr_file.plot_3D_intensity(array_group='Average_All', save_image=True, filename="Intensity_difference_Trial%d.png" % (trial_num))
array_names = ('Modulus_sum', 'Phase_sum', 'Support_sum')
pr_file.analysis_and_plot_3D('Average_All', array_names,
                             title='Average results of %d runs' % pr_file.get_para('nb_run'),
                             filename="Trial%d" % (trial_num), save_image=True,
                             save_as_vti=True, display_range=display_range)

# %% select results for SVD analysis or averaging
pr_file.further_analysis(further_analysis_selected, error_type=error_type_for_selection)
pr_file.plot_3D_intensity(array_group='Selected_average', save_image=True, filename="Selected_intensity_difference_Trial%d.png" % (trial_num))
pr_file.plot_error_matrix(filename="Error_Trial%d.png" % (trial_num))

array_names = ('Modulus_sum', 'Phase_sum', 'Support_sum')
pr_file.analysis_and_plot_3D('Selected_average', array_names,
                             title='Average results of %d runs with minimum error' % pr_file.get_para('further_analysis_selected'),
                             filename="Trial%02d_selected_average" % trial_num, save_image=True,
                             save_as_vti=True, display_range=display_range)

if pr_file.get_para('further_analysis_method') == 'SVD':
    evalue = pr_file.get_dataset("SVD_analysis/evalue")
    array_names = ('Mode1_Modulus', 'Mode1_Phase')
    pr_file.analysis_and_plot_3D('SVD_analysis', array_names,
                                 title='SVD Mode1 %.2f%%' % (evalue[0] * 100),
                                 filename="Trial%d_svd_mode1" % (trial_num), save_image=True,
                                 save_as_vti=True, display_range=display_range)

    array_names = ('Mode2_Modulus', 'Mode2_Phase')
    pr_file.analysis_and_plot_3D('SVD_analysis', array_names,
                                 title='SVD Mode2 %.2f%%' % (evalue[1] * 100),
                                 filename="Trial%d_svd_mode2" % (trial_num), save_image=True,
                                 save_as_vti=True, display_range=display_range)

# %% save the Information for the Phase retrieval
ending_time = time.time()
pr_file.add_para('total_calculation_time', ending_time - starting_time)
pr_file.save_para_list()
section = 'General Information'

para_name_list = [
    'year', 'beamtimeID', 'scan_number', 'p10_newfile', 'data_description', 'omega',
    'delta', 'omegastep', 'detector_distance', 'energy', 'pixelsize', 'intensity_file',
    'mask_file', 'pathsave']
pr_file.save_para_to_infor_file(path_retrieval_infor, section, para_name_list)
pr_infor.add_para('total_trial_num', section, trial_num)
pr_infor.infor_writer()

section = 'Trial %02d' % trial_num
para_name_list = [
    'pathresult', 'data_shape', 'use_mask', 'start_trial_num', 'nb_run',
    'voxel_size', 'Ortho_voxel_size', 'algorithm', 'flip_condition',
    'first_seed_flip', 'total_calculation_time', 'support_type',
    'support_from_trial', 'start_trial_num', 'auto_corr_thrpara',
    'Initial_support_threshold', 'percent_selected',
    'modulus_smooth_width', 'path_import_initial_support', 'Free_LLK',
    'FLLK_percentage', 'FLLK_radius', 'support_update', 'threhold_update_method',
    'support_update_loops', 'support_threshold_min', 'support_threshold_max',
    'support_smooth_width_begin', 'support_smooth_width_end', 'threhold_increase_rate',
    'detwin_axis', 'further_analysis_selected', 'further_analysis_method',
    'phase_unwrap_method', 'error_for_further_analysis_selection']
pr_file.save_para_to_infor_file(path_retrieval_infor, section, para_name_list)
