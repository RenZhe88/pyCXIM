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
path_scan_infor = r"E:\Work place 3\sample\XRD\20221103 BFO islands\BFO_LAO_4_7_00152\scan_0152_information.txt"
SeedNum = 100
# For 2D images the data description can be 'cutqz', 'cutqy', 'cutqx', 'cuty'
data_description = 'cutqz'
pathsave = r'E:\Work place 3\sample\XRD\20221103 BFO islands\BFO_LAO_4_7_00152\cutqz'
intensity_file = "%s.npy" % data_description
mask_file = "%s_mask.npy" % data_description

# algorithm = "(DIF**50)**2*(HIO**50*Sup)**20*(DIF**50)**2*(RAAR**80*ER**10*Sup)**30"
algorithm = "DIF**200*(RAAR**50*ER**10)**20"

# Input: parameters for creating the initial suppport.
# Please chose from 'auto_correlation', 'import', 'average', 'support_selected', or 'modulus_selected'
support_type = 'support_selected'
support_from_trial = 1

# If support_type is 'auto_correlation'
auto_corr_thrpara = 0.004
# If support_type is 'average', 'support_selected', or'modulus_selected'
Initial_support_threshold = 0.6
# If support_type is 'support_selected' or 'modulus_selected'
percent_selected = 10
# If support_type is 'modulus_selected'
modulus_smooth_width = 0.5
# If support_type is 'import'
path_import_initial_support = r'E:\Work place 3\sample\XRD\20221103 BFO islands\BFO_LAO_4_7_00087\cutqz\Trial02.npz'

# Input: starting image inherented from trial
start_trial_num = 1

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

# Input: parameters for flipping the images to remove the trival solutions.
# flip_condition = 'Support'
flip_condition = 'Phase'
# flip_condition ='Modulus'
first_seed_flip = False

# Input: Parameters for further analysis like SVD and average
further_analysis_selected = 10

# Input: Parameters determining the display of the images
display_range = [400, 400]
display_image_num = 10
# %% Load the image data and the mask

# Loading the intensity and the mask defining the dead pixels
print("Loading the information file...")

if data_description in ['cutqx', 'cutqy', 'cutqz']:
    para_name_list = [
        'year', 'beamtimeID', 'scan_number', 'p10_newfile', 'omega', 'delta',
        'omegastep', 'detector_distance', 'energy', 'pixelsize', 'RSM_unit',
        'pynx_box_size']
elif data_description == 'cuty':
    para_name_list = [
        'year', 'beamtimeID', 'scan_number', 'p10_newfile', 'omega', 'delta',
        'omegastep', 'detector_distance', 'energy', 'pixelsize', 'DC_unit',
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
        pr_infor.copy_para_file(scan_infor, para_name_list, 'General Information')
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
if data_description in ['cutqx', 'cutqy', 'cutqz']:
    unit = float(pr_infor.get_para_value('RSM_unit'))
    pynx_box_size = np.array(pr_infor.get_para_value('pynx_box_size'))
elif data_description == 'cuty':
    unit = float(pr_infor.get_para_value('DC_unit'))
    direct_cut_box_size = np.array(pr_infor.get_para_value('direct_cut_box_size'))

# %%Load information file and support
pr_file.create_initial_support(support_type, auto_corr_thrpara, support_from_trial,
                               Initial_support_threshold, percent_selected, modulus_smooth_width,
                               path_import_initial_support)

# %% Start the retrieval process
pr_file.phase_retrieval_main(algorithm, SeedNum, start_trial_num, Free_LLK,
                             FLLK_percentage, FLLK_radius, threhold_update_method,
                             support_para_update_precent, thrpara_min, thrpara_max,
                             support_smooth_width_begin, support_smooth_width_end,
                             flip_condition, first_seed_flip, display_image_num)

# %% plot and save the final results
voxel_size = ((2.0 * np.pi / yd / unit / 10.0), (2.0 * np.pi / xd / unit / 10.0))
pr_file.add_para('voxel_size', voxel_size)
array_names = ('Modulus_sum', 'Phase_sum', 'Support_sum')
pr_file.plot_2D_result('Average_All', array_names, voxel_size, display_range,
                       'Average results of %d runs' % pr_file.get_para('nb_run'),
                       save_image=True, filename="Trial%d" % (trial_num))
pr_file.plot_2D_intensity(save_image=True, filename="Intensity_difference_Trial%d.png" % (trial_num))
pr_file.plot_error_matrix(save_image=True, filename="Error_Trial%d.png" % (trial_num))

# %%Orthonormalization
if data_description == 'cuty':
    array_names = ('Modulus_sum', 'Phase_sum', 'Support_sum')
    pr_file.ortho_2D_transform('Average_All', array_names)
    Ortho_unit = pr_file.get_para('Ortho_unit')
    Ortho_voxel_size = (Ortho_unit, Ortho_unit)
    pr_file.add_para('Ortho_voxel_size', Ortho_voxel_size)
    array_names = ('Ortho_Modulus_sum', 'Ortho_Phase_sum', 'Ortho_Support_sum')
    pr_file.plot_2D_result('Ortho', array_names, Ortho_voxel_size, display_range,
                           pathsave=pathsave, filename="Trial%d_orthonormalized" % (trial_num))

# %% select results for SVD analysis or averaging
pr_file.further_analysis(further_analysis_selected, error_type='Fourier space error')
voxel_size = ((2.0 * np.pi / yd / unit / 10.0), (2.0 * np.pi / xd / unit / 10.0))
array_names = ('select_Modulus_sum', 'select_Phase_sum', 'select_Support_sum')
pr_file.plot_2D_result('Selected_average', array_names, voxel_size, display_range=display_range, title='Average results of %d runs with minimum error' % pr_file.get_para('selected_image_num'), save_image=True, filename="Trial%02d_selected_average" % trial_num)
if pr_file.get_para('further_analysis_method') == 'SVD':
    evalue = pr_file.get_dataset("SVD_analysis/evalue")
    array_names = ('Mode1_Modulus', 'Mode1_Phase', 'Mode2_Modulus', 'Mode2_Phase', 'Mode3_Modulus', 'Mode3_Phase')
    pr_file.plot_2D_result('SVD_analysis', array_names, voxel_size, display_range=display_range, title='SVD Mode1 %.2f%%, Mode2 %.2f%%, Mode3 %.2f%%' % (evalue[0] * 100, evalue[1] * 100, evalue[2] * 100), subplot_config=(3, 2), save_image=True, filename="Trial%02d_svd" % trial_num)

# %%Orthonormalization2
if data_description == 'cuty':
    array_names = ('select_Modulus_sum', 'select_Phase_sum', 'select_Support_sum')
    pr_file.ortho_2D_transform('Selected_average', array_names)
    Ortho_unit = pr_file.get_para('Ortho_unit')
    array_names = ('Ortho_select_Modulus_sum', 'Ortho_select_Phase_sum', 'Ortho_select_Support_sum')
    pr_file.plot_2D_result('Ortho', array_names, Ortho_voxel_size, display_range=display_range, title='Average results of %d runs with minimum error' % pr_file.get_para('selected_image_num'), save_image=True, filename="Trial%02d_ortho_selected_average" % trial_num)
    if pr_file.get_para('further_analysis_method') == 'SVD':
        array_names = ('Mode1_Modulus', 'Mode1_Phase', 'Mode2_Modulus', 'Mode2_Phase', 'Mode3_Modulus', 'Mode3_Phase')
        pr_file.ortho_2D_transform('SVD_analysis', array_names)
        array_names = ('Ortho_Mode1_Modulus', 'Ortho_Mode1_Phase, Ortho_Mode2_Modulus', 'Ortho_Mode2_Phase, Ortho_Mode3_Modulus', 'Ortho_Mode3_Phase')
        pr_file.plot_2D_result('Ortho', array_names, Ortho_voxel_size, display_range=display_range, title='SVD ortho Mode1 %.2f%%, Mode2 %.2f%%, Mode3 %.2f%%' % (evalue[0] * 100, evalue[1] * 100, evalue[2] * 100), subplot_config=(3, 2), save_image=True, filename="Trial%02d_svd_otho_mode1" % trial_num)

# %% save the Information for the Phase retrieval
ending_time = time.time()
pr_file.add_para('total_calculation_time', ending_time - starting_time)
pr_file.save_para_list()
section = 'General Information'
pr_infor.add_para('total_trial_num', section, trial_num)
pr_infor.infor_writer()
para_name_list = [
    'year', 'beamtimeID', 'scan_number', 'p10_newfile', 'data_description', 'omega',
    'delta', 'omegastep', 'detector_distance', 'energy', 'pixelsize', 'intensity_file',
    'mask_file', 'pathsave', 'RSM_unit', 'DC_unit']
pr_file.save_para_to_infor_file(path_retrieval_infor, section, para_name_list)

section = 'Trial %02d' % trial_num
para_name_list = [
    'pathresult', 'data_shape', 'use_mask', 'start_trial_num', 'nb_run',
    'voxel_size', 'Ortho_voxel_size', 'algorithm', 'flip_condition',
    'first_seed_flip', 'total_calculation_time', 'support_type',
    'auto_corr_thrpara', 'support_from_trial', 'start_trial_num',
    'auto_corr_thrpara', 'Initial_support_threshold', 'percent_selected',
    'modulus_smooth_width', 'path_import_initial_support', 'Free_LLK',
    'FLLK_percentage', 'FLLK_radius', 'support_update', 'threhold_update_method',
    'support_update_loops', 'support_threshold_min', 'support_threshold_max',
    'support_smooth_width_begin', 'support_smooth_width_end', 'threhold_increase_rate',
    'further_analysis_selected', 'selected_image_num', 'further_analysis_method']
pr_file.save_para_to_infor_file(path_retrieval_infor, section, para_name_list)
