# -*- coding: utf-8 -*-
"""
Description
Created on Mon Mar 20 12:17:53 2023

@author: renzhe
"""
import numpy as np
import os
import sys
sys.path.append(r'F:\Work place 3\testprog\pyCXIM_master')
from pyCXIM.Common.Information_file_generator import InformationFileIO
from pyCXIM.phase_retrieval.phase_retrieval_widget import PhaseRetrievalWidget

def plot_phase_retrieval_results():
    # %%Inputs
    pathsave = r'F:\Work place 4\Temp\B12SYNS1P1_00043\pynxpre\reciprocal_space_map'
    trial_num = 1
    path_scan_infor = r"F:\Work place 4\Temp\B12SYNS1P1_00043\scan_0043_information.txt"
    display_range = [500, 500, 500]

    # %%Load the information file
    print("Loading the information file...")
    pr_file = PhaseRetrievalWidget(pathsave, trial_num, mode='r')
    data_description = pr_file.get_para('data_description')
    print(data_description)

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
        pr_infor.add_para('total_trial_num', 'General Information', 0)
        if os.path.exists(path_scan_infor):
            scan_infor = InformationFileIO(path_scan_infor)
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
    pr_file.load_para_from_infor_file(path_retrieval_infor, para_name_list)

    # %% plot and save the final results
    pr_file.plot_3D_intensity(array_group='Average_All', save_image=True, filename="Intensity_difference_Trial%d.png" % (trial_num))
    array_names = ('Modulus_sum', 'Phase_sum', 'Support_sum')
    pr_file.analysis_and_plot_3D('Average_All', array_names,
                                 title='Average results of %d runs' % pr_file.get_para('nb_run'),
                                 filename="Trial%d" % (trial_num), save_image=True,
                                 save_as_vti=True, display_range=display_range)

    # %% select results for SVD analysis or averaging
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
    pr_file.save_para_list()

    section = 'General Information'
    if os.path.exists(path_retrieval_infor):
        pr_infor.infor_reader()
        if trial_num >= pr_infor.get_para_value('total_trial_num'):
            total_trial_num = trial_num
    else:
        total_trial_num = trial_num
    pr_infor.infor_writer()

    para_name_list = [
        'year', 'beamtimeID', 'scan_number', 'p10_newfile', 'data_description', 'omega',
        'delta', 'omegastep', 'detector_distance', 'energy', 'pixelsize', 'unit', 'q_vector', 'intensity_file',
        'mask_file', 'pathsave']
    pr_file.save_para_to_infor_file(path_retrieval_infor, section, para_name_list)
    pr_infor.add_para('total_trial_num', section, total_trial_num)
    pr_infor.infor_writer()

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
    plot_phase_retrieval_results()
