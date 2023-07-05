# -*- coding: utf-8 -*-
"""
Description
Created on Mon Mar 20 12:17:53 2023

@author: renzhe
"""
import numpy as np
import os
import sys
sys.path.append(r'E:\Work place 3\testprog\pyCXIM_master')
from pyCXIM.Common.Information_file_generator import InformationFileIO
from pyCXIM.phase_retrieval.phase_retrieval_widget import PhaseRetrievalWidget

def plot_phase_retrieval_results():
    # %%Inputs
    pathsave = r'E:\Work place 3\sample\XRD\20220620 Bihan Konstantin\cell2_p01_c1_00009\pynxpre\reciprocal_space_map'
    trial_num = 3
    path_scan_infor = r"E:\Work place 3\sample\XRD\20220620 Bihan Konstantin\cell2_p01_c1_00009\scan_0009_information.txt"
    display_range = [600, 600, 600]

    # %%Load the information file
    print("Loading the information file...")
    pr_file = PhaseRetrievalWidget(pathsave, trial_num, mode='r')
    data_description = pr_file.get_para('data_description')
    print(data_description)

    if data_description == 'reciprocal_space_map':
        para_name_list = [
            'year', 'beamtimeID', 'scan_number', 'p10_newfile',
            'detector_distance', 'energy', 'pixelsize', 'RSM_unit']
    elif data_description == 'stacked_detector_images':
        para_name_list = [
            'year', 'beamtimeID', 'scan_number', 'p10_newfile', 'omega', 'delta',
            'omegastep', 'detector_distance', 'energy', 'pixelsize', 'DC_unit']

    path_retrieval_infor = os.path.join(pathsave, "Phase_retrieval_information.txt")
    pr_infor = InformationFileIO(path_retrieval_infor)
    if not os.path.exists(path_retrieval_infor):
        pr_infor.add_para('total_trial_num', 'General Information', 0)
        if os.path.exists(path_scan_infor):
            scan_infor = InformationFileIO(path_scan_infor)
            pr_infor.copy_para_file(scan_infor, para_name_list, 'General Information')
        else:
            print('Could not find the desired scan parameter file! Generate the file with desired parameters!')
            pr_infor.gen_empty_para_file(para_name_list, 'General Information')
            assert False, 'Please fill the parameter file first, which is stored in the pathsave folder.'
    else:
        pr_infor.infor_reader()
    pr_file.load_para_from_infor_file(path_retrieval_infor, para_name_list)
    zd, yd, xd = pr_file.get_para('data_shape')
    if data_description == 'reciprocal_space_map':
        unit = float(pr_infor.get_para_value('RSM_unit'))
    elif data_description == 'stacked_detector_images':
        unit = float(pr_infor.get_para_value('DC_unit'))

    # %% plot and save the final results
    voxel_size = ((2.0 * np.pi / zd / unit / 10.0), (2.0 * np.pi / yd / unit / 10.0), (2.0 * np.pi / xd / unit / 10.0))
    pr_file.add_para('voxel_size', voxel_size)
    array_names = ('Modulus_sum', 'Phase_sum', 'Support_sum')
    pr_file.plot_3D_result('Average_All', array_names, voxel_size, display_range,
                           'Average results of %d runs' % pr_file.get_para('nb_run'),
                           True, "Trial%d" % (trial_num), save_as_vti=True)
    pr_file.plot_3D_intensity(array_group='Average_All', save_image=True, filename="Intensity_difference_Trial%d.png" % (trial_num))

    # %% select results for SVD analysis or averaging
    voxel_size = ((2.0 * np.pi / zd / unit / 10.0), (2.0 * np.pi / yd / unit / 10.0), (2.0 * np.pi / xd / unit / 10.0))
    array_names = ('Modulus_sum', 'Phase_sum', 'Support_sum')
    pr_file.plot_3D_result('Selected_average', array_names, voxel_size, display_range=display_range, title='Average results of %d runs with minimum error' % pr_file.get_para('further_analysis_selected'), save_image=True, filename="Trial%02d_selected_average" % trial_num, save_as_vti=True)
    if pr_file.get_para('further_analysis_method') == 'SVD':
        evalue = pr_file.get_dataset("SVD_analysis/evalue")
        array_names = ('Mode1_Modulus', 'Mode1_Phase')
        pr_file.plot_3D_result('SVD_analysis', array_names, voxel_size, display_range=display_range, title='SVD Mode1 %.2f%%' % (evalue[0] * 100), save_image=True, filename="Trial%02d_svd_mode1" % trial_num, save_as_vti=True)
        array_names = ('Mode2_Modulus', 'Mode2_Phase')
        pr_file.plot_3D_result('SVD_analysis', array_names, voxel_size, display_range=display_range, title='SVD Mode2 %.2f%%' % (evalue[1] * 100), save_image=True, filename="Trial%02d_svd_mode2" % trial_num, save_as_vti=False)

    pr_file.plot_3D_intensity(array_group='Selected_average', save_image=True, filename="Selected_intensity_difference_Trial%d.png" % (trial_num))
    pr_file.plot_error_matrix(unit, filename="Error_Trial%d.png" % (trial_num))
    # %% Transforming into Orthoganol coordinates
    if data_description == 'stacked_detector_images':
        array_names = ('Modulus_sum', 'Phase_sum', 'Support_sum')
        pr_file.ortho_3D_transform('Average_All', array_names)
        Ortho_unit = pr_file.get_para('Ortho_unit')
        Ortho_voxel_size = (Ortho_unit, Ortho_unit, Ortho_unit)
        pr_file.add_para('Ortho_voxel_size', Ortho_voxel_size)
        array_names = ('Ortho_Modulus_sum', 'Ortho_Phase_sum', 'Ortho_Support_sum')
        pr_file.plot_3D_result('Ortho/Average_All', array_names, Ortho_voxel_size, display_range,
                               save_image=True, filename="Trial%d_orthonormalized" % (trial_num),
                               save_as_vti=True)

        array_names = ('Modulus_sum', 'Phase_sum', 'Support_sum')
        pr_file.ortho_3D_transform('Selected_average', array_names)
        Ortho_unit = pr_file.get_para('Ortho_unit')
        array_names = ('Ortho_Modulus_sum', 'Ortho_Phase_sum', 'Ortho_Support_sum')
        pr_file.plot_3D_result('Ortho', array_names, Ortho_voxel_size, display_range=display_range, title='Average results of %d runs with minimum error' % pr_file.get_para('further_analysis_selected'), save_image=True, filename="Trial%02d_ortho_selected_average" % trial_num, save_as_vti=True)

        if pr_file.get_para('further_analysis_method') == 'SVD':
            evalue = pr_file.get_dataset("SVD_analysis/evalue")
            array_names = ('Mode1_Modulus', 'Mode1_Phase', 'Mode2_Modulus', 'Mode2_Phase', 'Mode3_Modulus', 'Mode3_Phase')
            pr_file.ortho_3D_transform('SVD_analysis', array_names)
            array_names = ('Ortho_Mode1_Modulus', 'Ortho_Mode1_Phase')
            pr_file.plot_3D_result('Ortho/Selected_average', array_names, Ortho_voxel_size, display_range=display_range, title='SVD Ortho Mode1 %.2f%%' % (evalue[0] * 100), save_image=True, filename="Trial%02d_svd_otho_mode1" % trial_num, save_as_vti=True)

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
        'delta', 'omegastep', 'detector_distance', 'energy', 'pixelsize', 'RSM_unit', 'DC_unit', 'intensity_file',
        'mask_file', 'pathsave']
    pr_file.save_para_to_infor_file(path_retrieval_infor, section, para_name_list)
    pr_infor.add_para('total_trial_num', section, total_trial_num)
    pr_infor.infor_writer()

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
        'further_analysis_selected', 'further_analysis_method', 'error_for_further_analysis_selection']
    pr_file.save_para_to_infor_file(path_retrieval_infor, section, para_name_list)
    return


if __name__ == '__main__':
    plot_phase_retrieval_results()
