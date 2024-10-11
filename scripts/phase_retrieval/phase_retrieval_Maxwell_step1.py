# -*- coding: utf-8 -*-
"""
Perform the 3D phase retrieval and save the results.

"""

import time
from pyCXIM.phase_retrieval.phase_retrieval_widget import PhaseRetrievalWidget

def phase_retrieval_main():
    starting_time = time.time()
    # %%Inputs
    pathsave = r'F:\Work place 4\Temp\B12SYNS1P1_00043\pynxpre\reciprocal_space_map'
    intensity_file = 'scan0043.npz'
    mask_file = 'scan0043_mask.npz'
    trial_num = 1
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
    Initial_support_threshold = 0.8
    # If support_type is 'support_selected' or 'modulus_selected'
    percent_selected = 10
    # If support_type is 'modulus_selected'
    modulus_smooth_width = 0.3
    # If support_type is 'import'
    path_import_initial_support = r'E:\Work place 3\sample\XRD\20221103 BFO islands\BFO_LAO_4_7_00087\cutqz\Trial02.npz'

    start_trial_num = 0
    SeedNum = 100
    precision = '32'
    algorithm = '(HIO**40*Sup)**20*DETWIN*(DIF**50)**2*(RAAR**60*ER**10*Sup)**40'
    # algorithm = "DIF**200*(RAAR**50*ER**10)**20"

    # If you want to perform Free loglikelihood calculation, please set Free_LLK to be True
    Free_LLK = False
    FLLK_percentage = 0.01
    FLLK_radius = 3

    # Inputs: parameters for the shrink wrap loop
    # threhold_update_method = 'random'
    threhold_update_method = 'exp_increase'
    # threhold_update_method = 'lin_increase'
    support_para_update_precent = 0.8
    thrpara_min = 0.08
    thrpara_max = 0.11
    support_smooth_width_begin = 3.5
    support_smooth_width_end = 1.0
    hybrid_para_begin = 0.0
    hybrid_para_end = 0.0

    # Input: parameters for the detwin operation
    detwin_axis = 0

    # Input: parameters for flipping the images to remove the trival solutions.
    flip_condition = 'Support'
    # flip_condition ='Phase'
    # flip_condition ='Modulus'
    first_seed_flip = False
    phase_unwrap_method = 6

    # Input: The number of images selected for further analysis like SVD and average
    further_analysis_selected = 10
    error_type_for_selection = 'Fourier space error'

    # Input: Parameters determining the display of the images
    display_image_num = 10

    # %% Load the image data and the mask
    pr_file = PhaseRetrievalWidget(pathsave, trial_num, data_description, mode='w')
    pr_file.load_image_data(intensity_file, mask_file)

    # %% create the initial support for the phase retrieval process
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

    # %% select results for SVD analysis or averaging
    pr_file.further_analysis(further_analysis_selected, error_type=error_type_for_selection)
    ending_time = time.time()
    pr_file.add_para('total_calculation_time', ending_time - starting_time)
    pr_file.save_para_list()
    return


if __name__ == '__main__':
    phase_retrieval_main()
