# -*- coding: utf-8 -*-

"""
Description.

Created on Thu Mar 30 11:35:00 2023

@author: renzhe
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import h5py
from ..Common.Information_file_generator import Information_file_io
try:
    from .phase_retrieval_GPU import Phase_Retrieval_funs as pr
    from .phase_retrieval_GPU import Free_LLK_FFTmask
except ModuleNotFoundError:
    from .phase_retrieval_numpy import Phase_Retrieval_funs as pr
    from .phase_retrieval_numpy import Free_LLK_FFTmask
from . import phase_retrieval_post_processing as pp
from scipy.ndimage import gaussian_filter
from scipy.linalg import svd


class phase_retrieval_widget():
    """
    The integrated functions for the phase retrieval.

    These functions shows the phase retrieval process based on the phase retrieval functions.
    The code generates, reads and modifies an h5 file defined by the pathsave and the trial num.
    The plotting functions are also based on the pathes within the h5 file.

    Parameters
    ----------
    pathsave : str
        The path to save the results files and information file.
    trial_num : int
        The number of the trial.
    data_description : str
        The description of the data.
        Data description could be cutqx, cutqy, cutqz, cuty, reciprocal_space_map, stacked_detector_images.
    mode : str, optional
        Reading mode as defined by other python codes, which can be 'w', 'w+', 'a', 'r', 'r+'. The default is 'a'.

    Returns
    -------
    None.

    """

    def __init__(self, pathsave, trial_num, data_description=None, mode='a'):
        # Initiate the phase retrieval process by creating the data file.
        assert os.path.exists(pathsave), 'The folder to save the results file must exist!'

        self.pathsave = pathsave
        self.trial_num = trial_num
        self.pathsaveimg = os.path.join(pathsave, "Trial%02d.h5" % trial_num)
        self.para_dict = {}
        imgfile = h5py.File(self.pathsaveimg, mode)
        for para_name in imgfile.attrs.keys():
            self.para_dict[para_name] = imgfile.attrs[para_name]
        imgfile.close()
        self.para_dict['pathsave'] = self.pathsave
        self.para_dict['trial_num'] = self.trial_num
        self.para_dict['pathresult'] = self.pathsaveimg
        if data_description is not None:
            assert (data_description in ['cutqx', 'cutqy', 'cutqz', 'cuty', 'reciprocal_space_map', 'stacked_detector_images']), 'Data description must be cutqx, cutqy, cutqz, cuty, reciprocal_space_map, stacked_detector_images.'
            self.para_dict['data_description'] = data_description
        return

    def load_image_data(self, intensity_file, mask_file=''):
        """
        Load the diffraction patterns and the mask to start the phase retrieval process.

        The diffraction patterns and the mask should be in the npy format or npz format with data under the keywords name 'data'.

        Parameters
        ----------
        intensity_file : str
            The filename of diffraction intensity.
        mask_file : str, optional
            The filename of 3D bad pixel mask. The default is ''.

        Returns
        -------
        None.

        """
        imgfile = h5py.File(self.pathsaveimg, "r+")
        # Loading the intensity and the mask defining the dead pixels
        pathread = os.path.join(self.pathsave, intensity_file)
        pathmask = os.path.join(self.pathsave, mask_file)
        print('Loading the measured intensity')
        if intensity_file[-4:] == '.npz':
            image = np.load(pathread)['data']
        elif intensity_file[-4:] == '.npy':
            image = np.load(pathread)

        if os.path.exists(pathmask):
            print('Loading the mask')
            if mask_file[-4:] == '.npz':
                MaskFFT = np.load(pathmask)['data']
            elif mask_file[-4:] == '.npy':
                MaskFFT = np.load(pathmask)
            self.para_dict['use_mask'] = True
        else:
            print('No mask will be used')
            MaskFFT = np.zeros_like(image, dtype=float)
            self.para_dict['use_mask'] = False

        self.para_dict['intensity_file'] = intensity_file
        self.para_dict['mask_file'] = mask_file
        self.para_dict['data_shape'] = list(image.shape)

        # Determining how the data should be compressed
        data_shape = self.para_dict['data_shape']
        if len(data_shape) == 3:
            chunks_size = (1, data_shape[1], data_shape[2])
        elif len(data_shape) == 2:
            chunks_size = (data_shape[0], data_shape[1])

        imgfile.create_dataset("Input/intensity", data=image, dtype='f', chunks=chunks_size, compression="gzip")
        imgfile.create_dataset("Input/mask", data=MaskFFT, dtype='f', chunks=chunks_size, compression="gzip")
        imgfile.close()
        return

    def create_initial_support(self, support_type, auto_corr_thrpara=0.004, support_from_trial=0, Initial_support_threshold=0.4, percent_selected=10, modulus_smooth_width=0.3, path_import_initial_support=''):
        """
        Create the initial support for the phase retrieval processes.

        Opitions for the creation method are 'auto_correlation', 'average', 'support_selected', 'modulus_selected', 'import'.
        'auto_correlation': Creat the support according to the autocorrelation function calculated from the diffraction pattern. Usually used if the support is unknown and generate the starting support for the later Shrink wrap processes.
        'average': Create the support based on the average support of the previous trials.
        'support_selected': Creat the support based on the average support of solutions with minimum error in the previous trial.
        'modulus_selected': Creat the support based on the average modulus of solutions with minimum error in the previous trial.
        'import': Import the support from another file. The support should be stored in the npz format with keyword name 'data'.

        Parameters
        ----------
        support_type : str
            Defines how the initial support should be calculated, opitions are 'auto_correlation', 'average', 'support_selected', 'modulus_selected', 'import'.
        auto_corr_thrpara : float, optional
            The threshold value for the autocorrelation function. The default is 0.004.
        support_from_trial : int, optional
            The number of the previous trials, from which the initial support should be calculated. Needed for 'average', 'support_selected', 'modulus_selected' methods. The default is 0.
        Initial_support_threshold : float, optional
            The threshold value for the average support. Needed for 'average', 'support_selected', 'modulus_selected' methods. The default is 0.4.
        percent_selected : float, optional
            The percentage of the previous trial solution to be used for the support calculation. Needed for 'support_selected', 'modulus_selected' methods. The default is 10.
        modulus_smooth_width : float, optional
            The Standard deviation for Gaussian kernel, which is used to blur the average modulus. The default is 0.3.
        path_import_initial_support : str, optional
            The path for importing the support file, the support file should bin the npz format with keyword name data. The default is ''.

        Returns
        -------
        None.

        """
        # Load information file and support
        imgfile = h5py.File(self.pathsaveimg, "r+")

        if support_type in ['average', 'support_selected', 'modulus_selected']:
            path_previous_result = os.path.join(self.pathsave, 'Trial%02d.h5' % support_from_trial)
            assert os.path.exists(path_previous_result), 'The previous results file for the support calculation does not exist! Please check the parameters again!'
        elif support_type == 'auto_correlation':
            assert ("Input/intensity" in imgfile), 'Please import the diffraction pattern first, so that the auto correlation funciton can be calculated!'
        elif support_type == 'import':
            assert os.path.exits(path_import_initial_support), 'The support file to be imported does not exist! Please check the path again!'
            assert path_import_initial_support[-4:] == '.npz', 'The support file has to be in the npz format, with data under the keyword name data.'

        self.para_dict['support_type'] = support_type
        data_shape = self.para_dict['data_shape']

        # Calculating the starting support
        if support_type == 'auto_correlation':
            print('Initial support calculated from the autocorrelation function.')
            image = np.array(imgfile["Input/intensity"], dtype=float)
            support = np.zeros_like(image, dtype=float)
            Startautocorrelation = np.abs(np.fft.fftshift(np.fft.fftn(np.fft.fftshift(np.sqrt(image)))))
            threshold = auto_corr_thrpara * (np.amax(Startautocorrelation) - np.amin(Startautocorrelation)) + np.amin(Startautocorrelation)
            support[Startautocorrelation >= threshold] = 1.0
            self.para_dict['auto_corr_thrpara'] = auto_corr_thrpara
        elif support_type == 'average':
            print('Initial support imported from Trial%02d.' % support_from_trial)
            previous_result_file = h5py.File(path_previous_result, 'r')
            support_sum = np.array(previous_result_file['Average_All/Support_sum'], dtype=float)
            support = np.zeros(data_shape)
            support[support_sum >= Initial_support_threshold] = 1.0
            previous_result_file.close()
            self.para_dict['support_from_trial'] = support_from_trial
            self.para_dict['Initial_support_threshold'] = Initial_support_threshold
        elif support_type == 'support_selected':
            print('Initial support calculated from Trial%02d' % support_from_trial)
            previous_result_file = h5py.File(path_previous_result, 'r')
            support_sum = np.zeros(data_shape)
            err_ar = np.array(previous_result_file['Error/error'])
            previous_SeedNum = int(previous_result_file.attrs['nb_run'])
            selected_img_num = int(percent_selected / 100.0 * previous_SeedNum)
            print('%d images selected to calcultate the support' % selected_img_num)
            Previous_Seed_selected = np.argsort(err_ar[:, 2])[:selected_img_num]
            for Previous_Seed in Previous_Seed_selected:
                support_sum += np.array(previous_result_file['Solutions/Seed%03d/support' % Previous_Seed], dtype=float)
            support_sum = support_sum / selected_img_num
            support = np.zeros_like(support_sum)
            support[support_sum >= Initial_support_threshold] = 1.0
            previous_result_file.close()
            self.para_dict['support_from_trial'] = support_from_trial
            self.para_dict['Initial_support_threshold'] = Initial_support_threshold
            self.para_dict['support_select_percent'] = percent_selected
        elif support_type == 'modulus_selected':
            print('Initial support calculated from Trial%02d' % support_from_trial)
            previous_result_file = h5py.File(path_previous_result, 'r')
            modulus_sum = np.zeros(data_shape)
            err_ar = np.array(previous_result_file['Error/error'], dtype=float)
            previous_SeedNum = previous_result_file.attrs['nb_run']
            selected_img_num = int(percent_selected / 100.0 * previous_SeedNum)
            print('%d images selected to calcultate the support' % selected_img_num)
            Previous_Seed_selected = np.argsort(err_ar[:, 2])[:selected_img_num]
            for Previous_Seed in Previous_Seed_selected:
                modulus_sum += np.abs(np.array(previous_result_file['Solutions/Seed%03d/image' % Previous_Seed]))
            Bluredimg = gaussian_filter(modulus_sum, sigma=modulus_smooth_width)
            threshold = Initial_support_threshold * (np.amax(Bluredimg) - np.amin(Bluredimg)) + np.amin(Bluredimg)
            support = np.zeros(data_shape)
            support[Bluredimg >= threshold] = 1.0
            previous_result_file.close()
            self.para_dict['support_from_trial'] = support_from_trial
            self.para_dict['modulus_select_percent'] = percent_selected
            self.para_dict['Initial_support_threshold'] = Initial_support_threshold
            self.para_dict['modulus_smooth_width'] = modulus_smooth_width
        elif support_type == 'import':
            support = np.load(path_import_initial_support)['data']
            support[support > Initial_support_threshold] = 1
            support[support <= Initial_support_threshold] = 0
            self.para_dict['Initial_support_threshold'] = Initial_support_threshold
            self.para_dict['path_import_initial_support'] = path_import_initial_support

        # Determining how the data should be compressed
        if len(data_shape) == 3:
            chunks_size = (1, data_shape[1], data_shape[2])
            plt_result_3D_simple((support,), ('Initial support',))
        elif len(data_shape) == 2:
            chunks_size = (data_shape[0], data_shape[1])
            plt_result_2D_simple((support,), ('Initial support',))

        imgfile.create_dataset("Initial_support/support", data=support, dtype='f', chunks=chunks_size, compression="gzip")
        imgfile.close()
        return

    def phase_retrieval_main(self, algorithm, SeedNum, start_trial_num=0, Free_LLK=False, FLLK_percentage=1, FLLK_radius=3, threhold_update_method='exp_increase', support_para_update_precent=0.8, thrpara_min=0.1, thrpara_max=0.12, support_smooth_width_begin=3.5, support_smooth_width_end=1.0, flip_condition='Support', first_seed_flip=False, display_image_num=5):
        """
        Perform the phase retrieval with the aimed algorithm.

        Parameters
        ----------
        algorithm : str
            The algorithms defining the phase retrieval process.
        SeedNum : int
            The total number of runs to be performed with different initial starting point.
        start_trial_num : int, optional
            Start the phase retrieval based on the solutions generated from the previous trials. To start with the random guesses, use 0. The default is 0.
        Free_LLK : bool, optional
            If true, the free log likelihood mask will be generated and used during the phase retrieval process. The default is False.
        FLLK_percentage : float, optional
            The percentage of pixels to be masked for the Free Log likelihood calculation. The default is 1.
        FLLK_radius : int, optional
            The radius of masked pixel clusters to be used in the free log likelihood mask. The default is 3.
        threhold_update_method : str, optional
            Deteriming how the threshold parameter should be updated during the shrinkwrap process.
            'exp_increase': The threshold parameter will exponentially increase from the thrpara_min value to thrpara_max value.
            'lin_increase': The threshold parameter will linearly increase from the thrpara_min value to thrpara_max value.
            'random': The threshold parameter will be randomly adjusted between the thrpara_min value to thrpara_max value.
            The default is 'exp_increase'.
        support_para_update_precent : float, optional
            Determines the number of support updates to be performed before the support para reaches and stays at the aimed value. The default is 0.8.
        thrpara_min : float, optional
            The minimun threshold value to be used in the shrink wrap process. The default is 0.1.
        thrpara_max : float, optional
            The maximum threshold value to be used in the shrink wrap process. The default is 0.12.
        support_smooth_width_begin : float, optional
            The standard deviation of Gaussian kernal used at the begining of the shrink wrap process. The default is 3.5.
        support_smooth_width_end : TYPE, optional
            The standard deviation of Gaussian kernal to be reached at the end of the shrink wrap process. The default is 1.0.
        flip_condition : str, optional
            Determins which image is used to judge whether an image should be flip after the phase retrieval process.
            Opitions are 'Support', 'Modulus', 'Phase'. The default is 'Support'.
        first_seed_flip : bool, optional
            If true, the first image generated will be flipped, which will also flip all the solutions afterwards. The default is False.
        display_image_num : int, optional
            The number of runs to be displayed during the run. The default is 5.

        Returns
        -------
        None.
        """
        # Start the retrieval process
        imgfile = h5py.File(self.pathsaveimg, "r+")
        Solution_group = imgfile.create_group('Solutions')

        assert ("Input/intensity" in imgfile), 'Please import the diffraction pattern first!'
        assert ("Initial_support/support" in imgfile), 'Please generated the initial suppport for the phase retrieval process first!'

        image = np.array(imgfile["Input/intensity"], dtype=float)
        MaskFFT = np.array(imgfile["Input/mask"], dtype=float)
        support = np.array(imgfile["Initial_support/support"], dtype=float)

        if start_trial_num != 0:
            path_start_trial = os.path.join(self.pathsave, 'Trial%02d.h5' % start_trial_num)
            if os.path.exists(path_start_trial):
                previous_result_file = h5py.File(path_start_trial, 'r')
                if SeedNum > previous_result_file.attrs['nb_run']:
                    print('The number of starting point is larger than the previous one!')
                    SeedNum = previous_result_file.attrs['nb_run']
                    print('Use %d Seed instead!' % SeedNum)
            else:
                print('The file for the starting trial does not exist! Start the phase retrieval process with random guesses!')
                start_trial_num = 0

        # Determining how the data should be compressed
        data_shape = self.para_dict['data_shape']
        if len(data_shape) == 3:
            chunks_size = (1, data_shape[1], data_shape[2])
        elif len(data_shape) == 2:
            chunks_size = (data_shape[0], data_shape[1])

        if Free_LLK:
            LLKmask, MaskFFT = Free_LLK_FFTmask(MaskFFT, percentage=FLLK_percentage, r=FLLK_radius)
            self.para_dict['Free_LLK'] = True
            self.para_dict['FLLK_percentage'] = FLLK_percentage
            self.para_dict['FLLK_radius'] = FLLK_radius
            imgfile.create_dataset("Free_LLK_mask/LLKmask", data=LLKmask, dtype='f', chunks=chunks_size, compression="gzip")
            imgfile.create_dataset("Free_LLK_mask/MaskFFT", data=MaskFFT, dtype='f', chunks=chunks_size, compression="gzip")
        else:
            LLKmask = None
            self.para_dict['Free_LLK'] = False

        Modulus_sum = np.zeros_like(image)
        Phase_sum = np.zeros_like(image)
        Img_sum = np.zeros_like(image, dtype=complex)
        Support_sum = np.zeros_like(image)
        intensity_sum = np.zeros_like(image)
        err_ar = np.zeros((SeedNum, 6))

        # Making folders to store the images
        for Seed in range(SeedNum):
            if start_trial_num != 0:
                starting_img = np.array(previous_result_file['Solutions/Seed%03d/image' % Seed], dtype=complex)
                PR_seed = pr(image, Seed, starting_img=starting_img, support=support, MaskFFT=MaskFFT)
            else:
                PR_seed = pr(image, Seed, support=support, MaskFFT=MaskFFT)

            algor_extend = PR_seed.Algorithm_expander(algorithm)

            if int(algor_extend.count(('Sup', 0))) > 0:
                Gaussiandelta = support_smooth_width_begin
                thrpara = thrpara_min
                support_update_num = np.around(algor_extend.count(('Sup', 0)) * support_para_update_precent)
                support_decay_rate = np.power(support_smooth_width_end / support_smooth_width_begin, 1.0 / support_update_num)
                if threhold_update_method == 'exp_increase':
                    thr_increase_rate = np.power(thrpara_max / thrpara_min, 1.0 / support_update_num)
                elif threhold_update_method == 'lin_increase':
                    thr_increase_rate = (thrpara_max - thrpara_min) / support_update_num

            for method, loopnum in algor_extend:
                if method == 'ER':
                    PR_seed.ER(loopnum)
                elif method == 'HIO':
                    PR_seed.HIO(loopnum)
                elif method == 'NHIO':
                    PR_seed.NHIO(loopnum)
                elif method == 'RAAR':
                    PR_seed.RAAR(loopnum)
                elif method == 'DIF':
                    PR_seed.DIFFERNCE_MAP(loopnum)
                elif method == 'Sup':
                    PR_seed.Sup(Gaussiandelta, thrpara)
                    if Gaussiandelta > support_smooth_width_end:
                        Gaussiandelta = Gaussiandelta * support_decay_rate
                        if threhold_update_method == 'random':
                            thrpara = thrpara_min + np.random.rand() * (thrpara_max - thrpara_min)
                        elif threhold_update_method == 'exp_increase':
                            thrpara = thrpara * thr_increase_rate
                        elif threhold_update_method == 'lin_increase':
                            thrpara = thrpara + thr_increase_rate
                elif method == 'End':
                    PR_seed.End()
                    Support_final = PR_seed.get_support()
                    Modulus_final = PR_seed.get_img_Modulus()
                    Phase_final = pp.phase_corrector(PR_seed.get_img_Phase(), PR_seed.get_support())
                    intensity_sum = intensity_sum + PR_seed.get_intensity()

                    # removing the symmetrical cases by fliping the images
                    if flip_condition == 'Modulus':
                        flip_con = (np.sum(Modulus_sum * Modulus_final) < np.sum(Modulus_sum * np.flip(Modulus_final)))
                    elif flip_condition == 'Support':
                        flip_con = (np.sum(Support_sum * Support_final) < np.sum(Support_sum * np.flip(Support_final)))
                    elif flip_condition == 'Phase':
                        flip_con = (np.sum(Phase_sum * Phase_final) < np.sum(Phase_sum * -1.0 * np.flip(Phase_final)))

                    if Seed == 0 and first_seed_flip:
                        flip_con = True

                    if flip_con:
                        PR_seed.flip_img()
                        Support_final = PR_seed.get_support()
                        Modulus_final = PR_seed.get_img_Modulus()
                        Phase_final = pp.phase_corrector(PR_seed.get_img_Phase(), PR_seed.get_support())
                        err_ar[Seed, :] = np.array([Seed, np.sum(PR_seed.get_support()), PR_seed.get_Fourier_space_error(), PR_seed.get_Poisson_Likelihood(), PR_seed.get_Free_LogLikelihood(LLKmask), 1])
                        sys.stdout.write(', flip:1')
                    else:
                        err_ar[Seed, :] = np.array([Seed, np.sum(PR_seed.get_support()), PR_seed.get_Fourier_space_error(), PR_seed.get_Poisson_Likelihood(), PR_seed.get_Free_LogLikelihood(LLKmask), 0])
            sys.stdout.write('\n')
            Seed_group = Solution_group.create_group("Seed%03d" % Seed)
            Seed_group.create_dataset("image", data=PR_seed.get_img(), dtype='complex128', chunks=chunks_size, compression="gzip")
            Seed_group.create_dataset("support", data=PR_seed.get_support(), dtype='f', chunks=chunks_size, compression="gzip")

            Support_sum = Support_sum + Support_final
            Modulus_sum = Modulus_sum + Modulus_final
            Phase_sum = Phase_sum + Phase_final
            Img_sum = Img_sum + np.multiply(Modulus_final, np.exp(1j * Phase_final))

            if Seed < display_image_num:
                if len(data_shape) == 3:
                    plt_result_3D_simple((Support_final, Modulus_final, Phase_final), ('Support', 'Modulus', 'Phase'))
                elif len(data_shape) == 2:
                    plt_result_2D_simple((Support_final, Modulus_final, Phase_final), ('Support', 'Modulus', 'Phase'))

        if start_trial_num != 0:
            previous_result_file.close()

        Modulus_sum = Modulus_sum / SeedNum
        Phase_sum = Phase_sum / SeedNum
        Img_sum = Img_sum / SeedNum
        intensity_sum = intensity_sum / SeedNum
        Support_sum = Support_sum / SeedNum

        PRTF = pp.cal_PRTF(image, Img_sum, MaskFFT)
        self.para_dict['nb_run'] = SeedNum
        self.para_dict['algorithm'] = algorithm
        self.para_dict['start_trial_num'] = start_trial_num

        if int(algor_extend.count(('Sup', 0))) != 0:
            self.para_dict['support_update'] = True
            self.para_dict['support_smooth_width_begin'] = support_smooth_width_begin
            self.para_dict['support_smooth_width_end'] = support_smooth_width_end
            self.para_dict['support_threshold_min'] = thrpara_min
            self.para_dict['support_threshold_max'] = thrpara_max
            self.para_dict['support_update_loops'] = support_update_num
            self.para_dict['support_decay_rate'] = support_decay_rate
            self.para_dict['threhold_update_method'] = threhold_update_method
            if 'increase' in threhold_update_method:
                self.para_dict['threhold_increase_rate'] = thr_increase_rate
        else:
            self.para_dict['support_update'] = False

        self.para_dict['flip_condition'] = flip_condition
        self.para_dict['first_seed_flip'] = first_seed_flip

        imgfile.create_dataset("Error/error", data=err_ar, dtype='float')
        imgfile['Error'].attrs['column_names'] = ['Seed', 'support_size', 'Fourier space error', 'Poisson logLikelihood', 'Free logLikelihood', 'Flip']
        imgfile.create_dataset("Average_All/Support_sum", data=Support_sum, dtype='float', chunks=chunks_size, compression="gzip")
        imgfile.create_dataset("Average_All/Modulus_sum", data=Modulus_sum, dtype='float', chunks=chunks_size, compression="gzip")
        imgfile.create_dataset("Average_All/Img_sum", data=Img_sum, dtype='complex128', chunks=chunks_size, compression="gzip")
        imgfile.create_dataset("Average_All/Phase_sum", data=Phase_sum, dtype='float', chunks=chunks_size, compression="gzip")
        imgfile.create_dataset("Average_All/intensity_sum", data=intensity_sum, dtype='float', chunks=chunks_size, compression="gzip")
        imgfile.create_dataset("Average_All/phase_retrieval_transfer_function", data=PRTF, dtype='float')
        imgfile.close()
        return

    def further_analysis(self, further_analysis_selected, error_type='Fourier space error'):
        """
        Select the results with minimun error to perform further analysises.

        Two analysis methods posssible:
        If the support is not updated during the phase retrieval process, then SVD analysis is performed.
        If the support is updatedd during the phase retrieval process, then the selected solutions are averaged.

        Parameters
        ----------
        further_analysis_selected : float
            The percentage of the images to be used in the further analysis.
        error_type : str, optional
            The type of error to be used for the selection of images.
            Three options are 'Fourier space error', 'Poisson logLikelihood', 'Free logLikelihood'.
            The default is 'Fourier space error'.

        Returns
        -------
        None.

        """
        imgfile = h5py.File(self.pathsaveimg, "r+")
        SeedNum = self.para_dict['nb_run']
        selected_image_num = int(further_analysis_selected / 100.0 * SeedNum)

        # import the error matrix for the image selection.
        err_ar = np.array(imgfile["Error/error"], dtype=float)
        if error_type == 'Fourier space error':
            err_ar = err_ar[:, 2]
        elif error_type == 'Poisson logLikelihood':
            err_ar = err_ar[:, 3]
        elif error_type == 'Free logLikelihood':
            err_ar = err_ar[:, 4]
        Seed_selected = np.argsort(err_ar)[:int(selected_image_num)]

        # Determining how the data should be compressed.
        data_shape = self.para_dict['data_shape']
        if len(data_shape) == 3:
            chunks_size = (1, data_shape[1], data_shape[2])
        elif len(data_shape) == 2:
            chunks_size = (data_shape[0], data_shape[1])

        if (not self.para_dict['support_update']) and selected_image_num >= 3:
            further_analysis_method = 'SVD'
            support = np.array(imgfile["Initial_support/support"], dtype=float)
            if self.para_dict['first_seed_flip']:
                support = np.flip(support)

            print("%d images selected for SVD analysis" % selected_image_num)
            result_matrix = np.zeros((int(np.sum(support)), selected_image_num), dtype=complex)
            Mode1 = np.zeros(data_shape, dtype=complex)
            Mode2 = np.zeros(data_shape, dtype=complex)
            Mode3 = np.zeros(data_shape, dtype=complex)
            Avr_Modulus = np.zeros(data_shape, dtype=float)
            Avr_Phase = np.zeros(data_shape, dtype=float)

            for i, Seed in enumerate(Seed_selected):
                Img = np.array(imgfile['Solutions/Seed%03d/image' % Seed], dtype=complex)
                Modulus = np.abs(Img)
                Phase = pp.phase_corrector(np.angle(Img), support, 0)
                result_matrix[:, i] = np.multiply(Modulus, np.exp(1j * Phase))[support == 1]
                Avr_Modulus = Avr_Modulus + Modulus
                Avr_Phase = Avr_Phase + Phase

            try:
                import torch
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                result_matrix = torch.from_numpy(result_matrix).to(device)

                u_vector, evalue, v_vector = torch.linalg.svd(result_matrix, full_matrices=False)
                u_vector = u_vector.cpu().numpy()
                evalue = evalue.cpu().numpy()
            except ImportError:
                u_vector, evalue, v_vector = svd(result_matrix, full_matrices=False)

            Mode1[support == 1] = u_vector[:, 0]
            Mode2[support == 1] = u_vector[:, 1]
            Mode3[support == 1] = u_vector[:, 2]
            Mode1_Modulus = np.abs(Mode1)
            Mode1_Phase = pp.phase_corrector(np.angle(Mode1), support, 0)
            Mode2_Modulus = np.abs(Mode2)
            Mode2_Phase = pp.phase_corrector(np.angle(Mode2), support, 0)
            Mode3_Modulus = np.abs(Mode3)
            Mode3_Phase = pp.phase_corrector(np.angle(Mode3), support, 0)
            evalue = np.square(evalue)
            evalue = evalue / np.sum(evalue)
            Avr_Modulus = Avr_Modulus / selected_image_num
            Avr_Phase = Avr_Phase / selected_image_num

            self.para_dict['further_analysis_selected'] = further_analysis_selected
            self.para_dict['selected_image_num'] = selected_image_num
            self.para_dict['further_analysis_method'] = further_analysis_method
            imgfile.create_dataset("Selected_average/select_Modulus_sum", data=Avr_Modulus, dtype='float', chunks=chunks_size, compression="gzip")
            imgfile.create_dataset("Selected_average/select_Phase_sum", data=Avr_Phase, dtype='float', chunks=chunks_size, compression="gzip")
            imgfile.create_dataset("Selected_average/select_Support_sum", data=support, dtype='float', chunks=chunks_size, compression="gzip")
            imgfile.create_dataset("SVD_analysis/Mode1_Modulus", data=Mode1_Modulus, dtype='float', chunks=chunks_size, compression="gzip")
            imgfile.create_dataset("SVD_analysis/Mode1_Phase", data=Mode1_Phase, dtype='float', chunks=chunks_size, compression="gzip")
            imgfile.create_dataset("SVD_analysis/Mode2_Modulus", data=Mode2_Modulus, dtype='float', chunks=chunks_size, compression="gzip")
            imgfile.create_dataset("SVD_analysis/Mode2_Phase", data=Mode2_Phase, dtype='float', chunks=chunks_size, compression="gzip")
            imgfile.create_dataset("SVD_analysis/Mode3_Modulus", data=Mode3_Modulus, dtype='float', chunks=chunks_size, compression="gzip")
            imgfile.create_dataset("SVD_analysis/Mode3_Phase", data=Mode3_Phase, dtype='float', chunks=chunks_size, compression="gzip")
            imgfile.create_dataset("SVD_analysis/evalue", data=evalue, dtype='float', compression="gzip")
        else:
            further_analysis_method = 'Average'
            Avr_Modulus = np.zeros(data_shape, dtype=float)
            Avr_Phase = np.zeros(data_shape, dtype=float)
            Avr_support = np.zeros(data_shape, dtype=float)

            for Seed in Seed_selected:
                Img = np.array(imgfile['Solutions/Seed%03d/image' % Seed], dtype=complex)
                support = np.array(imgfile['Solutions/Seed%03d/support' % Seed], dtype=float)
                Modulus = np.abs(Img)
                Phase = pp.phase_corrector(np.angle(Img), support, 0)
                Avr_Modulus = Avr_Modulus + Modulus
                Avr_Phase = Avr_Phase + Phase
                Avr_support = Avr_support + support

            Avr_Modulus = Avr_Modulus / selected_image_num
            Avr_Phase = Avr_Phase / selected_image_num
            Avr_support = Avr_support / selected_image_num

            self.para_dict['further_analysis_selected'] = further_analysis_selected
            self.para_dict['selected_image_num'] = selected_image_num
            self.para_dict['further_analysis_method'] = further_analysis_method
            imgfile.create_dataset("Selected_average/select_Modulus_sum", data=Avr_Modulus, dtype='float', chunks=chunks_size, compression="gzip")
            imgfile.create_dataset("Selected_average/select_Phase_sum", data=Avr_Phase, dtype='float', chunks=chunks_size, compression="gzip")
            imgfile.create_dataset("Selected_average/select_Support_sum", data=Avr_support, dtype='float', chunks=chunks_size, compression="gzip")
        imgfile.close()
        return

    def ortho_2D_transform(self, input_group, input_names):
        """
        Orthonormalize the phase retrieval results.

        Parameters
        ----------
        input_group : str
            The group name of the input array in the result h5 file.
        input_names : list
            The array names in the group.

        Returns
        -------
        None.

        """
        assert ('omega' in self.para_dict.keys()), 'Please import the motor values of omega, delta, omega_step, distance, pixelsize and energy in the parameter list first!'
        imgfile = h5py.File(self.pathsaveimg, "r+")
        omega = self.para_dict['omega']
        omegastep = self.para_dict['omegastep']
        delta = self.para_dict['delta']
        distance = self.para_dict['distance']
        pixelsize = self.para_dict['pixelsize']
        energy = self.para_dict['energy']
        direct_cut_box_size = self.get_para_value('direct_cut_box_size')
        yd = direct_cut_box_size[1]
        for input_name in input_names:
            input_array = imgfile['%s/%s' % (input_group, input_name)]
            Ortho_result, Ortho_unit = pp.Orth2D(input_array, yd, omega, omegastep, delta, distance, pixelsize, energy)

            ny, nx = Ortho_result.shape
            imgfile.create_dataset("Ortho/Ortho_%s" % (input_name), data=Ortho_result, dtype='f', chunks=(ny, nx), compression="gzip")
        self.para_dict['Ortho_unit'] = Ortho_unit
        imgfile.close()
        return

    def ortho_3D_transform(self, input_group, input_names):
        """
        Orthonormalize the phase retrieval results.

        Parameters
        ----------
        input_group : str
            The group name of the input array in the result h5 file.
        input_names : list
            The array names in the group.

        Returns
        -------
        None.

        """
        assert ('omega' in self.para_list), 'Please import the motor values of omega, delta, omega_step, distance, pixelsize and energy in the parameter list first!'
        imgfile = h5py.File(self.pathsaveimg, "r+")
        omega = self.para_list['omega']
        omegastep = self.para_list['omegastep']
        delta = self.para_list['delta']
        distance = self.para_list['distance']
        pixelsize = self.para_list['pixelsize']
        energy = self.para_list['energy']
        for input_name in input_names:
            input_array = imgfile['%s/%s' % (input_group, input_name)]
            Ortho_result, Ortho_unit = pp.Orth3D(input_array, omega, omegastep, delta, distance, pixelsize, energy)

            nz, ny, nx = Ortho_result.shape
            imgfile.create_dataset("Ortho/Ortho_%s" % (input_name), data=Ortho_result, dtype='f', chunks=(1, ny, nx), compression="gzip")
        self.para_dict['Ortho_unit'] = Ortho_unit
        imgfile.close()
        return

    def get_dataset(self, dataset_group):
        """
        Read the dataset in the file.

        Parameters
        ----------
        dataset_group : str
            The path of the dataset in the h5 file.

        Returns
        -------
        dataset : ndarray
            The aimed dataset.

        """
        imgfile = h5py.File(self.pathsaveimg, "r")
        assert (dataset_group in imgfile), 'The aimed dataset does not exist, please check the path of the dataset again!'
        dataset = np.array(imgfile[dataset_group])
        imgfile.close()
        return dataset

    def save_dataset(self, dataset_group, filename):
        """
        Save the dataset in npz format.

        Parameters
        ----------
        dataset_group : str
            The path of the dataset in the h5 file.

        Returns
        -------
        None.

        """
        imgfile = h5py.File(self.pathsaveimg, "r")
        assert (dataset_group in imgfile), 'The aimed dataset does not exist, please check the path of the dataset again!'
        dataset = np.array(imgfile['dataset_group'])
        np.savez(os.path.join(self.pathsave, filename), data=dataset)
        imgfile.close()
        return

    def add_para(self, para_name, para_value):
        """
        Add a parameter to the parameter list.

        Parameters
        ----------
        para_name : str
            The name of the aimed parameter.
        para_value : object
            The value of the aimed parameter.

        Returns
        -------
        None.

        """
        self.para_dict[para_name] = para_value
        return

    def get_para(self, para_name):
        """
        Get parameter from the memory.

        Parameters
        ----------
        para_name : str
            The name of the parameters.

        Returns
        -------
        object
            The value of the desired parameter.

        """
        try:
            return self.para_dict[para_name]
        except KeyError:
            # print('The desired parameter %s does not exist in memory, please check it again!' % para_name)
            return None

    def del_para(self, para_name):
        """
        Delete the parameter in the memory.

        Parameters
        ----------
        para_name : str
            The name of the parameter.

        Returns
        -------
        None.

        """
        del self.para_dict[para_name]
        return

    def save_para_list(self, para_name_list=None):
        """
        Save the parameters in the image file as attributes in h5 file.

        Parameters
        ----------
        para_name_list : list, optional
            List of the parameters to be save. If None, all the parameters will be saved in the image file. The default is None.

        Returns
        -------
        None.

        """
        imgfile = h5py.File(self.pathsaveimg, "r+")
        if para_name_list is None:
            for para_name in self.para_dict:
                imgfile.attrs[para_name] = self.para_dict[para_name]
        else:
            for para_name in para_name_list:
                imgfile.attrs[para_name] = self.para_dict[para_name]
        return

    def load_para_from_infor_file(self, pathinfor, para_name_list, section=''):
        """
        Load the parameters in the information file.

        Parameters
        ----------
        pathinfor : str
            Path of the information file.
        para_name_list : list
            List of the parameter names to be imported.
        section : str, optional
            The section name of the aimed parameter in the information file.
            The parameter should be given, if the parameter name is not unique in the previous information file.The default is ''.

        Returns
        -------
        None.

        """
        assert os.path.exists(pathinfor), 'The information file does not exist! Please check the path of the information file again!'
        infor = Information_file_io(pathinfor)
        infor.infor_reader()
        for para_name in para_name_list:
            self.para_dict[para_name] = infor.get_para_value(para_name, section)
        return

    def save_para_to_infor_file(self, pathinfor, section, para_name_list=None):
        """
        Save the parameters to the aimed infomration file.

        Parameters
        ----------
        pathinfor : str
            The path for the information file.
        section : str
            The section name to be stored in the information file.
        para_name_list : list, optional
            The list of parameters to be saved, if the parameter exist in the memory. The default is None.

        Returns
        -------
        None.

        """
        infor = Information_file_io(pathinfor)
        infor.infor_reader()
        infor.del_para_section(section)
        if para_name_list is None:
            for para_name in self.para_dict:
                infor.add_para(para_name, section, self.para_dict[para_name])
        else:
            for para_name in para_name_list:
                if self.get_para(para_name) is not None:
                    infor.add_para(para_name, section, self.para_dict[para_name])
        infor.infor_writer()
        return

    def plot_3D_result(self, array_group, array_names, voxel_size, display_range=None, title='', save_image=True, filename='', save_as_vti=False):
        """
        Plot and save the 3D phase retrieval results.

        Parameters
        ----------
        array_group : str
            The group name, where the arrays are stored in the h5 file.
        array_names : list
            The array names to be plotted.
        voxel_size : tuple
            The unit of different axis in Z, Y, X order.
        display_range : list, optional
            The range for the display in Z, Y, X order. The default is None.
        title : str, optional
            The title of the whole plot. The default is ''.
        save_image : bool, optional
            If ture, the image will be saved in the pathsave folder. The default is True.
        filename : str, optional
            The filename to save the images. The default is ''.
        save_as_vti : bool, optional
            If ture, the arrays will be saved as vti format for 3D visualization. The default is False.

        Returns
        -------
        None.

        """
        imgfile = h5py.File(self.pathsaveimg, "r+")
        num_of_arrays = len(array_names)
        plt_arrays = ()

        fig, axs = plt.subplots(num_of_arrays, 3, figsize=(24, num_of_arrays * 8))
        if title != '':
            fig.suptitle(title, fontsize=28)
        for i in range(num_of_arrays):
            plt_array = np.array(imgfile['%s/%s' % (array_group, array_names[i])])
            if ('Modulus' in array_names[i]) or ('modulus' in array_names[i]):
                plt_array = plt_array / np.amax(plt_array)
            plt_arrays = plt_arrays + (plt_array,)
            if i == 0:
                zd, yd, xd = plt_array.shape
                unitz = np.arange(-zd / 2 + 0.5, zd / 2 + 0.5) * voxel_size[0]
                unity = np.arange(-yd / 2 + 0.5, yd / 2 + 0.5) * voxel_size[1]
                unitx = np.arange(-xd / 2 + 0.5, xd / 2 + 0.5) * voxel_size[2]
                if display_range is None:
                    display_range = np.zeros(3, dtype=int)
                    display_range[0] = int(np.amax(unitz) / 2.0)
                    display_range[1] = int(np.amax(unity) / 2.0)
                    display_range[2] = int(np.amax(unitx) / 2.0)
            color_bar_range = np.linspace(np.amin(plt_array), np.amax(plt_array), 150)
            im = axs[i, 0].contourf(unitx, unity, plt_array[int(zd / 2), :, :], color_bar_range, cmap='jet')
            plt.colorbar(im, ax=axs[i, 0], shrink=0.6)
            axs[i, 0].set_title('%s z cut' % array_names[i], fontsize=24)
            axs[i, 0].set_xlabel('x (nm)', fontsize=24)
            axs[i, 0].set_ylabel('y (nm)', fontsize=24)
            axs[i, 0].axis('scaled')
            axs[i, 0].set_xlim(-display_range[2], display_range[2])
            axs[i, 0].set_ylim(-display_range[1], display_range[1])
            im = axs[i, 1].contourf(unitx, unitz, plt_array[:, int(yd / 2), :], color_bar_range, cmap='jet')
            plt.colorbar(im, ax=axs[i, 1], shrink=0.6)
            axs[i, 1].set_title('%s y cut' % array_names[i], fontsize=24)
            axs[i, 1].set_xlabel('x (nm)', fontsize=24)
            axs[i, 1].set_ylabel('z (nm)', fontsize=24)
            axs[i, 1].axis('scaled')
            axs[i, 1].set_xlim(-display_range[2], display_range[2])
            axs[i, 1].set_ylim(-display_range[0], display_range[0])
            im = axs[i, 2].contourf(unity, unitz, plt_array[:, :, int(xd / 2)], color_bar_range, cmap='jet')
            plt.colorbar(im, ax=axs[i, 2], shrink=0.6)
            axs[i, 2].set_title('%s x cut' % array_names[i], fontsize=24)
            axs[i, 2].set_xlabel('y (nm)', fontsize=24)
            axs[i, 2].set_ylabel('z (nm)', fontsize=24)
            axs[i, 2].axis('scaled')
            axs[i, 2].set_xlim(-display_range[1], display_range[1])
            axs[i, 2].set_ylim(-display_range[0], display_range[0])
        fig.tight_layout()
        if filename == '':
            filename = "Trial%d" % (self.para_dict['trial_num'])
        if save_image:
            plt.savefig(os.path.join(self.pathsave, filename + '.png'))
            plt.close()
        else:
            plt.show()
        imgfile.close()
        if save_as_vti:
            pathsavevti = os.path.join(self.pathsave, filename + '.vti')
            pp.save_to_vti(pathsavevti, plt_arrays, array_names, voxel_size=voxel_size)
        return

    def plot_2D_result(self, array_group, array_names, voxel_size, display_range=None, title='', subplot_config=None, save_image=True, filename=''):
        """
        Plot and save the 2D phase retrieval results.

        Parameters
        ----------
        array_group : str
            The group name, where the arrays are stored in the h5 file.
        array_names : list
            The array names to be plotted.
        voxel_size : tuple
            The unit of different axis in Z, Y, X order.
        display_range :  list, optional
            The range for the display in Z, Y, X order. The default is None.
        title : str, optional
            The title of the whole plot. The default is ''.
        subplot_config : tuple, optional
            The row and column number of the subplots. The default is None.
        save_image : bool, optional
            If ture, the image will be saved in the pathsave folder. The default is True.
        filename : str, optional
            The filename to save the images. The default is ''.

        Returns
        -------
        None.

        """
        imgfile = h5py.File(self.pathsaveimg, "r+")
        num_of_arrays = len(array_names)
        if subplot_config is None:
            fig, axs = plt.subplots(1, num_of_arrays, figsize=(num_of_arrays * 8, 8))
        else:
            assert num_of_arrays <= (subplot_config[0] * subplot_config[1]), 'The too many images to be plotted, please increase the number in the subplots_config.'
            fig, axs = plt.subplots(subplot_config[0], subplot_config[1], figsize=(subplot_config[1] * 8, subplot_config[0] * 8))
            axs = axs.flatten()
        if title != '':
            fig.suptitle(title, fontsize=28)
        for i in range(num_of_arrays):
            plt_array = np.array(imgfile['%s/%s' % (array_group, array_names[i])])
            if ('Modulus' in array_names[i]) or ('modulus' in array_names[i]):
                plt_array = plt_array / np.amax(plt_array)
            if i == 0:
                yd, xd = plt_array.shape
                unity = np.arange(-yd / 2 + 0.5, yd / 2 + 0.5) * voxel_size[0]
                unitx = np.arange(-xd / 2 + 0.5, xd / 2 + 0.5) * voxel_size[1]
                if display_range is None:
                    display_range = np.zeros(2, dtype=int)
                    display_range[0] = int(np.amax(unity) / 2.0)
                    display_range[1] = int(np.amax(unitx) / 2.0)
            color_bar_range = np.linspace(np.amin(plt_array), np.amax(plt_array), 150)
            im = axs[i].contourf(unitx, unity, plt_array, color_bar_range, cmap='jet')
            plt.colorbar(im, ax=axs[i], shrink=0.6)
            if self.get_para('data_description') == 'cutqx':
                axs[i].set_title('%s qx cut' % array_names[i], fontsize=24)
                axs[i].set_xlabel('y (nm)', fontsize=24)
                axs[i].set_ylabel('z (nm)', fontsize=24)
            elif self.get_para('data_description') == 'cutqy':
                axs[i].set_title('%s qy cut' % array_names[i], fontsize=24)
                axs[i].set_xlabel('x (nm)', fontsize=24)
                axs[i].set_ylabel('z (nm)', fontsize=24)
            elif self.get_para('data_description') == 'cutqz':
                axs[i].set_title('%s qz cut' % array_names[i], fontsize=24)
                axs[i].set_xlabel('x (nm)', fontsize=24)
                axs[i].set_ylabel('y (nm)', fontsize=24)
            elif self.get_para('data_description') == 'cuty':
                if array_group == 'Ortho':
                    axs[i].set_title('%s y cut (orthogonal)' % array_names[i], fontsize=24)
                    axs[i].set_xlabel('x (nm)', fontsize=24)
                    axs[i].set_ylabel('z (nm)', fontsize=24)
                else:
                    axs[i].set_title('%s y cut (non-orthogonal)' % array_names[i], fontsize=24)
                    axs[i].set_xlabel('x (nm)', fontsize=24)
                    axs[i].set_ylabel('z (nm)', fontsize=24)

            axs[i].axis('scaled')
            axs[i].set_xlim(-display_range[1], display_range[1])
            axs[i].set_ylim(-display_range[0], display_range[0])
        fig.tight_layout()
        if filename == '':
            filename = "Trial%d" % (self.para_dict['trial_num'])
        if save_image:
            plt.savefig(os.path.join(self.pathsave, filename + '.png'))
            plt.close()
        else:
            plt.show()
        imgfile.close()
        return

    def plot_3D_intensity(self, save_image=True, filename=''):
        """
        Plot the calculated intensity and the measured intensity from the phase retrieval results.

        Parameters
        ----------
        save_image : bool, optional
            If ture, the image will be saved in the pathsave folder. The default is True.
        filename : str, optional
            The filename to save the images. The default is ''.

        Returns
        -------
        None.

        """
        imgfile = h5py.File(self.pathsaveimg, "r+")
        measured_intensity = np.array(imgfile["Input/intensity"], dtype=float)
        calculated_intensity = np.array(imgfile["Average_All/intensity_sum"], dtype=float)
        imgfile.close()
        zd, yd, xd = measured_intensity.shape
        fig, axs = plt.subplots(3, 3, figsize=(24, 24))
        im = axs[0, 0].imshow(np.log10(calculated_intensity[int(zd / 2), :, :] + 1.0), cmap="jet")
        plt.colorbar(im, ax=axs[0, 0], shrink=0.6)
        axs[0, 0].axis('off')
        axs[0, 0].set_title('Retrieved intensity Qz', fontsize=24)
        im = axs[0, 1].imshow(np.log10(calculated_intensity[:, int(yd / 2), :] + 1.0), cmap="jet")
        plt.colorbar(im, ax=axs[0, 1], shrink=0.6)
        axs[0, 1].axis('off')
        axs[0, 1].set_title('Retrieved intensity Qy', fontsize=24)
        im = axs[0, 2].imshow(np.log10(calculated_intensity[:, :, int(xd / 2)] + 1.0), cmap="jet")
        plt.colorbar(im, ax=axs[0, 2], shrink=0.6)
        axs[0, 2].axis('off')
        axs[0, 2].set_title('Retrieved intensity Qx', fontsize=24)
        im = axs[1, 0].imshow(np.log10(measured_intensity[int(zd / 2), :, :] + 1.0), cmap="jet")
        plt.colorbar(im, ax=axs[1, 0], shrink=0.6)
        axs[1, 0].axis('off')
        axs[1, 0].set_title('Measured intensity Qz', fontsize=24)
        im = axs[1, 1].imshow(np.log10(measured_intensity[:, int(yd / 2), :] + 1.0), cmap="jet")
        plt.colorbar(im, ax=axs[1, 1], shrink=0.6)
        axs[1, 1].axis('off')
        axs[1, 1].set_title('Measured intensity Qy', fontsize=24)
        im = axs[1, 2].imshow(np.log10(measured_intensity[:, :, int(xd / 2)] + 1.0), cmap="jet")
        plt.colorbar(im, ax=axs[1, 2], shrink=0.6)
        axs[1, 2].axis('off')
        axs[1, 2].set_title('Measured intensity Qx', fontsize=24)
        im = axs[2, 0].imshow((calculated_intensity - measured_intensity)[int(zd / 2), :, :], cmap="jet", vmax=1.0e3, vmin=-1.0e3)
        plt.colorbar(im, ax=axs[2, 0], shrink=0.6)
        axs[2, 0].axis('off')
        axs[2, 0].set_title('Intensity difference Qz', fontsize=24)
        im = axs[2, 1].imshow((calculated_intensity - measured_intensity)[:, int(yd / 2), :], cmap="jet", vmax=1.0e3, vmin=-1.0e3)
        plt.colorbar(im, ax=axs[2, 1], shrink=0.6)
        axs[2, 1].axis('off')
        axs[2, 1].set_title('Intensity difference Qy', fontsize=24)
        im = axs[2, 2].imshow((calculated_intensity - measured_intensity)[:, :, int(xd / 2)], cmap="jet", vmax=1.0e3, vmin=-1.0e3)
        plt.colorbar(im, ax=axs[2, 2], shrink=0.6)
        axs[2, 2].axis('off')
        axs[2, 2].set_title('Intensity difference Qx', fontsize=24)
        fig.tight_layout()
        if filename == '':
            filename = "Intensity_difference_Trial%d.png" % (self.para_dict['trial_num'])
        if save_image:
            plt.savefig(os.path.join(self.pathsave, filename))
            plt.close()
        else:
            plt.show()
        return

    def plot_2D_intensity(self, save_image=True, filename=''):
        """
        Plot the calculated intensity and the measured intensity from the phase retrieval results.

        Parameters
        ----------
        pathsave : str, optional
            The folder path to save the images. If the path folder does not exist, the image will not be saved. The default is ''.
        filename : str, optional
            The filename to save the images. The default is ''.

        Returns
        -------
        None.

        """
        imgfile = h5py.File(self.pathsaveimg, "r+")
        measured_intensity = np.array(imgfile["Input/intensity"], dtype=float)
        calculated_intensity = np.array(imgfile["Average_All/intensity_sum"], dtype=float)
        imgfile.close()
        yd, xd = measured_intensity.shape
        fig, axs = plt.subplots(1, 3, figsize=(24, 8))
        im = axs[0].imshow(np.log10(calculated_intensity + 1.0), cmap="jet")
        plt.colorbar(im, ax=axs[0], shrink=0.6)
        axs[0].axis('off')
        axs[0].set_title('Retrieved intensity', fontsize=24)
        im = axs[1].imshow(np.log10(measured_intensity + 1.0), cmap="jet")
        plt.colorbar(im, ax=axs[1], shrink=0.6)
        axs[1].axis('off')
        axs[1].set_title('Measured intensity Qz', fontsize=24)
        im = axs[2].imshow((calculated_intensity - measured_intensity), cmap="jet", vmax=1.0e3, vmin=-1.0e3)
        plt.colorbar(im, ax=axs[2], shrink=0.6)
        axs[2].axis('off')
        axs[2].set_title('Intensity difference Qy', fontsize=24)
        fig.tight_layout()
        if filename == '':
            filename = "Intensity_difference_Trial%d.png" % (self.para_dict['trial_num'])
        if save_image:
            plt.savefig(os.path.join(self.pathsave, filename))
            plt.close()
        else:
            plt.show()
        return

    def plot_error_matrix(self, save_image=True, filename=''):
        """
        Plot the error matrix for the phase retrieval results.

        Parameters
        ----------
        pathsave : str, optional
            The path to save the result plot. The default is ''.
        filename : str, optional
            The filename to save the result plot. The default is ''.

        Returns
        -------
        None.

        """
        imgfile = h5py.File(self.pathsaveimg, "r+")
        err_ar = np.array(imgfile["Error/error"], dtype=float)
        PRTF = np.array(imgfile["Average_All/phase_retrieval_transfer_function"], dtype=float)
        imgfile.close()
        if self.para_dict['support_update']:
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
            axs[1, 1].plot(PRTF, 'r.')
            axs[1, 1].set_title('Phase retrieval transfer function', fontsize=24)
            axs[1, 1].set_xlabel('pixel', fontsize=24)
            axs[1, 1].set_ylabel('PRTF', fontsize=24)
            fig.tight_layout()
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
            axs[1, 1].plot(PRTF, 'r.')
            axs[1, 1].set_title('Phase retrieval transfer function', fontsize=24)
            axs[1, 1].set_xlabel('pixel', fontsize=24)
            axs[1, 1].set_ylabel('PRTF', fontsize=24)
        fig.tight_layout()
        if filename == '':
            filename = "Error_Trial%d.png" % (self.para_dict['trial_num'])
        if save_image:
            plt.savefig(os.path.join(self.pathsave, filename))
            plt.close()
        else:
            plt.show()
        return


def plt_result_2D_simple(plt_arrays, array_names, pathsave='', filename=''):
    """
    Plot cuts of the 2D arrays without units.

    Parameters
    ----------
    plt_arrays : tuple
        tuple of arrays for the plot.
    array_names : tuple
        tuple of array names.
    pathsave : str, optional
        the path for saveing the plot, if not given, the plot will not be saved. The default is ''.
    filename : str, optional
        the filename for saving the plot. The default is ''.

    Returns
    -------
    None.

    """
    num_of_arrays = len(plt_arrays)
    if num_of_arrays != 1:
        fig, axs = plt.subplots(1, num_of_arrays, figsize=(num_of_arrays * 4, 4))
        for i in range(num_of_arrays):
            plt_array = plt_arrays[i]
            yd, xd = plt_array.shape
            im = axs[i].imshow(plt_array[int(yd / 4):int(yd * 3 / 4), int(xd / 4):int(xd * 3 / 4)])
            axs[i].set_xlabel('pixel', fontsize=24)
            axs[i].set_ylabel('pixel', fontsize=24)
            plt.colorbar(im, ax=axs[i], shrink=0.6)
            axs[i].set_title('%s' % array_names[i], fontsize=24)
        fig.tight_layout()
    else:
        plt_array = plt_arrays[0]
        yd, xd = plt_array.shape
        plt.imshow(plt_array[int(yd / 4):int(yd * 3 / 4), int(xd / 4):int(xd * 3 / 4)])
        plt.xlabel('pixel', fontsize=24)
        plt.ylabel('pixel', fontsize=24)
        plt.colorbar(shrink=0.6)
        plt.title('%s' % array_names[0], fontsize=24)

    if os.path.exists(pathsave):
        plt.savefig(os.path.join(pathsave, filename))
    else:
        plt.show()
    plt.close()
    return


def plt_result_3D_simple(plt_arrays, array_names, pathsave='', filename=''):
    """
    Plot cuts of the 3D arrays without units.

    Parameters
    ----------
    plt_arrays : tuple
        tuple of arrays for the plot.
    array_names : tuple
        tuple of array names.
    pathsave : str, optional
        the path for saveing the plot, if not given, the plot will not be saved. The default is ''.
    filename : str, optional
        the filename for saving the plot. The default is ''.

    Returns
    -------
    None.

    """
    num_of_arrays = len(plt_arrays)
    fig, axs = plt.subplots(num_of_arrays, 3, figsize=(12, num_of_arrays * 4))
    if num_of_arrays != 1:
        for i in range(num_of_arrays):
            plt_array = plt_arrays[i]
            zd, yd, xd = plt_array.shape
            im = axs[i, 0].imshow(plt_array[int(zd / 2), int(yd / 4):int(yd * 3 / 4), int(xd / 4):int(xd * 3 / 4)])
            axs[i, 0].set_xlabel('x (pixel)', fontsize=24)
            axs[i, 0].set_ylabel('y (pixel)', fontsize=24)
            plt.colorbar(im, ax=axs[i, 0], shrink=0.6)
            axs[i, 0].set_title('%s z cut' % array_names[i], fontsize=24)
            im = axs[i, 1].imshow(plt_array[int(zd / 4):int(zd * 3 / 4), int(yd / 2), int(xd / 4):int(xd * 3 / 4)])
            axs[i, 1].set_xlabel('x (pixel)', fontsize=24)
            axs[i, 1].set_ylabel('z (pixel)', fontsize=24)
            plt.colorbar(im, ax=axs[i, 1], shrink=0.6)
            axs[i, 1].set_title('%s y cut' % array_names[i], fontsize=24)
            im = axs[i, 2].imshow(plt_array[int(zd / 4):int(zd * 3 / 4), int(yd / 4):int(yd * 3 / 4), int(xd / 2)])
            axs[i, 2].set_xlabel('y (pixel)', fontsize=24)
            axs[i, 2].set_ylabel('z (pixel)', fontsize=24)
            plt.colorbar(im, ax=axs[i, 2], shrink=0.6)
            axs[i, 2].set_title('%s x cut' % array_names[i], fontsize=24)
    else:
        plt_array = plt_arrays[0]
        zd, yd, xd = plt_array.shape
        im = axs[0].imshow(plt_array[int(zd / 2), int(yd / 4):int(yd * 3 / 4), int(xd / 4):int(xd * 3 / 4)])
        axs[0].set_xlabel('x (pixel)', fontsize=24)
        axs[0].set_ylabel('y (pixel)', fontsize=24)
        plt.colorbar(im, ax=axs[0], shrink=0.6)
        axs[0].set_title('%s z cut' % array_names[0], fontsize=24)
        im = axs[1].imshow(plt_array[int(zd / 4):int(zd * 3 / 4), int(yd / 2), int(xd / 4):int(xd * 3 / 4)])
        axs[1].set_xlabel('x (pixel)', fontsize=24)
        axs[1].set_ylabel('z (pixel)', fontsize=24)
        plt.colorbar(im, ax=axs[1], shrink=0.6)
        axs[1].set_title('%s y cut' % array_names[0], fontsize=24)
        im = axs[2].imshow(plt_array[int(zd / 4):int(zd * 3 / 4), int(yd / 4):int(yd * 3 / 4), int(xd / 2)])
        axs[2].set_xlabel('y (pixel)', fontsize=24)
        axs[2].set_ylabel('z (pixel)', fontsize=24)
        plt.colorbar(im, ax=axs[2], shrink=0.6)
        axs[2].set_title('%s x cut' % array_names[0], fontsize=24)
    fig.tight_layout()
    if os.path.exists(pathsave):
        plt.savefig(os.path.join(pathsave, filename))
    else:
        plt.show()
    plt.close()
    return
