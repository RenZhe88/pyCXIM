# -*- coding: utf-8 -*-
"""
The common phase retrieval functions with pytorch based GPU acceleration.

Please install pytorch library before the running this code.
In case of questions, please contact us.

Created on Thu Mar 30 11:35:00 2023

@author: Ren Zhe, Xu Han
@email: renzhe@ihep.ac.cn, xuhan@ihep.ac.cn, or renzhetu001@gmail.com
"""

import numpy as np
import re
import sys
from scipy.special import gammaln
# from skimage.morphology import convex_hull_image
import torch


class PhaseRetrievalFuns():
    """
    Functions for the phase retrieval process.

    If the starting image is not given, generate a random phase for the starting image according to the given Seed.
    If the initial support is not given, generate the support according to the autocorrelation function.

    Parameters
    ----------
    measured_intensity : ndarray
        The 2D or 3D array containing measured diffraction intensity.
    Seed : int
        The seed to control the starting random start for phase retrieval.
    starting_img : ndarray, optional
        The complex array representing the starting image.
        The default is None.
    support : ndarray, optional
        The array defining the starting support for phase retrieval (
        containing data with boolean type).
        0(False) corresponds to the pixels outside the support area.
        1(True) corresponds to the pixels within the support area.
        if support is none, the autocorrelation function will be used to
        generate the initial support.
        The default is None.
    MaskFFT : ndarrary, optional
        The array defining the masked pixel.
        0 corresponds to the pixels that are not masked.
        1 corresponds to the pixels that are masked.
        masked pixels should be let free during phase retrieval process.
        The default is None.
        Then the all the pixels are considered not masked.

    Attributes
    ----------
    intensity : ndarray
        Measured intensity.
    ModulusFFT : ndarray
        Modulus calculated from the measured intensity.
    dim : int
        The dimension of the diffraction intensity.
    MaskFFT : ndarray
        Mask for the bad pixels in the measured intensity.
    loop_dict : dict
        records number of loops performed for different algorithms.
    img : ndarray
        the result image
    support : ndarray
        the result support
    """

    def __init__(self, measured_intensity, Seed, starting_img=None, support=None, MaskFFT=None, LLKmask=None, precision='64'):
        print('Seed %04d' % Seed)
        self.Seed = Seed

        # Set torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.determinsitic = True
        torch.manual_seed(Seed)

        if precision == '64':
            self.dtype_list = [np.float64, np.complex128, torch.float64, torch.complex128]
        elif precision == '32':
            self.dtype_list = [np.float32, np.complex64, torch.float32, torch.complex64]
        self.intensity = torch.tensor(measured_intensity, dtype=self.dtype_list[2], device=self.device)
        self.ModulusFFT = torch.fft.fftshift(torch.sqrt(self.intensity))
        self.dim = self.intensity.ndim

        # loop index records number of loops performed for different algorithms
        self.loop_dict = {}

        # generate or import the starting image for the phase retrieval process
        if starting_img is not None:
            print('Given starting image used.')
            self.img = torch.tensor(starting_img, dtype=self.dtype_list[3], device=self.device)
        else:
            print('Random starting image used.')
            self.img = torch.fft.ifftn(torch.multiply(self.ModulusFFT, torch.exp(1j * torch.rand_like(self.ModulusFFT, dtype=self.dtype_list[2]) * 2 * torch.pi)))
            # self.img = self.img.type(self.dtype_list[3])

        # generate according to the autocorrelation or import the starting support
        if support is not None:
            self.support = torch.tensor(support, dtype=self.dtype_list[2], device=self.device)
        else:
            self.support = torch.zeros_like(self.ModulusFFT, dtype=self.dtype_list[2], device=self.device)
            autocorr = torch.abs(torch.fft.fftshift(torch.fft.fftn(self.intensity)))
            # Gaussian filter to be added
            threshold = 4.0 / 1000.0 * (autocorr.max() - autocorr.min()) + autocorr.min()
            self.support[autocorr >= threshold] = 1.0

        # Indicate the bad pixels during the phase retrieval process
        if MaskFFT is not None:
            self.MaskFFT = torch.tensor(MaskFFT, dtype=self.dtype_list[2], device=self.device)
            self.MaskFFT = torch.fft.fftshift(self.MaskFFT)
        else:
            self.MaskFFT = torch.zeros_like(self.ModulusFFT, dtype=self.dtype_list[2], device=self.device)

        # Indicate the pixels masked for LLK calculation during the phase retrieval process
        if LLKmask is not None:
            self.LLKmask = torch.tensor(LLKmask, dtype=self.dtype_list[2], device=self.device)
        else:
            self.LLKmask = None

        self.err_ar_dict = {'Seed': self.get_SeedNum,
                            'Support size': self.get_support_size,
                            'Fourier space error': self.get_Fourier_space_error,
                            'Poisson logLikelihood error': self.get_Poisson_Likelihood_error,
                            'Object domain error': self.get_object_domain_error,
                            'Modulus STD': self.get_modulus_std,
                            'Free logLikelihood': self.get_Free_LogLikelihood_error
                            }
        return

    def End(self):
        """
        End of the phase retrieval process.

        Transfers the data from GPU to memory.
        Empty the GPU space.

        Returns
        -------
        None.

        """
        err_ar = []
        for err_name in self.err_ar_dict.keys():
            err_ar.append(self.err_ar_dict[err_name]())
        self.err_ar = np.array(err_ar)

        self.img = (self.img * self.support).cpu().numpy()
        self.support = self.support.cpu().numpy()
        self.MaskFFT = self.MaskFFT.cpu().numpy()
        torch.cuda.empty_cache()
        return

    def flip_img(self):
        """
        Flip the result image to remove the trivial solutions.

        Returns
        -------
        None.

        """
        self.loop_dict['flip'] = 1
        self.print_loop_num()

        self.img = np.flip(self.img)
        self.img = np.conjugate(self.img)
        self.support = np.flip(self.support)
        return

    def GaussianBlur(self, img, sigma):
        """
        GPU based Gaussian Blur for 2D or 3D images.

        Parameters
        ----------
        img : ndarray
            3D image array with 'float' type to be blurred.
        sigma : float
            Standard deviation for Gaussian kernel.

        Returns
        -------
        img : ndarray
            Blurred image with 'float' type.

        """
        img = img.unsqueeze(0).unsqueeze(0)
        kernel_size = max(3, min(51, int(2 * round(7.0 * sigma) + 1)))

        x = torch.linspace(-kernel_size // 2, kernel_size // 2, kernel_size,
                           dtype=self.dtype_list[2], device=self.device)
        kernel = torch.exp(-torch.square(x / sigma) / 2)
        kernel = kernel / torch.sum(kernel)
        for i in range(self.dim):
            gaussian1d_conv = kernel.view([1, 1] + [1] * i + [-1] + [1] * (self.dim - i - 1))
            padding = [0] * self.dim
            padding[i] = kernel_size // 2
            if self.dim == 2:
                img = torch.nn.functional.conv2d(img, gaussian1d_conv, stride=1, padding=tuple(padding))
            elif self.dim == 3:
                img = torch.nn.functional.conv3d(img, gaussian1d_conv, stride=1, padding=tuple(padding))
        img = img.squeeze(0).squeeze(0)
        return img

    def CenterSup(self):
        """
        Center the retrieved image according to the support.

        Returns
        -------
        None.

        """
        for i in range(self.dim):
            sum_axis = list(range(self.dim))
            sum_axis.pop(i)
            shift = torch.round(self.support.size()[i] / 2.0 - 0.5 - torch.sum(torch.arange(self.support.size()[i]).to(self.device) * torch.sum(self.support, axis=tuple(sum_axis)) / torch.sum(self.support)))
            self.support = torch.roll(self.support, int(shift), dims=i)
            self.img = torch.roll(self.img, int(shift), dims=i)
        return

    def Sup(self, Gaussiandelta, thrpara, hybrid_para=0):
        """
        Update the support according to the current images or average modulus of previous results.

        if hybrid_para is 0, the algorithm is the same as the old shrinkwrap method,
        where the modulus of the current image is used to updata the support
        else, the algorithm comobines the average modulus of the previous results with the current image,
        the support is updated according to this combination.

        In both cases, 3D Schrinkwarp method performed according to the paper:
                X-ray image reconstruction from a diffraction pattern alone
                S. Marchesini, et al. PRB, 68, 140101 (2003)

        Parameters
        ----------
        Gaussiandelta : float
            Standard deviation for Gaussian kernel.
        thrpara : float
            The threhold parameter for updating the suppoart.
            Value should be from 0 to 1.
        hybrid_para : float, optional
            The ratio of the avearge modulus to considered when updating the support.
            Should be between 0, and 1.
            The default is 0, where classical shrinkwrap method is used.

        Returns
        -------
        None.

        """
        if 'Sup' in self.loop_dict.keys():
            self.loop_dict['Sup'] += 1
        else:
            self.loop_dict['Sup'] = 1
        self.print_loop_num()

        if not hasattr(self, 'Modulus_sum'):
            # the array to hold the previous image during the HIO calculation
            self.Modulus_sum = torch.zeros_like(self.ModulusFFT, dtype=self.dtype_list[2]).to(self.device)

        # Update the support function according to the shrink wrap method
        # self.Modulus_sum += torch.abs(self.img)
        # Bluredimg = self.GaussianBlur(self.Modulus_sum / float(self.loop_dict['Sup']) * hybrid_para + (1.0 - hybrid_para) * torch.abs(self.img), Gaussiandelta)
        self.Modulus_sum = self.Modulus_sum * hybrid_para + torch.abs(self.img)
        # corr = (1 - hybrid_para) / (1.0 - np.power(hybrid_para, float(self.loop_dict['Sup'])))
        Bluredimg = self.GaussianBlur(self.Modulus_sum, Gaussiandelta)
        threshold = thrpara * (torch.amax(Bluredimg) - torch.amin(Bluredimg)) + torch.amin(Bluredimg)
        self.support = torch.zeros_like(self.img, dtype=self.dtype_list[2])
        self.support[Bluredimg >= threshold] = 1.0
        # Center the support to the center of the image
        if self.loop_dict['Sup'] % 10 == 0:
            self.CenterSup()
        return

    def ND(self, num_ND_loop, vl, vu):
        if 'ND' in self.loop_dict.keys():
            self.loop_dict['ND'] += num_ND_loop
        else:
            self.loop_dict['ND'] = num_ND_loop
        self.print_loop_num()

        self.img = self.img * self.support
        img_phase = torch.angle(self.img)
        img_modulus = torch.abs(self.img)
        for i in range(num_ND_loop):
            img_max = torch.amax(img_modulus)
            img_modulus -= vl * img_max
            img_modulus = img_modulus / (1 - vu - vl)
            img_modulus = torch.clip(img_modulus, 0, (1 - vu) * img_max)
        self.img = torch.multiply(img_modulus, torch.exp(1j * img_phase))
        return

    def ER(self, num_ER_loop):
        """
        Use error reduction method to update the image.

        Error reduction performed according to the paper:
            Phase retrieval by iterated projections
            RW. Gerchberg, WO. Saxton, Optik, 35, 237 (1972)

        Parameters
        ----------
        num_ER_loop : int
            Number of error reduction loops to be performed

        Returns
        -------
        None.

        """
        if 'ER' in self.loop_dict.keys():
            self.loop_dict['ER'] += num_ER_loop
        else:
            self.loop_dict['ER'] = num_ER_loop
        self.print_loop_num()

        for i in range(num_ER_loop):
            self.img = self.support * self.img
            self.img = self.ModulusProj(self.img)
        return

    def DETWIN(self, axis=0):
        """
        Remove part of the reconstructed image to reduce the effect of twinning in the reconstruction.

        Parameters
        ----------
        axis : Union(int,tuple), optional
            The axis for half cutting the images. The default is 0.

        Returns
        -------
        None.

        """
        if 'DETWIN' in self.loop_dict.keys():
            self.loop_dict['DETWIN'] += 1
        else:
            self.loop_dict['DETWIN'] = 1
        self.print_loop_num()

        if isinstance(axis, int):
            axis = (axis,)
        if self.dim == 2:
            yd, xd = self.img.size()

            if 0 in axis:
                self.img[int(yd / 2):, :] = 0
            if 1 in axis:
                self.img[:, int(xd / 2):] = 0
        elif self.dim == 3:
            zd, yd, xd = self.img.size()

            if 0 in axis:
                self.img[int(zd / 2):, :, :] = 0
            if 1 in axis:
                self.img[:, int(yd / 2):, :] = 0
            if 2 in axis:
                self.img[:, :, int(xd / 2):] = 0
        return

    # def ConvexSup(self):
    #     self.support = self.support.cpu().numpy()
    #     self.support = np.array(convex_hull_image(self.support), dtype=float)
    #     self.support = torch.from_numpy(self.support).to(self.device)
    #     return

    def HIO(self, num_HIO_loop):
        """
        Use hybrid input and output (HIO) method to update the image.

        HIO performed according to the paper:
            Phase retrieval algorithms: a comparison
            J.R. Fienup. Applied Optics, 21, 2758 (1982)

        Parameters
        ----------
        num_HIO_loop : int
            Number of HIO loops to be performed

        Returns
        -------
        None.

        """
        if 'HIO' in self.loop_dict.keys():
            self.loop_dict['HIO'] += num_HIO_loop
        else:
            self.loop_dict['HIO'] = num_HIO_loop
        self.print_loop_num()

        if not hasattr(self, 'holderimg_HIO'):
            # the array to hold the previous image during the HIO calculation
            self.holderimg_HIO = torch.zeros_like(self.ModulusFFT, dtype=self.dtype_list[3]).to(self.device)

        para = 0.9  # Parameter for the HIO

        self.img = self.support * self.img
        for i in range(num_HIO_loop):
            self.holderimg_HIO = self.support * self.img + (1.0 - self.support) * (self.holderimg_HIO - self.img * para)
            self.img = self.ModulusProj(self.holderimg_HIO)
        return

    def NHIO(self, num_NHIO_loop):
        """
        Use noise robust HIO method to update the image.

        Noise robust HIO performed according to the paper:
            Noise-robust coherent diffractive imaging with a single diffraction pattern
            A. Martin, et al. Optics Express, 20, 16650(2012)

        Parameters
        ----------
        num_NHIO_loop : int
            Number of loops to be performed.

        Returns
        -------
        None.

        """
        if 'NHIO' in self.loop_dict.keys():
            self.loop_dict['NHIO'] += num_NHIO_loop
        else:
            self.loop_dict['NHIO'] = num_NHIO_loop
        self.print_loop_num()

        if not hasattr(self, 'std_noise'):
            # The standarded deviation of noise used for NHIO method
            self.holderimg_NHIO = torch.zeros_like(self.ModulusFFT, dtype=self.dtype_list[3]).to(self.device)
            self.std_noise = torch.sqrt(torch.sum(self.intensity * (1.0 - self.MaskFFT)) / torch.sum(1.0 - self.MaskFFT))

        para = 0.9  # Parameter for the HIO

        self.img = self.support * self.img
        for i in range(num_NHIO_loop):
            self.holderimg_NHIO = self.support * self.img + (1.0 - self.support) * (self.holderimg_NHIO - self.img * para)
            zero_select_con = torch.logical_and(self.support == 0, torch.abs(self.holderimg_NHIO) < 3.0 * self.std_noise)
            self.holderimg_NHIO[zero_select_con] = 0
            self.img = self.ModulusProj(self.holderimg_NHIO)
        return

    def RAAR(self, num_RAAR_loop):
        """
        Use relaxed averaged alternating reflections(RAAR) method to update the image.

        RAAR performed according to the paper:
            Relaxed averaged alternating reflections for diffraction imaging
            D.R. Luke. Inverse Problems, 21, 37–50, (2005)

        Parameters
        ----------
        num_RAAR_loop : int
            Number of loops to be performed.

        Returns
        -------
        None.

        """
        if 'RAAR' in self.loop_dict.keys():
            self.loop_dict['RAAR'] += num_RAAR_loop
        else:
            self.loop_dict['RAAR'] = num_RAAR_loop
        self.print_loop_num()

        if not hasattr(self, 'holderimg_RAAR'):
            # the array to hold the previous image during the RAAR calculation
            self.holderimg_RAAR = torch.zeros_like(self.ModulusFFT, dtype=self.dtype_list[3]).to(self.device)

        para0 = 0.75  # Starting parameter for the RAAR

        self.img = self.support * self.img
        for i in range(num_RAAR_loop):
            para = para0 + (1.0 - para0) * (1.0 - np.exp(-(i / 12.0)**3.0))
            self.holderimg_RAAR = self.support * self.img + (1 - self.support) * (para * self.holderimg_RAAR + (1 - 2.0 * para) * self.img)
            self.img = self.ModulusProj(self.holderimg_RAAR)
        return

    def ModulusProj(self, point):
        """
        Modulus projection.

        Parameters
        ----------
        point : ndarray
            Image to be projected (data with complex type).

        Returns
        -------
        point : ndarray
            Updated imaged after modulus projection (data with complex type).

        """
        AmpFFT = torch.fft.fftn(point)
        AmpFFT = (1.0 - self.MaskFFT) * torch.multiply(self.ModulusFFT, torch.exp(1j * torch.angle(AmpFFT))) + 0.98 * self.MaskFFT * AmpFFT
        # AmpFFT = (1.0 - self.MaskFFT) * torch.multiply(self.ModulusFFT, torch.exp(1j * torch.angle(AmpFFT))) + self.MaskFFT * torch.fft.fftn(self.support * self.img)
        point = torch.fft.ifftn(AmpFFT)
        return point

    def SupProj(self, point):
        """
        Suppport projection.

        Parameters
        ----------
        point : ndarray
            Image (data with complex type).

        Returns
        -------
        point : ndarray
            Updated imaged after support projection(data with complex type).

        """
        point = point * self.support
        return point

    def DIFFERNCE_MAP(self, num_DIF_loop):
        """
        Use difference map method to update the image.

        Difference map method performed according to the paper:
            Phase retrieval by iterated projections
            V. Elser, et al. Journal of the Optical Society of America A, 40, 20, (2003)

        Parameters
        ----------
        num_DIF_loop : int
            Number of loops to be performed.

        Returns
        -------
        None.

        """
        if 'DIF' in self.loop_dict.keys():
            self.loop_dict['DIF'] += num_DIF_loop
        else:
            self.loop_dict['DIF'] = num_DIF_loop
        self.print_loop_num()

        para_dif = 0.9  # parameter for the difference map
        invpara = 1.0 / para_dif

        self.img = self.support * self.img
        for i in range(num_DIF_loop):
            map1 = (1 + invpara) * self.ModulusProj(self.img) - invpara * self.img
            map2 = (1 - invpara) * self.SupProj(self.img) + invpara * self.img
            self.img = self.img + para_dif * (self.SupProj(map1) - self.ModulusProj(map2))
        return

    def ADMM(self, num_ADMM_loop):
        """
        Use alternating direction method of multipliers (ADMM) to update the image.

        ADMM according to the lecture nots:
            http://faculty.bicmr.pku.edu.cn/~wenzw/bigdata/lect-phase.pdf

        Parameters
        ----------
        num_ADMM_loop : int
            Number of loops to be performed.

        Returns
        -------
        None.

        """
        if 'ADMM' in self.loop_dict.keys():
            self.loop_dict['ADMM'] += num_ADMM_loop
        else:
            self.loop_dict['ADMM'] = num_ADMM_loop
        self.print_loop_num()

        if not hasattr(self, 'scaled_dual_variable'):
            # the array to hold the previous image during the HIO calculation
            self.x = torch.zeros_like(self.ModulusFFT, dtype=self.dtype_list[3], device=self.device)
            self.scaled_dual_variable = torch.zeros_like(self.ModulusFFT, dtype=self.dtype_list[3], device=self.device)

        para_admm = 0.5
        self.img = self.support * self.img
        for i in range(num_ADMM_loop):
            self.x = self.SupProj(self.img - self.scaled_dual_variable)
            self.img = self.ModulusProj(self.x + self.scaled_dual_variable)
            self.scaled_dual_variable = self.scaled_dual_variable + para_admm * (self.x - self.img)
        return

    def get_SeedNum(self):
        """
        Get the Seed number.

        Returns
        -------
        int
            Seed Number.

        """
        return self.Seed

    def get_error_names(self):
        """
        Get the error names.

        Returns
        -------
        list
            List of error names.

        """
        return list(self.err_ar_dict.keys())

    def get_img(self):
        """
        Get the result image (real space).

        Should be used only after the phase retrieval calculation is finished.
        when the end function tranfers the result from GPU to memory.

        Returns
        -------
        ndarray
            Result image of the sample (data with complex type).

        """
        return self.img

    def get_support(self):
        """
        Get the result support (real space).

        Should be used only after the phase retrieval calculation is finished.
        when the end function tranfers the result from GPU to memory.

        Returns
        -------
        ndarray
            Result support of the sample (data with boolean type).

        """
        return self.support

    def get_img_Modulus(self):
        """
        Get the modulus value for the result image (real space).

        Should be used only after the phase retrieval calculation is finished.
        when the end function tranfers the result from GPU to memory.

        Returns
        -------
        ndarray
            Modulus of the result reconstruction (data with float type).

        """
        return np.array(np.abs(self.img) * self.support, dtype=self.dtype_list[0])

    def get_img_Phase(self):
        """
        Get the phase value for the result image (real space).

        Should be used only after the phase retrieval calculation is finished.
        when the end function tranfers the result from GPU to memory.
        Phase unwrap should be performed afterwards.

        Returns
        -------
        ndarray
            Phase of the result reconstruction (data with float type).

        """
        return np.array(np.angle(self.img) * self.support, dtype=self.dtype_list[0])

    def get_FFT_Modulus(self):
        """
        Get the Modulus value of the reconstructed intensity (reciprocal space).

        Should be used only after the phase retrieval calculation is finished.
        when the end function tranfers the result from GPU to memory.

        Returns
        -------
        ndarray
            Modulus of the reconstructed intensity (data with float type).

        """
        return np.array(np.abs(np.fft.fftshift(np.fft.fftn(self.img * self.support))), dtype=self.dtype_list[0])

    def get_FFT_amplitude(self):
        """
        Get the amplitude of the reconstructed intensity (reciprocal space).

        Should be used only after the phase retrieval calculation is finished.
        when the end function tranfers the result from GPU to memory.



        Returns
        -------
        ndarray
            Amplitude of the reconstructed intensity (data with complex type).

        """
        return np.fft.fftshift(np.fft.fftn(self.img * self.support)).astype(self.dtype_list[1])

    def get_intensity(self):
        """
        Get the reconstructed intensity (reciprocal space).

        Should be used only after the phase retrieval calculation is finished,
        when the end function tranfers the result from GPU to memory.

        Returns
        -------
        ndarray
            The reconstructed intensity (data with float type).

        """
        if torch.is_tensor(self.img):
            return torch.square(torch.abs(torch.fft.fftshift(torch.fft.fftn(self.img * self.support)))).type(self.dtype_list[2])
        else:
            return np.square(np.abs(np.fft.fftshift(np.fft.fftn(self.img * self.support)))).astype(self.dtype_list[0])

    def get_support_size(self):
        """
        Get the total size of the support.

        Returns
        -------
        int
            The total size of the support.

        """
        if torch.is_tensor(self.img):
            return torch.sum(self.support).item()
        else:
            return np.sum(self.support)

    def get_object_domain_error(self):
        """
        Calculate the object domain error of the result reconstruction.

        Returns
        -------
        object_domain_error : float
           Object domain error of the result reconstruction.

        """
        if torch.is_tensor(self.img):
            object_domain_error = torch.sum(torch.square(torch.abs(self.img * (1.0 - self.support)))) / torch.sum(torch.square(torch.abs(self.img)))
            return object_domain_error.item()
        else:
            object_domain_error = np.sum(np.square(np.abs(self.img * (1.0 - self.support)))) / np.sum(np.square(np.abs(self.img)))
            return object_domain_error

    def get_Fourier_space_error(self):
        """
        Calculate the fourier space error of the result reconstruction.

        Returns
        -------
        Fourier_space_error : float
            Fourier space error of the result reconstruction.

        """
        if torch.is_tensor(self.img):
            Fourier_space_error = torch.sum((torch.square(torch.abs(torch.fft.fftn(self.img * self.support)) - self.ModulusFFT) * (1.0 - self.MaskFFT))) / torch.sum(torch.square(self.ModulusFFT * (1.0 - self.MaskFFT)))
            return Fourier_space_error.item()
        else:
            Fourier_space_error = np.sum(np.square(np.abs(np.fft.fftn(self.img * self.support)) * (1.0 - self.MaskFFT) - self.ModulusFFT * (1.0 - self.MaskFFT))) / np.sum(np.square(self.ModulusFFT * (1.0 - self.MaskFFT)))
            return Fourier_space_error

    def get_Poisson_Likelihood_error(self):
        """
        Calculate the loglikelihood of the result reconstruction considering poisson distribution.

        Returns
        -------
        loglikelihood : float
            The loglikeihood error of the result reconstruction.

        """
        cal_inten = self.get_intensity()
        if torch.is_tensor(self.img):
            loglikelihood_error = torch.sum(torch.fft.fftshift(1.0 - self.MaskFFT) * (cal_inten + torch.lgamma(self.intensity + torch.finfo(self.dtype_list[2]).eps) - self.intensity * torch.log(cal_inten))) / torch.sum(1.0 - self.MaskFFT)
            return loglikelihood_error.item()
        else:
            loglikelihood_error = np.sum(np.fft.fftshift(1.0 - self.MaskFFT) * (cal_inten + gammaln(self.intensity + np.finfo(self.dtype_list[0]).eps) - self.intensity * np.log(cal_inten))) / np.sum(1.0 - self.MaskFFT)
            return loglikelihood_error

    def get_modulus_std(self):
        """
        Calculate the standarded deviation of the Modulus.

        Returns
        -------
        modulus_std : float
            The modulus_std error of the result reconstruction.

        """
        if torch.is_tensor(self.img):
            modulus_std = torch.std(torch.abs(self.img)[self.support == 1])
            return modulus_std.item()
        else:
            modulus_std = np.std(np.abs(self.img)[self.support == 1])
            return modulus_std

    def get_Free_LogLikelihood_error(self):
        """
        Calculate the Free loglikelihood error using the previously generated mask.

        Parameters
        ----------
        LLKmask : ndarray, optional
            The mask for the free loglikelihood calculation. The default is None.

        Returns
        -------
        loglikelihood : float
            Thre free loglikelihood error calculated.

        """
        if self.LLKmask is not None:
            cal_inten = self.get_intensity()
            if torch.is_tensor(self.img):
                loglikelihood = torch.sum(self.LLKmask * (cal_inten + torch.lgamma(self.intensity + torch.finfo(self.dtype_list[2]).eps) - self.intensity * torch.log(cal_inten))) / torch.sum(self.LLKmask)
                loglikelihood = loglikelihood.item()
            else:
                loglikelihood = np.sum(self.LLKmask * (cal_inten + gammaln(self.intensity + np.finfo(self.dtype_list[0]).eps) - self.intensity * np.log(cal_inten + np.finfo(self.dtype_list[0]).eps))) / np.sum(self.LLKmask)
        else:
            loglikelihood = 0
        return loglikelihood

    def get_error_array(self):
        """
        Get the error array generated at the end of Phase retrieval process

        Returns
        -------
        err_ar : ndarray
            The error array.

        """
        return self.err_ar

    def print_loop_num(self):
        """
        Print the loop numbers already performed.

        The print text would be like
        'HIO: 120 RAAR: 340 Sup: 120'

        Returns
        -------
        None.

        """
        display_str = '\r'
        for key in self.loop_dict.keys():
            display_str += '%s: %d ' % (key, self.loop_dict[key])
        sys.stdout.write(display_str)
        return

    def Algorithm_expander(self, algorithm0):
        """
        Expand the algorithm defined into steps to be operated.

        Parameters
        ----------
        algorithm0 : str
            The algorithm chain defining the Phase retrieval process.

        Example
        -------
        Algorithm_expander((HIO**50)**2*ER**10*Sup)
        [(HIO, 50), (HIO, 50), (ER, 10), (Sup, 1), (End, 1)]

        Returns
        -------
        steps : list
            List of the tuples defining the steps to be operated.

        """
        steps = []
        # Pattern 0 matches all three types of pattern like RAAR**50, (HIO**20*ER**20)**20, Sup
        pattern0 = r'(\w+\*\*\d+)|(\(.+?\)\*\*\d+)|([a-zA-Z]+)'
        # Pattern 1 matches pattern like RAAR**50
        pattern1 = r'(\w+)\*\*(\d+)'
        # Pattern 2 matches pattern like Sup
        pattern2 = r'([a-zA-Z]+)'
        # Pattern 3 matches pattern like (HIO**20*ER**20)**20
        pattern3 = r'\((.+?)\)\*\*(\d+)'

        for level0 in re.finditer(pattern0, algorithm0):
            if bool(re.match(pattern1, level0.group())):
                steps.append((re.findall(pattern1, level0.group())[0][0], int(re.findall(pattern1, level0.group())[0][1])))
            elif bool(re.match(pattern2, level0.group())):
                steps.append((re.findall(pattern2, level0.group())[0], 1))
            elif bool(re.match(pattern3, level0.group())):
                algorithm1, loopnum1 = re.findall(pattern3, level0.group())[0]
                for i in range(int(loopnum1)):
                    for level1 in re.finditer(pattern0, algorithm1):
                        if bool(re.match(pattern1, level1.group())):
                            steps.append((re.findall(pattern1, level1.group())[0][0], int(re.findall(pattern1, level1.group())[0][1])))
                        elif bool(re.match(pattern2, level1.group())):
                            steps.append((re.findall(pattern2, level1.group())[0], 1))
        steps.append(('End', 1))
        return steps


def Free_LLK_FFTmask(MaskFFT, percentage=4, r=3):
    """
    Generate the mask for the free loglikelihood calculation.

    Parameters
    ----------
    MaskFFT : ndarray
        The array defining the masked pixel.
        0 corresponds to the piexels that are not masked.
        1 corresponds to the pixels that are masked.
        masked pixels should be let free during phase retrieval process.
    percentage : flaot, optional
        The percentage of pixels that should be used for the Free LLk mask. The default is 4.
    r : int, optional
        The radius of pixels. The default is 3.

    Returns
    -------
    LLKmask : ndarray
        The free log-likelihood mask.
    MaskFFT : ndarray
        The real mask, which defines the free pixels in the reconstruction.

    """
    LLKmask = np.zeros_like(MaskFFT)
    if len(MaskFFT.shape) == 2:
        yd, xd = MaskFFT.shape
        npoints = int(percentage / 100 * yd * xd / r / r)
        ycen_ar = np.random.randint(r, yd - r, size=npoints)
        xcen_ar = np.random.randint(r, xd - r, size=npoints)
        for i in range(npoints):
            ycen = ycen_ar[i]
            xcen = xcen_ar[i]
            if not ((0.95 * yd / 2.0 < ycen < 1.05 * yd / 2) and (0.95 * xd / 2.0 < xcen < 1.05 * xd / 2)):
                LLKmask[(ycen - r):(ycen + r + 1), (xcen - r):(xcen + r + 1)] = 1.0
    elif len(MaskFFT.shape) == 3:
        zd, yd, xd = MaskFFT.shape
        npoints = int(percentage / 100 * zd * yd * xd / r / r / r)
        zcen_ar = np.random.randint(r, zd - r, size=npoints)
        ycen_ar = np.random.randint(r, yd - r, size=npoints)
        xcen_ar = np.random.randint(r, xd - r, size=npoints)
        for i in range(npoints):
            zcen = zcen_ar[i]
            ycen = ycen_ar[i]
            xcen = xcen_ar[i]
            if not ((0.95 * zd / 2.0 < zcen < 1.05 * zd / 2) and (0.95 * yd / 2.0 < ycen < 1.05 * yd / 2) and (0.95 * xd / 2.0 < xcen < 1.05 * xd / 2)):
                LLKmask[(zcen - r):(zcen + r + 1), (ycen - r):(ycen + r + 1), (xcen - r):(xcen + r + 1)] = 1.0
    LLKmask = (1.0 - MaskFFT) * LLKmask
    MaskFFT = MaskFFT + LLKmask
    return LLKmask, MaskFFT
