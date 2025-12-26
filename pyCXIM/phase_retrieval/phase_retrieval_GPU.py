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
        intensity = torch.tensor(measured_intensity, dtype=self.dtype_list[2], device=self.device)
        self.ModulusFFT = torch.fft.fftshift(torch.sqrt(intensity))
        self.dim = intensity.ndim

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
            autocorr = torch.abs(torch.fft.fftshift(torch.fft.fftn(intensity)))
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

        self.PC_cal = False

        self.err_ar_dict = {'Seed': self.get_SeedNum,
                            'Support size': self.get_support_size,
                            'Fourier space error': self.get_Fourier_space_error,
                            'Poisson logLikelihood error': self.get_Poisson_Likelihood_error,
                            'Object domain error': self.get_object_domain_error,
                            'Modulus STD': self.get_modulus_std,
                            'Difference map error': self.get_difference_map_error,
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
        if hasattr(self, 'psf'):
            self.psf = self.psf.cpu().numpy()
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

    def fftconvolve(self, point, kernel):
        """
        Convolution performed by FFT, similar to scipy fftconvolve.

        1. the result size is cutted to be the same size as the first image.
        2. two images should be real valued arrays.

        Parameters
        ----------
        point : Tensor
            The real input tensor.
        kernel : Tensor
            The real input tensor.

        Returns
        -------
        result : tensor
            The convolution results, whose size will be the same as point.

        """
        point_size = np.array(point.shape)
        kernel_size = np.array(kernel.shape)
        padded_size = tuple(point_size + kernel_size - 1)

        cut_start = np.array(kernel_size // 2, dtype=int)
        cut_end = np.array(cut_start + point_size, dtype=int)
        slices = tuple(slice(start, end) for start, end in zip(cut_start, cut_end))

        fftpoint = torch.fft.rfftn(point, s=padded_size)
        fftkernel = torch.fft.rfftn(kernel, s=padded_size)

        result = torch.fft.irfftn(fftpoint * fftkernel)
        result = result[slices]
        return result

    def GaussianKernel(self, sigma):
        """
        Generate gaussian kernel.

        Sigma value smaller than 3.5 is recommanded.

        Parameters
        ----------
        sigma : int, float, tuple
            The sigma value of the gaussian function.

        Returns
        -------
        gaussian : Tensor
            Gaussian kernel generated.

        """
        if isinstance(sigma, (int, float)):
            sizes = [max(3, min(51, int(2 * round(7.0 * sigma) + 1))) for _ in range(self.dim)]
            sigma_values = [sigma] * self.dim
        else:
            sizes = [max(3, min(51, int(2 * round(7.0 * s) + 1))) for s in sigma[:self.dim]]
            sigma_values = list(sigma) + [sigma[-1]] * (self.dim - len(sigma))
            sigma_values = sigma_values[:self.dim]

        squared_dist = torch.zeros(sizes, dtype=self.dtype_list[2], device=self.device)

        for i, size in enumerate(sizes):
            coord_sq = torch.square(torch.linspace(-size // 2, size // 2, size, dtype=self.dtype_list[2], device=self.device))

            view_shape = [1] * self.dim
            view_shape[i] = -1
            coord_sq = coord_sq.view(*view_shape)

            squared_dist += coord_sq / (2.0 * sigma_values[i] ** 2)

        gaussian = torch.exp(-squared_dist)
        gaussian = gaussian / torch.sum(gaussian)
        return gaussian

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
            shift = torch.round(self.support.size()[i] / 2.0 - torch.sum(torch.arange(self.support.size()[i]).to(self.device) * torch.sum(self.support, axis=tuple(sum_axis)) / torch.sum(self.support)))
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
        kernel = self.GaussianKernel(Gaussiandelta)
        Bluredimg = self.fftconvolve(self.Modulus_sum, kernel)
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
            self.std_noise = torch.sqrt(torch.sum(self.get_measured_intensity() * (1.0 - self.MaskFFT)) / torch.sum(1.0 - self.MaskFFT))

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

        Additional modulus projection mode with partial coherence according to the paper:
            High-resolution three-dimensional partially coherent diffraction imaging
            J.N. Clark. et al Nature Communications, 3, 993, (2012)

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
        if self.PC_cal:
            Amppc = torch.square(torch.abs(torch.fft.fftshift(AmpFFT))).type(self.dtype_list[2])
            Amppc = torch.sqrt(torch.fft.fftshift(torch.clamp(self.fftconvolve(Amppc, self.psf), min=0.1)))
            AmpFFT = (1.0 - self.MaskFFT) * torch.multiply(self.ModulusFFT / Amppc, AmpFFT) + 0.98 * self.MaskFFT * AmpFFT
        else:
            AmpFFT = (1.0 - self.MaskFFT) * torch.multiply(self.ModulusFFT, torch.exp(1j * torch.angle(AmpFFT))) + 0.98 * self.MaskFFT * AmpFFT
            # AmpFFT = (1.0 - self.MaskFFT) * torch.multiply(self.ModulusFFT, torch.exp(1j * torch.angle(AmpFFT))) + self.MaskFFT * torch.fft.fftn(self.support * self.img)
        point = torch.fft.ifftn(AmpFFT)
        return point

    def PSFon(self, sigma=1.2):
        """
        Initiate partial coherence according to the paper:
            High-resolution three-dimensional partially coherent diffraction imaging
            J.N. Clark. et al Nature Communications, 3, 993, (2012)

        Parameters
        ----------
        sigma : float, optional
            The gaussian detla value of initial PSF. The default is 1.2.

        Returns
        -------
        None.

        """
        if not ('PSFon' in self.loop_dict.keys()):
            if 'PSFoff' in self.loop_dict.keys():
                del self.loop_dict['PSFoff']
            self.loop_dict['PSFon'] = None
        self.print_loop_num()

        if not hasattr(self, 'psf'):
            self.psf = self.GaussianKernel(sigma)
        self.PC_cal = True
        return

    def PSFoff(self):
        """
        Close the partical coherence mode.

        Parameters
        ----------
        sigma : float, optional
            The gaussian detla value of initial PSF. The default is 1.2.

        Returns
        -------
        None.

        """
        if 'PSFon' in self.loop_dict.keys():
            del self.loop_dict['PSFon']
            self.loop_dict['PSFoff'] = None
        self.print_loop_num()

        self.PC_cal = False
        return

    def PSFupdate(self, num_PSF_loop, num_ER_loop=5):
        """
        Update the psf function for partial coherence calculation.

        Calculation according to the paper:
            High-resolution three-dimensional partially coherent diffraction imaging
            J.N. Clark. et al Nature Communications, 3, 993, (2012)

        Parameters
        ----------
        num_PSF_loop : int
            Number of luck-richardson algorithms to be performed for the psf retrieval.
        num_ER_loop : int, optional
            Number of ER loops to be performed to update intensity. The default is 5.

        Returns
        -------
        None.

        """
        if 'PSFupdate' in self.loop_dict.keys():
            self.loop_dict['PSFupdate'] += num_PSF_loop
        else:
            self.loop_dict['PSFupdate'] = num_PSF_loop
        self.print_loop_num()

        deltaIntensity = self.get_intensity()
        self.ER(num_ER_loop)
        deltaIntensity = 2.0 * self.get_intensity() - deltaIntensity

        img_size = np.array(deltaIntensity.shape)
        kernel_size = np.array(self.psf.shape)
        padded_size = tuple(img_size * 2 - 1)

        cut_start = (img_size - kernel_size // 2 - 1).astype(int)
        cut_end = (cut_start + kernel_size).astype(int)
        slices = tuple(slice(start, end) for start, end in zip(cut_start, cut_end))

        deltaIntenFFT = torch.fft.rfftn(torch.flip(deltaIntensity, tuple(range(self.dim))), s=padded_size)
        Intensity = self.get_measured_intensity()

        for i in range(num_PSF_loop):
            Amppc = Intensity / torch.clamp(self.fftconvolve(deltaIntensity, self.psf), min=0.1)
            Amppc = torch.fft.rfftn(Amppc, s=padded_size)
            Amppc = torch.fft.irfftn(deltaIntenFFT * Amppc)
            Amppc = Amppc[slices]

            self.psf = self.psf * Amppc
            self.psf = torch.clamp(self.psf, min=0)
            self.psf = self.psf / torch.sum(self.psf)
        return

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

        ADMM according to the lecture notes and paper:
            http://faculty.bicmr.pku.edu.cn/~wenzw/bigdata/lect-phase.pdf
            Z. Wen, et al. Inverse Problems 28, 115010, (2012)

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

    def get_psf(self):
        """
        Get the psf function calculated during partial coherence.

        Returns
        -------
        ndarray
            Result of the psf used for partial coherence calculation.

        """
        if hasattr(self, 'psf'):
            return self.psf
        else:
            return None

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

        Returns
        -------
        Tensor, ndarray
            The reconstructed intensity (data with float type).

        """
        if torch.is_tensor(self.img):
            return torch.square(torch.abs(torch.fft.fftshift(torch.fft.fftn(self.img * self.support)))).type(self.dtype_list[2])
        else:
            return np.square(np.abs(np.fft.fftshift(np.fft.fftn(self.img * self.support)))).astype(self.dtype_list[0])

    def get_measured_intensity(self):
        """
        Get the measured intensity (reciprocal space).

        Returns
        -------
        Tensor
            The reconstructed intensity (data with float type).

        """
        if torch.is_tensor(self.img):
            return torch.fft.fftshift(torch.square(self.ModulusFFT)).type(self.dtype_list[2])
        else:
            return np.fft.fftshift(np.square(self.ModulusFFT)).astype(self.dtype_list[0])

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

    def get_difference_map_error(self):
        if torch.is_tensor(self.img):
            difference_map_error = torch.linalg.norm(self.ModulusProj(self.SupProj(self.img)) - self.SupProj(self.img))
            return difference_map_error.item()
        else:
            # difference_map_error = np.sum(np.square(np.abs(self.img * (1.0 - self.support)))) / np.sum(np.square(np.abs(self.img)))
            return 0

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
        measured_inten = self.get_measured_intensity()
        if torch.is_tensor(self.img):
            loglikelihood_error = torch.sum(torch.fft.fftshift(1.0 - self.MaskFFT) * (cal_inten + torch.lgamma(measured_inten + torch.finfo(self.dtype_list[2]).eps) - measured_inten * torch.log(cal_inten))) / torch.sum(1.0 - self.MaskFFT)
            return loglikelihood_error.item()
        else:
            loglikelihood_error = np.sum(np.fft.fftshift(1.0 - self.MaskFFT) * (cal_inten + gammaln(measured_inten + np.finfo(self.dtype_list[0]).eps) - measured_inten * np.log(cal_inten))) / np.sum(1.0 - self.MaskFFT)
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
        measured_inten = self.get_measured_intensity()
        if self.LLKmask is not None:
            cal_inten = self.get_intensity()
            if torch.is_tensor(self.img):
                loglikelihood = torch.sum(self.LLKmask * (cal_inten + torch.lgamma(measured_inten + torch.finfo(self.dtype_list[2]).eps) - measured_inten * torch.log(cal_inten))) / torch.sum(self.LLKmask)
                loglikelihood = loglikelihood.item()
            else:
                loglikelihood = np.sum(self.LLKmask * (cal_inten + gammaln(measured_inten + np.finfo(self.dtype_list[0]).eps) - measured_inten * np.log(cal_inten + np.finfo(self.dtype_list[0]).eps))) / np.sum(self.LLKmask)
        else:
            loglikelihood = 0
        return loglikelihood

    def get_error_names(self):
        """
        Get the error names.

        Returns
        -------
        list
            List of error names.

        """
        return list(self.err_ar_dict.keys())

    def get_error(self, err_name):
        """
        Get the error according to their names.

        Parameters
        ----------
        err_name : str
            The error name to be calculated.

        Returns
        -------
        float
            The error value.

        """
        if not (err_name in self.get_error_names()):
            raise AttributeError('error could only be selected from %s"!' % str(self.get_error_names()))
        return self.err_ar_dict[err_name]()

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
            if self.loop_dict[key] is not None:
                display_str += '%s: %d ' % (key, self.loop_dict[key])
            else:
                display_str += '%s ' % key
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

        self.Algorithm_check(algorithm0)

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

    def Algorithm_check(self, algorithm0):
        psfon_index = algorithm0.find("PSFon")
        critcheck_index = algorithm0.find("CRITcheck")
        if psfon_index < critcheck_index:
            raise ValueError('The first time partial coherence calculation is turned on should always be before the CRITcheck')
        return


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
