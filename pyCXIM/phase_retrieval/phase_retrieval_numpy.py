# -*- coding: utf-8 -*-
"""
The common phase retrieval functions for CPU based phase retrieval process.

Due to the speed limitations this package should be only used for 2D phase retieval process.
In case of questions, please contact us.

Created on Thu Mar 30 11:35:00 2023

@author: Ren Zhe, Xu Han
@email: renzhe@ihep.ac.cn, xuhan@ihep.ac.cn, or renzhetu001@gmail.com
"""

import numpy as np
import re
import sys
from scipy.ndimage import gaussian_filter
from scipy.ndimage import measurements
from scipy.special import gammaln
from skimage.morphology import convex_hull_image


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
        Then the all the pixels are considered unmasked.

    Attributes
    ----------
    intensity : ndarray
        Measured intensity
    ModulusFFT : ndarray
        Modulus calculated from the measured intensity
    MaskFFT : ndarray
        Mask for the bad pixels in the measured intensity
    loop_dict : dict
        records number of loops performed for different algorithms.
    img : ndarray
        the result image
    support : ndarray
        the result support
    """

    def __init__(self, measured_intensity, Seed, starting_img=None, support=None, MaskFFT=None):
        print('Seed %04d' % Seed)
        self.intensity = np.array(measured_intensity, dtype=float)
        self.ModulusFFT = np.fft.fftshift(np.sqrt(measured_intensity))

        # loop index records number of loops performed for different algorithms
        self.loop_dict = {}

        # generate or import the starting image for the phase retrieval process
        if starting_img is not None:
            print('Given starting image used.')
            self.img = np.array(starting_img, dtype=complex)
        else:
            print('Random starting image used.')
            np.random.seed(seed=Seed)
            self.img = np.fft.ifftn(np.multiply(self.ModulusFFT, np.exp(1j * np.random.rand(*self.ModulusFFT.shape) * 2 * np.pi)))

        # generate according to the autocorrelation or import the starting support
        if support is not None:
            self.support = np.array(support, dtype=float)
        else:
            self.support = np.zeros_like(self.ModulusFFT, dtype=float)
            Startautocorrelation = np.absolute(np.fft.fftshift(np.fft.fftn(self.intensity)))
            threshold = 4.0 / 1000.0 * (np.amax(Startautocorrelation) - np.amin(Startautocorrelation)) + np.amin(Startautocorrelation)
            self.support[Startautocorrelation >= threshold] = 1.0

        if MaskFFT is not None:
            self.MaskFFT = np.array(MaskFFT, dtype=float)
            self.MaskFFT = np.fft.fftshift(self.MaskFFT)
        else:
            self.MaskFFT = np.zeros_like(self.ModulusFFT, dtype=float)
        return

    def End(self):
        """
        End of the phase retrieval process.

        Returns
        -------
        None.

        """
        self.img = self.img * self.support
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

    def CenterSup(self):
        """
        Center the 2D retrieved image according to the support.

        Returns
        -------
        None.

        """
        support_shift = np.around(np.array(self.support.shape, dtype=float) / 2.0 - 0.5 - measurements.center_of_mass(self.support))
        for i, shift in enumerate(support_shift):
            self.support = np.roll(self.support, int(shift), axis=i)
            self.img = np.roll(self.img, int(shift), axis=i)
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
            self.Modulus_sum = np.zeros_like(self.ModulusFFT, dtype=float)

        # Update the support function according to the shrink wrap method
        self.Modulus_sum += np.abs(self.img)
        Bluredimg = gaussian_filter(self.Modulus_sum / float(self.loop_dict['Sup']) * hybrid_para + (1.0 - hybrid_para) * np.abs(self.img), sigma=Gaussiandelta)
        threshold = thrpara * (np.amax(Bluredimg) - np.amin(Bluredimg)) + np.amin(Bluredimg)
        self.support = np.zeros_like(self.img, dtype=float)
        self.support[Bluredimg >= threshold] = 1.0
        # Center the support to the center of the image
        if self.loop_dict['Sup'] % 10 == 0:
            self.CenterSup()
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
    #     self.support = np.array(convex_hull_image(self.support), dtype=float)
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
            self.holderimg_HIO = np.zeros_like(self.ModulusFFT, dtype=complex)

        para = 0.9  # parameter for the HIO

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
            self.std_noise = np.sqrt(np.sum(self.intensity * (1.0 - self.MaskFFT)) / np.sum(1.0 - self.MaskFFT))

        para = 0.9  # parameter for the HIO

        for i in range(num_NHIO_loop):
            self.holderimg_HIO = self.support * self.img + (1.0 - self.support) * (self.holderimg_HIO - self.img * para)
            zero_select_con = np.logical_and(self.support == 0, np.abs(self.holderimg_HIO) < 3.0 * self.std_noise)
            self.holderimg_HIO[zero_select_con] = 0
            self.img = self.ModulusProj(self.holderimg_HIO)
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
            self.holderimg_RAAR = np.zeros_like(self.ModulusFFT, dtype=complex)

        para0 = 0.75  # parameter for the RAAR

        for i in range(num_RAAR_loop):
            para = para0 + (1.0 - para0) * (1.0 - np.exp(-(i / 12.0) ** 3.0))
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
        AmpFFT = np.fft.fftn(point)
        AmpFFT = (1.0 - self.MaskFFT) * np.multiply(self.ModulusFFT, np.exp(1j * np.angle(AmpFFT))) + 0.98 * self.MaskFFT * AmpFFT
        point = np.fft.ifftn(AmpFFT)
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
        for i in range(num_DIF_loop):
            map1 = (1 + invpara) * self.ModulusProj(self.img) - invpara * self.img
            map2 = (1 - invpara) * self.SupProj(self.img) + invpara * self.img
            self.img = self.img + para_dif * (self.SupProj(map1) - self.ModulusProj(map2))
        self.img = self.img * self.support
        return

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
        return np.array(np.abs(self.img) * self.support, dtype=float)

    def get_img_Phase(self):
        """
        Get the phase value for the result image (real space).

        Should be used only after the phase retrieval calculation is finished.
        when the end function tranfers the result from GPU to memory.
        Phase unwrap should be performed afterwards

        Returns
        -------
        ndarray
            Phase of the result reconstruction (data with float type).

        """
        return np.array(np.angle(self.img) * self.support, dtype=float)

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
        return np.abs(np.fft.fftshift(np.fft.fftn(self.img * self.support)))

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
        return np.fft.fftshift(np.fft.fftn(self.img * self.support))

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
        return np.square(np.abs(np.fft.fftshift(np.fft.fftn(self.img * self.support))))

    def get_support_size(self):
        """
        Get the total size of the support.

        Returns
        -------
        int
            The total size of the support.

        """
        return np.sum(self.support)

    def get_Fourier_space_error(self):
        """
        Calculate the fourier space error of the result reconstruction.

        Should be used only after the phase retrieval calculation is finished,
        when the end function tranfers the result from GPU to memory.

        Returns
        -------
        error : float
            Fourier space error of the result reconstruction.

        """
        error = np.sum(np.square(np.abs(np.fft.fftn(self.img * self.support)) * (1.0 - self.MaskFFT) - self.ModulusFFT * (1.0 - self.MaskFFT))) / np.sum(np.square(self.ModulusFFT * (1.0 - self.MaskFFT)))
        return error

    def get_Poisson_Likelihood(self):
        """
        Calculate the loglikelihood of the result reconstruction considering poisson distribution.

        Should be used only after the phase retrieval calculation is finished,
        when the end function tranfers the result from GPU to memory.

        Returns
        -------
        loglikelihood : float
            The loglikeihood error of the result reconstruction.

        """
        loglikelihood = np.sum(np.fft.fftshift(1.0 - self.MaskFFT) * ((self.get_intensity() + np.finfo(float).eps) + gammaln(self.intensity + np.finfo(float).eps) - (self.intensity + np.finfo(float).eps) * np.log(self.get_intensity() + np.finfo(float).eps))) / np.sum(1.0 - self.MaskFFT)
        return loglikelihood

    def get_Free_LogLikelihood(self, LLKmask=None):
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
        if LLKmask is not None:
            loglikelihood = np.sum(LLKmask * ((self.get_intensity() + np.finfo(float).eps) + gammaln(self.intensity + np.finfo(float).eps) - (self.intensity + np.finfo(float).eps) * np.log(self.get_intensity() + np.finfo(float).eps))) / np.sum(LLKmask)
        else:
            loglikelihood = 0
        return loglikelihood

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
        pattern0 = r'(\w+\*\*\d+)|(\(.+?\)\*\*\d+)|(\w+)'
        # Pattern 1 matches pattern like RAAR**50
        pattern1 = r'(\w+)\*\*(\d+)'
        # Pattern 2 matches pattern like Sup
        pattern2 = r'(\w+)'
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


def Free_LLK_FFTmask(MaskFFT, percentage=0.04, r=3):
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
        The percentage of pixels that should be used for the Free LLk mask. The default is 0.04.
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
        npoints = int(percentage * yd * xd / r / r)
        ycen_ar = np.random.randint(r, yd - r, size=npoints)
        xcen_ar = np.random.randint(r, xd - r, size=npoints)
        for i in range(npoints):
            ycen = ycen_ar[i]
            xcen = xcen_ar[i]
            if not ((0.95 * yd / 2.0 < ycen < 1.05 * yd / 2) and (0.95 * xd / 2.0 < xcen < 1.05 * xd / 2)):
                LLKmask[(ycen - r):(ycen + r + 1), (xcen - r):(xcen + r + 1)] = 1.0
    elif len(MaskFFT.shape) == 3:
        zd, yd, xd = MaskFFT.shape
        npoints = int(percentage * zd * yd * xd / r / r / r)
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
