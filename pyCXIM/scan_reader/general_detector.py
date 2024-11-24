# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:38:10 2024

@author: Lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.ndimage import center_of_mass
from scipy.ndimage import median_filter


class DetectorMixin(object):
    """
    A Mixin class that adds detector methods to the scan classes.

    This Mixin provides methods for the two dimensional detectors, which can be used
    to generate detector masks, loop through detector images, and load the two-dimensional detector data.

    Methods:
        get_detector_size() : Get the size of detector.
        get_detector_pixelsize() : Get the pixel size of detector.
        load_mask(pathmask) : Load the mask files defining the bad pixels on the detector.
    """

    def get_detector_size(self):
        """
        Get the size of detector.

        Returns
        -------
        tuple
            The pixel number of defined detector in Y, X order.

        """
        return self.detector_size

    def get_detector_pixelsize(self):
        """
        Get the pixel size of detector.

        Returns
        -------
        float
            The pixel size of defined detector.

        """
        return self.pixel_size

    def load_mask(self, pathmask=''):
        """
        Load the mask files defining the bad pixels on the detector.

        The mask file is padded with 0 and 1.
        0 means that the pixel is not masked and 1 means that the pixel is masked.

        Parameters
        ----------
        pathmask : str, optional
            The path for the mask file. The default is None.
        threshold : float, optional
            The threshold value defining the hot pixels on the detector. The default is ''.

        Returns
        -------
        ndarray
            The generated mask file.

        """
        if os.path.exists(pathmask):
            print('Predefined mask loaded')
            self.mask = np.load(pathmask)
            if self.mask.shape != self.detector_size:
                raise ValueError('The mask size does not match with the detector size, please check it again!')
        else:
            print('Could not find the predefined mask.')
            self.mask = np.zeros((self.detector_size[0], self.detector_size[1]))

        self.img_correction = 1.0 - self.mask
        return self.mask

    def get_mask(self):
        """
        Get the mask file of the detector.

        Returns
        -------
        ndarray
            The generated mask file.

        """
        return self.mask

    def get_image_correction(self):
        """
        Get the image correction file used for the mask correction.

        Returns
        -------
        ndarray
            The image used for the mask correction.

        """
        return self.img_correction

    def get_mask_for_plot(self):
        """
        Get the mask for the plot.

        The zeros in the mask are masked, so that it would not shadow the real image in the plotting process.

        Returns
        -------
        ndarray
            The mask image where zeros are masked.

        """
        return np.ma.masked_where(self.mask == 0, self.mask)

    def add_mask_circle(self, cen, r0):
        """
        Generate a circular mask for the normal beam stop.

        Parameters
        ----------
        cen : Union[ndarray, list, tuple]
            The center of the circle in Y, X order.
        r0 : float
            The radius of the circle.

        Returns
        -------
        ndarray
            The updated version of the maks file with a circular area masked.

        """
        temp = np.linalg.norm(np.indices(self.mask.shape) - np.array(cen)[:, np.newaxis, np.newaxis], axis=0)
        self.mask[temp < r0] = 1
        self.img_correction = 1.0 - self.mask
        return self.mask

    def add_mask_inverse_circle(self, cen, r0):
        """
        Generate a circular hole in the mask for the flight path tube.

        Parameters
        ----------
        cen : Union[ndarray, list, tuple]
            The center of the circle in Y, X order.
        r0 : float
            The radius of the circle.

        Returns
        -------
        ndarray
            The updated version of the mask file leaving the central circular area free.

        """
        temp = np.linalg.norm(np.indices(self.mask.shape) - np.array(cen)[:, np.newaxis, np.newaxis], axis=0)
        self.mask[temp > r0] = 1
        self.img_correction = 1.0 - self.mask
        return self.mask

    def add_mask_rectangle(self, pos):
        """
        Generate a rectangular mask for the 2D detector.

        Parameters
        ----------
        pos : Union[ndarray, list, tuple]
            The position of the rectangular mask on the detector in Ymin, Ymax, Xmin, Xmax order.

        Returns
        -------
        ndarray
            The updated version of the maks file with a rectangular area masked.

        """
        self.mask[pos[0]:pos[1], pos[2]:pos[3]] = 1
        self.img_correction = 1.0 - self.mask
        return self.mask

    def add_semi_transparent_mask(self, abs_pos, trans, margin):
        """
        Generate the mask file for the semitransparent beamstop and the corresponding correction image.

        By multiplying the img_correction with the original image, the masked pixels are set to zeros and the absorption fromt the Si wafers are corrected.

        Parameters
        ----------
        abs_pos : Union[ndarray, list, tuple]
            The position for the Si wafer absorber in Ymin, Ymax, Xmin, Xmax order.
        trans : float
            The transmission of the Si wafer.
        margin : int
            The width of the wafer broader to be masked.

        Returns
        -------
        ndarray
            The mask file for the bad pixels.
        ndarray
            The image correction file to be multiplied with the original image.

        """
        self.img_correction[abs_pos[0]:(abs_pos[1] + 1), abs_pos[2]:(abs_pos[3] + 1)] = self.img_correction[abs_pos[0]:(abs_pos[1] + 1), abs_pos[2]:(abs_pos[3] + 1)] / trans
        self.temp = np.zeros_like(self.mask)
        self.temp[(abs_pos[0] - margin):(abs_pos[1] + margin + 1), (abs_pos[2] - margin):(abs_pos[3] + margin + 1)] = 1
        self.temp[abs_pos[0]:(abs_pos[1] + 1), abs_pos[2]:(abs_pos[3] + 1)] = 0
        self.mask[self.temp == 1] = 1
        self.img_correction[self.mask == 1] = 0
        return self.mask, self.img_correction

    def image_mask_correction(self, image, correction_mode='constant'):
        """
        Correction of the intensities of the masked pixels.

        Parameters
        ----------
        image : ndarray
            The original images to be corrected.
        correction_mode : str, optional
            If correction_mode is 'constant',intensity of the masked pixels will be corrected according to the img_correction array generated before.
            Most of the time, intensity of the masked pixels will be set to zero.
            However, for the semitransparent mask the intensity will be corrected according to the transmission.
            If the correction_mode is 'medianfilter', the intensity of the masked pixels will be set to the median filter value according the surrounding pixels.
            If the correction_mode is 'off', the intensity of the masked pixels will not be corrected.
            The default is 'constant'.

        Returns
        -------
        image : ndarray
            The corrected intensity.

        """
        if correction_mode == 'constant':
            image = image * self.img_correction
        elif correction_mode == 'medianfilter':
            image_filtered = median_filter(image, size=3)
            image[self.mask == 1] = image_filtered[self.mask == 1]
        elif correction_mode == 'off':
            pass
        else:
            raise KeyError('The image correction mode could not be recognized!')
        return image

    def image_roi_check(self, roi):
        """
        Check the roi size for the detector.

        Parameters
        ----------
        roi : list
            Region of interest in [Ymin, Ymax, Xmin, Xmax] form.

        Returns
        -------
        roi : list
            The new region of interest which fits on the detector.

        """
        assert len(roi) == 4, 'The region of interest must be four integer numbers!'
        if roi is None:
            roi = [0, 0, 0, 0]
        if roi[1] > self.detector_size[0]:
            roi[1] = self.detector_size[0]
        if roi[3] > self.detector_size[1]:
            roi[3] = self.detector_size[1]
        if roi[0] > roi[1]:
            roi[0] = roi[1]
        if roi[2] > roi[3]:
            roi[2] = roi[3]
        return roi

    def image_cut_check(self, cen, width):
        """
        Check the maximum width that can be cutted around the peak position on the detector.

        Parameters
        ----------
        cen : list
            The center of the diffraction peak on the detector.
        width : list
            The symmetric width to be cutted in the [Y, X] or [Ymin, Ymax, Xmin, Xmax] form.

        Returns
        -------
        width : list
            The maximum allowed symmetric width to be cutted in the [Y, X] form.

        """
        if len(width) == 2:
            width[0] = int(np.amin([width[0], cen[0] * 0.95, 0.95 * (self.detector_size[0] - cen[0])]))
            width[1] = int(np.amin([width[1], cen[1] * 0.95, 0.95 * (self.detector_size[1] - cen[1])]))
            print('box size is %d * %d' % (width[0], width[1]))
        elif len(width) == 4:
            width[0] = int(np.amin([width[0], cen[0] * 0.95]))
            width[1] = int(np.amin([width[1], 0.95 * (self.detector_size[0] - cen[0])]))
            width[2] = int(np.amin([width[2], cen[1] * 0.95]))
            width[3] = int(np.amin([width[3], 0.95 * (self.detector_size[1] - cen[1])]))
            print('box size is %d, %d, %d, %d' % (width[0], width[1], width[2], width[3]))
        return width

    def load_rois(self, roi=None, show_cen_image=False, normalize_signal=None, correction_mode='constant'):
        """
        Load the images within the region of interest.

        Parameters
        ----------
        roi : list, optional
            The region of interest in [Ymin, Ymax, Xmin, Xmax] order.
            If not given, the complete detector image will be loaded. The default is None.
        show_cen_image : bool, optional
            If true the central image of the data will be shown to help select the roi. The default is False.
        normalize_signal : str, optional
            The name of the signal used to normalize the diffraction intensity. The default is None.

        Returns
        -------
        dataset : ndarray
            The 3D diffraction intensity in the region of interest.
        mask_3D : ndarray
            The corresponding mask.
        pch : list
            The position of the highest integrated diffraction intensity.
            pch[0] corresponds to the frame with the maximum intensity in the rocking curve
            pch[1] corresponds to the peak position Y on the detector
            pch[2] corresponds to the peak position X on the detector
        roi : list
            The new roi.

        """
        if not hasattr(self, 'load_single_image'):
            'Reading method for the detector type not defined, please check the code or contact the author renzhe@ihep.ac.cn!'
        print('Loading data....')
        if roi is None:
            roi = [0, self.detector_size[0], 0, self.detector_size[1]]
        else:
            roi = self.image_roi_check(roi)

        if show_cen_image:
            plt.imshow(np.log10(self.load_single_image(self.npoints / 2, correction_mode=correction_mode) + 1.0), cmap='jet')
            plt.show()

        dataset = np.zeros((self.npoints, roi[1] - roi[0], roi[3] - roi[2]))
        pch = np.zeros(3, dtype=int)

        if type(normalize_signal) == str:
            assert (normalize_signal in self.get_counter_names()), "The given signal for the normalization does not exist in the scan!"
            normal_int = self.get_scan_data(normalize_signal)
            normal_int = normal_int / np.average(normal_int)
        else:
            normal_int = np.ones(self.npoints)

        for i in range(self.npoints):
            image = self.load_single_image(i, correction_mode=correction_mode)
            dataset[i, :, :] = image[roi[0]:roi[1], roi[2]:roi[3]] / normal_int[i]
            sys.stdout.write('\rprogress:%d%%' % ((i + 1) * 100.0 / self.npoints))
            sys.stdout.flush()
        print()

        roi_int = np.sum(dataset, axis=(1, 2))
        self.add_scan_data('%s_roi1' % self.detector, roi_int)
        self.add_motor_pos('%s_roi1' % self.detector, list(roi))

        pch = np.array([np.argmax(np.sum(dataset, axis=(1, 2))), np.argmax(np.sum(dataset, axis=(0, 2))), np.argmax(np.sum(dataset, axis=(0, 1)))], dtype=int) + np.array([0, roi[0], roi[2]])
        print("maximum intensity of the scan find at %s" % str(pch))

        mask_3D = np.repeat(self.mask[np.newaxis, roi[0]:roi[1], roi[2]:roi[3]], self.npoints, axis=0)
        return dataset, mask_3D, pch, roi

    def load_images(self, roi=None, width=None, show_cen_image=False, normalize_signal=None, correction_mode='constant'):
        """
        Load the image images in the scan.

        The images will be loaded around the maximum diffraction intensity in the region of interest.

        Parameters
        ----------
        roi : list, optional
            The region of interest in [Ymin, Ymax, Xmin, Xmax] order. The default is None.
        width : list, optional
            The half width for cutting around the highest intensity. The default is None.
        show_cen_image : bool, optional
            If true, the center image in the scan will be plotted for the selection of the roi. The default is False.
        normalize_signal : str, optional
            The name of the signal used to normalize the diffraction intensity. The default is None.

        Returns
        -------
        dataset : ndarray
            Diffraction intensity in 3D.
        mask_3D : ndarray
            The corresponding mask calculated.
        pch : list
            The position of the highest integrated diffraction intensity.
            pch[0] corresponds to the frame with the maximum intensity in the rocking curve
            pch[1] corresponds to the peak position Y on the detector
            pch[2] corresponds to the peak position X on the detector
        width : list
            The half width of the final cut.

        """
        if not hasattr(self, 'load_single_image'):
            'Reading method for the detector type not defined, please check the code or contact the author renzhe@ihep.ac.cn!'
        print('Loading data....')
        dataset = np.zeros((self.npoints, self.detector_size[0], self.detector_size[1]))
        if roi is None:
            roi = [0, self.detector_size[0], 0, self.detector_size[1]]
        else:
            roi = self.image_roi_check(roi)

        if type(normalize_signal) == str:
            assert (normalize_signal in self.get_counter_names()), "The given signal for the normalization does not exist in the scan!"
            normal_int = self.get_scan_data(normalize_signal)
            normal_int = normal_int / np.average(normal_int)
        else:
            normal_int = np.ones(self.npoints)

        pch = np.zeros(3, dtype=int)
        for i in range(self.npoints):
            image = self.load_single_image(i, correction_mode=correction_mode)
            dataset[i, :, :] = image / normal_int[i]
            sys.stdout.write('\rprogress:%d%%' % ((i + 1) * 100.0 / self.npoints))
            sys.stdout.flush()
        print()
        if show_cen_image:
            plt.imshow(np.log10(self.load_single_image(self.npoints / 2, correction_mode=correction_mode) + 1.0), cmap='jet')
            plt.show()

        roi_int = np.sum(dataset[:, roi[0]:roi[1], roi[2]:roi[3]], axis=(1, 2))
        self.add_scan_data('%s_roi1' % self.detector, roi_int)
        self.add_motor_pos('%s_roi1' % self.detector, roi)

        pch = np.array([np.argmax(roi_int), np.argmax(np.sum(dataset[:, roi[0]:roi[1], roi[2]:roi[3]], axis=(0, 2))), np.argmax(np.sum(dataset[:, roi[0]:roi[1], roi[2]:roi[3]], axis=(0, 1)))], dtype=int) + np.array([0, roi[0], roi[2]])
        print("maximum intensity of the scan find at " + str(pch))

        if width is None:
            width = [400, 400]
        if len(width) == 2:
            width = self.image_cut_check(pch[1:], width)
            dataset = dataset[:, (pch[1] - width[0]):(pch[1] + width[0]), (pch[2] - width[1]):(pch[2] + width[1])]
            mask_3D = np.repeat(self.mask[np.newaxis, (pch[1] - width[0]):(pch[1] + width[0]), (pch[2] - width[1]):(pch[2] + width[1])], self.npoints, axis=0)
        elif len(width) == 4:
            width = self.image_cut_check(pch[1:], width)
            dataset = dataset[:, (pch[1] - width[0]):(pch[1] + width[1]), (pch[2] - width[2]):(pch[2] + width[3])]
            mask_3D = np.repeat(self.mask[np.newaxis, (pch[1] - width[0]):(pch[1] + width[1]), (pch[2] - width[2]):(pch[2] + width[3])], self.npoints, axis=0)

        return dataset, mask_3D, pch, width

    def image_roi_sum(self, rois, roi_order='YX', save_img_sum=True):
        """
        Calculate the integrated intensity in different region of interests.

        Parameters
        ----------
        rois : ndarray
            Region of interest to be integrated. Rois should be given in the form of [roi1, roi2, roi3].
            Each roi is described in [Ymin, Ymax, Xmin, Xmax] order.
        roi_oder : str, optional
            If roi_order is 'XY', the roi is described in [Xmin, Xmax, Ymin, Ymax] order
            If roi_order is 'YX', the roi is described in [Ymin, Ymax, Xmin, Xmax] order
            The default is 'YX'.
        save_img_sum : bool, optional
            If true, the integrated diffraction pattern of the entire scan will be saved. The default is True.

        Returns
        -------
        None.

        """
        if not hasattr(self, 'load_single_image'):
            'Reading method for the detector type not defined, please check the code or contact the author renzhe@ihep.ac.cn!'
        rois = np.array(rois, dtype=int)
        if rois.ndim == 1:
            if rois.shape == (0,):
                num_of_rois = 0
            elif rois.shape == (4,):
                rois = rois[np.newaxis, :]
                num_of_rois = 1
            else:
                raise ValueError('The roi must contain four integers!')
        elif rois.ndim == 2:
            assert rois.shape[1] == 4, 'The roi must contain four integers!'
            num_of_rois = rois.shape[0]
        else:
            raise ValueError('The roi given does not have the correct dimensions!')

        for i in range(num_of_rois):
            roi = rois[i, :]
            if roi_order == 'XY':
                roi[[0, 1, 2, 3]] = roi[[2, 3, 0, 1]]
            rois[i, :] = self.image_roi_check(roi)

        img_sum = np.zeros((self.detector_size[0], self.detector_size[1]))
        rois_int = np.zeros((self.npoints, num_of_rois + 1))

        for i in range(self.npoints):
            image = self.load_single_image(i, correction_mode='constant')
            rois_int[i, 0] = np.sum(image[:, :])
            for j in range(num_of_rois):
                roi = rois[j, :]
                rois_int[i, j + 1] = np.sum(image[roi[0]:roi[1], roi[2]:roi[3]])
            img_sum += image
            sys.stdout.write('\rprogress:%d%%' % ((i + 1) * 100.0 / self.npoints))
            sys.stdout.flush()

        print('')
        self.add_scan_data('%s_full' % self.detector, rois_int[:, 0])
        for j in range(num_of_rois):
            self.add_motor_pos('%s_roi%d' % (self.detector, (j + 1)), list(rois[j, :]))
            self.add_scan_data('%s_roi%d' % (self.detector, (j + 1)), rois_int[:, j + 1])

        if self.pathsave != '' and save_img_sum:
            np.save(self.path_imgsum, img_sum)
        return

    def image_sum(self, start_frame=None, end_frame=None, save_img_sum=True, correction_mode='constant'):
        """
        Integrate the diffraction patterns in the scan.

        Parameters
        ----------
        start_frame : int, optional
            The start frame number for the integration. If not given, the first image will be used. The default is None.
        end_frame : int, optional
            The end frame number for the integration. If not given, all the images within the scan will be used. The default is None.
        save_img_sum : bool, optional
            If true the integrate diffraction pattern will be saved. The default is True.
        correction_mode : str, optional
            The correction correction_mode for the masked detector pixels, can be selected between 'off', 'constant', and 'medianfilter'. The default is 'constant'.

        Returns
        -------
        img_sum : darray
            The integrated diffraction intensity.

        """
        img_sum = np.zeros(self.detector_size)
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.npoints
        assert end_frame > start_frame, 'The end frame should always be larger than the start frame!'

        for i in range(start_frame, end_frame):
            image = self.load_single_image(i, correction_mode=correction_mode)
            img_sum += image
            sys.stdout.write('\rprogress:%d%%' % ((i + 1) * 100.0 / self.npoints))
            sys.stdout.flush()

        print('')
        if (self.pathsave != '') and save_img_sum:
            np.save(self.path_imgsum, img_sum)
            self.add_scan_infor('path_imgsum')
        return img_sum

    def image_peak_pos_per_frame(self, cut_width=[30, 30], save_img_sum=False):
        """
        Find the peak position in each frame of the scan.

        Used in the calibration scan with the direct beam or to find the position of the truncation rod.

        Parameters
        ----------
        cut_width : list, optional
            The roi discribing the width of the peak. The default is [20, 20].
        save_img_sum : bool, optional
            If ture, the sum of the cutted peak will be saved. The default is False.

        Returns
        -------
        X_pos : ndarray
            The X position of the peak on the detector in each frame.
        Y_pos : ndarray
            The Y position of the peak on the detector in each frame.
        int_ar : ndarray
            The summed intensity of the peak.

        """
        if not hasattr(self, 'load_single_image'):
            'Reading method for the detector type not defined, please check the code or contact the author renzhe@ihep.ac.cn!'
        print('Loading data....')
        X_pos = np.zeros(self.npoints)
        Y_pos = np.zeros(self.npoints)
        int_ar = np.zeros(self.npoints)
        img_sum = np.zeros((2 * cut_width[0], 2 * cut_width[1]))

        for i in range(self.npoints):
            image = self.load_single_image(i, correction_mode='constant')
            Y_pos[i], X_pos[i] = np.unravel_index(np.argmax(image), image.shape)
            Y_pos[i] = np.clip(Y_pos[i], cut_width[0], self.detector_size[0] - cut_width[0])
            X_pos[i] = np.clip(X_pos[i], cut_width[1], self.detector_size[1] - cut_width[1])
            Y_shift, X_shift = center_of_mass(image[int(Y_pos[i] - cut_width[0]):int(Y_pos[i] + cut_width[0]), int(X_pos[i] - cut_width[1]):int(X_pos[i] + cut_width[1])])
            Y_pos[i] = Y_pos[i] + Y_shift - cut_width[0]
            X_pos[i] = X_pos[i] + X_shift - cut_width[1]
            if save_img_sum:
                img_sum += image[int(np.around(Y_pos[i] - cut_width[0])):int(np.around(Y_pos[i] + cut_width[0])), int(np.around(X_pos[i] - cut_width[1])):int(np.around(X_pos[i] + cut_width[1]))]
            int_ar[i] = np.sum(image[int(np.around(Y_pos[i] - cut_width[0])):int(np.around(Y_pos[i] + cut_width[0])), int(np.around(X_pos[i] - cut_width[1])):int(np.around(X_pos[i] + cut_width[1]))])
            sys.stdout.write('\rprogress:%d%%' % ((i + 1) * 100.0 / self.npoints))
            sys.stdout.flush()

        print()
        if self.pathsave != '' and save_img_sum:
            np.save(self.path_imgsum, img_sum)
        return X_pos, Y_pos, int_ar

    def image_find_peak_position(self, roi=None, cut_width=None, normalize_signal=None):
        """
        Find the peak position in the scan.

        Parameters
        ----------
        roi : list, optional
            The region of interest. The default is None.
        cut_width : list, optional
            The cut width in Y, X direction. The default is None.
        normalize_signal : string, optional
            The name of the signal used to normalize the diffraction intensity. The default is None.

        Returns
        -------
        pch : list
            The position of the highest integrated diffraction intensity.
            pch[0] corresponds to the frame with the maximum intensity in the rocking curve
            pch[1] corresponds to the peak position Y on the detector
            pch[2] corresponds to the peak position X on the detector

        """
        if not hasattr(self, 'load_single_image'):
            'Reading method for the detector type not defined, please check the code or contact the author renzhe@ihep.ac.cn!'
        print("Finding the frames with the highest intensity....")
        if roi is None:
            roi = [0, self.detector_size[0], 0, self.detector_size[1]]
        else:
            roi = self.image_roi_check(roi)

        roi_int = np.zeros(self.npoints)
        pch = np.zeros(3)

        if type(normalize_signal) == str:
            assert (normalize_signal in self.get_counter_names()), "The given signal for the normalization does not exist in the scan!"
            normal_int = self.get_scan_data(normalize_signal)
            normal_int = normal_int / np.average(normal_int)
        else:
            normal_int = np.ones(self.npoints)

        for i in range(self.npoints):
            image = self.load_single_image(i, correction_mode='constant')
            roi_int[i] = np.sum(image[roi[0]:roi[1], roi[2]:roi[3]] / normal_int[i])
            sys.stdout.write('\rprogress:%d%%' % ((i + 1) * 100.0 / self.npoints))
            sys.stdout.flush()

        pch[0] = int(np.argmax(roi_int))
        image = self.load_single_image(pch[0], correction_mode='constant')
        pch[-2:] = center_of_mass(image[roi[0]:roi[1], roi[2]:roi[3]]) + np.array([roi[0], roi[2]])
        if cut_width is not None:
            pch[-2:] = center_of_mass(image[int(pch[1] - cut_width[0]):int(pch[1] + cut_width[0]), int(pch[2] - cut_width[1]):int(pch[2] + cut_width[1])]) + np.array([int(pch[1] - cut_width[0]), int(pch[2] - cut_width[1])])
        print("")
        print("peak position on the detector (Z, Y, X): " + str(np.around(pch, 2)))
        return pch
