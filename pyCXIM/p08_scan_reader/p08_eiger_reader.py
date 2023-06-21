# -*- coding: utf-8 -*-
"""
Read and treat the p08 scans with eiger detectors.
Created on Thu Apr 27 15:33:21 2023

@author: renzhe
"""

import os
import sys
import numpy as np
import hdf5plugin
import h5py
from scipy.ndimage import measurements
import matplotlib.pyplot as plt
from .p08_scan_reader import P08Scan


class P08EigerScan(P08Scan):
    """
    Read and treat the scans with eiger detector. It is a child class of p08_scan.

    Parameters
    ----------
    path : str
        The path for the raw file folder.
    p08_file : str
        The name of the sample defined by the p08_newfile name in the system.
    scan : int
        The scan number.
    detector : str, optional
        The name of the detecter, it can be either 'eiger1m' or 'e500'. The default is 'eiger1m'.
    pathsave : str, optional
        The folder to save the results, if not given no results will be saved. The default is ''.
    pathmask : str, optional
        The path of the detector mask. If not given, mask will be generated according to the hot pixels in the first image of the scan. The default is ''.
    creat_save_folder : boolen, optional
        Whether the save folder should be created. The default is True.

    Returns
    -------
    None.

    """

    def __init__(self, path, p08_file, scan, detector='eiger1m', pathsave='', pathmask='', creat_save_folder=True):
        super().__init__(path, p08_file, scan, pathsave, creat_save_folder)
        self.detector = detector
        self.path_eiger_folder = os.path.join(self.path, self.detector)
        self.path_eiger_img = os.path.join(self.path_eiger_folder, "%s_%05d_data_%06d.h5")
        self.path_eiger_imgsum = os.path.join(self.pathsave, '%s_scan%05d_%s_imgsum.npy' % (self.p08_file, self.scan, self.detector))
        assert os.path.exists(self.path_eiger_folder), 'The image folder for %s images %s does not exist, please check the path again!' % (self.detector, self.path_eiger_folder)
        if self.detector == 'eiger1m':
            self.detector_size = (1062, 1028)

        self.eiger_load_mask(pathmask)

    def eiger_load_mask(self, pathmask="", threshold=1.0e7):
        """
        Load the mask files defining the bad pixels on the detector.

        The mask file is padded with 0 and 1.
        0 means that the pixel is not masked and 1 means that the pixel is masked.
        If the mask file does not exist, the image will be mask according to the hot pixels in the first image of the scan.

        Parameters
        ----------
        pathmask : string, optional
            The path for the mask file. The default is "".
        threshold : float, optional
            The threshold value defining the hot pixels on the detector. The default is "".

        Returns
        -------
        ndarray
            The generated mask file.

        """
        if os.path.exists(pathmask):
            print('Predefined mask loaded')
            self.mask = np.load(pathmask)
        else:
            print('Could not find the predefined mask, Generate the mask according to the first image in the scan')
            self.mask = np.zeros((self.detector_size[0], self.detector_size[1]))
        pathimg = (self.path_eiger_img) % (self.p08_file, self.scan, 1)
        f = h5py.File(pathimg, "r")
        dataset = f['entry/data/data']
        image = np.array(dataset[0, :, :], dtype=float)
        self.mask[image > threshold] = 1
        self.img_correction = 1.0 - self.mask
        return self.mask

    def eiger_mask_circle(self, cen, r0):
        """
        Generate a circular mask for the normal beam stop.

        Parameters
        ----------
        cen : Union[ndarray, list, turple]
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

    def eiger_mask_rectangle(self, pos):
        """
        Generate a rectangular mask for the 2D detector.

        Parameters
        ----------
        pos : Union[ndarray, list, turple]
            The position of the rectangular mask on the detector in Ymin, Ymax, Xmin, Xmax order.

        Returns
        -------
        ndarray
            The updated version of the maks file with a rectangular area masked.

        """
        self.mask[pos[0]:pos[1], pos[2]:pos[3]] = 1
        self.img_correction = 1.0 - self.mask
        return self.mask

    def eiger_semi_transparent_mask(self, cen, r0, large_abs_pos, trans1, small_abs_pos, trans2, margin):
        """
        Generate the mask file for the semitransparent beamstop and the corresponding correction image.

        By multiplying the img_correction with the original image, the masked pixels are set to zeros and the absorption fromt the Si wafers are corrected.

        Parameters
        ----------
        cen : Union[ndarray, list, turple]
            The center of the circle in Y, X order.
        r0 : float
            The radius of the circle.
        large_abs_pos : Union[ndarray, list, turple]
            The position for the large Si wafer in Ymin, Ymax, Xmin, Xmax order.
        trans1 : float
            The transmission of the large Si wafer.
        small_abs_pos : Union[ndarray, list, turple]
            The position for the large Si wafer in Ymin, Ymax, Xmin, Xmax order.
        trans2 : float
            The transmission due to the additional thickness from the small Si wafer.
        margin : int
            The width of the wafer broader to be masked.

        Returns
        -------
        ndarray
            The mask file for the bad pixels.
        ndarray
            The image correction file to be multiplied with the original image.

        """
        self.eiger_mask_circle(cen, r0)
        self.img_correction = np.ones_like(self.mask)
        self.img_correction[self.mask == 1] = 0
        self.img_correction[large_abs_pos[0]:(large_abs_pos[1] + 1), large_abs_pos[2]:(large_abs_pos[3] + 1)] = self.img_correction[large_abs_pos[0]:(large_abs_pos[1] + 1), large_abs_pos[2]:(large_abs_pos[3] + 1)] / trans1
        self.img_correction[small_abs_pos[0]:(small_abs_pos[1] + 1), small_abs_pos[2]:(small_abs_pos[3] + 1)] = self.img_correction[small_abs_pos[0]:(small_abs_pos[1] + 1), small_abs_pos[2]:(small_abs_pos[3] + 1)] / trans2
        self.temp = np.zeros_like(self.mask)
        self.temp[(large_abs_pos[0] - margin):(large_abs_pos[1] + margin + 1), (large_abs_pos[2] - margin):(large_abs_pos[3] + margin + 1)] = 1
        self.temp[large_abs_pos[0]:(large_abs_pos[1] + 1), large_abs_pos[2]:(large_abs_pos[3] + 1)] = 0
        self.mask[self.temp == 1] = 1
        self.temp[(small_abs_pos[0] - margin):(small_abs_pos[1] + margin + 1), (small_abs_pos[2] - margin):(small_abs_pos[3] + margin + 1)] = 1
        self.temp[small_abs_pos[0]:(small_abs_pos[1] + 1), small_abs_pos[2]:(small_abs_pos[3] + 1)] = 0
        self.mask[self.temp == 1] = 1
        return self.mask, self.img_correction

    def eiger_mask_correction(self, image):
        """
        Correction of the images according to the mask generated.

        The value of the masked arrays are set to zero. The absorption of the semitransparent masks are corrected.

        Parameters
        ----------
        image : ndarray
            The original images to be corrected.

        Returns
        -------
        image : ndarray
            The corrected intensity.

        """
        image = image * self.img_correction
        return image

    def get_eiger_imgsum(self):
        """
        Get the eiger image sum.

        Returns
        -------
        ndarray
            The sum of the eiger 4M detector images.

        """
        assert os.path.exists(self.path_eiger_imgsum), print('Could not find the summarized eiger detector images!')
        return np.load(self.path_eiger_imgsum)

    def eiger_load_single_image(self, img_index, mask_correction=True):
        """
        Load a single eiger4M image in the scan.

        Parameters
        ----------
        img_index : int
            The index of the image in the scan.
        path_mask : str, optional
            The path for the mask files. The default is ''.
        mask_correction : bool, optional
            If true, the masked pixels will be set to . The default is True.

        Returns
        -------
        image : ndarray
            The result image in the scan.

        """
        assert img_index < self.npoints, 'The image number wanted is larger than the total image number in the scan!'
        img_index = int(img_index)

        pathimg = (self.path_eiger_img) % (self.p08_file, self.scan, img_index // 2000 + 1)
        f = h5py.File(pathimg, "r")
        dataset = f['entry/data/data']
        image = np.array(dataset[img_index % 2000, :, :], dtype=float)

        if mask_correction:
            image = self.eiger_mask_correction(image)
        return image

    def eiger_roi_check(self, roi):
        """
        Check the roi size for the eiger detector.

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

    def eiger_cut_check(self, cen, width):
        """
        Cut the maximum symmetric width that can be cutted around the peak position on the detector.

        Parameters
        ----------
        cen : list
            The center of the diffraction peak on the detector.
        width : list
            The symmetric width to be cutted in the [Y, X] form.

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

    def eiger_roi_converter(self, roi):
        """
        Change the region of interest order between [Xmin, Xmax, Ymin, Ymax] and [Ymin, Ymax, Xmin, Xmax].

        Parameters
        ----------
        roi : ndarray
            The region of interest.

        Returns
        -------
        roi : ndarray
            The converted region of interest.

        """
        roi = np.array(roi, dtype=int)
        roi[[0, 1, 2, 3]] = roi[[2, 3, 0, 1]]
        return roi

    def eiger_load_rois(self, roi=None, roi_order='YX', show_cen_image=False):
        """
        Load the images with certain region of interest.

        Parameters
        ----------
        roi : list, optional
            The region of interest. The default is None.
        roi_oder : str, optional
            If roi_order is 'XY', the roi is described in [Xmin, Xmax, Ymin, Ymax] order
            If roi_order is 'YX', the roi is described in [Ymin, Ymax, Xmin, Xmax] order
            The default order is 'YX'.
        show_cen_image : bool, optional
            If true the central image of the data will be shown to help select the rois. The default is False.

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
        print('Loading data....')
        if roi is None:
            roi = [0, self.detector_size[0], 0, self.detector_size[1]]
        else:
            if roi_order == 'XY':
                roi = self.eiger_roi_converter(roi)
            roi = self.eiger_roi_check(roi)

        if show_cen_image:
            plt.imshow(np.log10(self.eiger_load_single_image(int(self.npoints / 2)) + 1.0), cmap='jet')
            plt.show()

        dataset = np.zeros((self.npoints, roi[1] - roi[0], roi[3] - roi[2]))
        pch = np.zeros(3, dtype=int)
        ion_bl_int = self.get_scan_data('ion_bl')
        ion_bl_int = ion_bl_int / np.round(np.average(ion_bl_int))

        for i in range(self.npoints):
            pathimg = self.path_eiger_img % (self.p08_file, self.scan, i // 2000 + 1)
            f = h5py.File(pathimg, "r")
            image = np.array(f['entry/data/data'][i % 2000, :, :], dtype=float)
            image = self.eiger_mask_correction(image)
            dataset[i, :, :] = image[roi[0]:roi[1], roi[2]:roi[3]] / ion_bl_int[i]
            sys.stdout.write('\rprogress:%d%%' % ((i + 1) * 100.0 / self.npoints))
            sys.stdout.flush()
        print()

        roi_int = np.sum(dataset, axis=(1, 2))
        self.add_scan_data('%s_roi1' % self.detector, roi_int)
        self.add_motor_pos('%s_roi1' % self.detector, list(roi))

        pch = np.array([np.argmax(np.sum(dataset, axis=(1, 2))), np.argmax(np.sum(dataset, axis=(0, 2))), np.argmax(np.sum(dataset, axis=(0, 1)))], dtype=int) + np.array([0, roi[0], roi[2]])
        print("maximum intensity of the scan find at %s" % str(pch))

        mask_3D = np.repeat(self.mask[np.newaxis, roi[0]:roi[1], roi[2]:roi[3]], self.npoints, axis=0)

        if roi_order == 'XY':
            roi = self.eiger_roi_converter(roi)
        return dataset, mask_3D, pch, roi

    def eiger_load_images(self, roi=None, width=None, roi_order='YX', show_cen_image=False):
        """
        Load the eiger images in the scan.

        The maximum integrated diffraction intensity in the region of interest will be located.
        The diffraction intensity will be cutted around the highest intensity.

        Parameters
        ----------
        roi : list, optional
            The region of interest on the detector. The default is None.
        roi_oder : str, optional
            If roi_order is 'XY', the roi is described in [Xmin, Xmax, Ymin, Ymax] order
            If roi_order is 'YX', the roi is described in [Ymin, Ymax, Xmin, Xmax] order
            The default is 'YX'.
        width : list, optional
            The half width for cutting around the highest intensity. The default is None.
        show_cen_image : bool, optional
            If true, the center image in the scan will be plotted for the selection of the roi. The default is False.

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
        print('Loading data....')
        assert (len(width) == 2 or len(width) == 4), 'The cut width must contain either two or four integer numbers!'
        dataset = np.zeros((self.npoints, self.detector_size[0], self.detector_size[1]))
        if roi is None:
            roi = [0, self.detector_size[0], 0, self.detector_size[1]]
        else:
            if roi_order == 'XY':
                roi = self.eiger_roi_converter(roi)
            roi = self.eiger_roi_check(roi)

        pch = np.zeros(3, dtype=int)
        ion_bl_int = self.get_scan_data('ion_bl')
        ion_bl_int = ion_bl_int / np.round(np.average(ion_bl_int))

        for i in range(self.npoints):
            pathimg = self.path_eiger_img % (self.p08_file, self.scan, i // 2000 + 1)
            f = h5py.File(pathimg, "r")
            image = np.array(f['entry/data/data'][i % 2000, :, :], dtype=float)
            image = self.eiger_mask_correction(image)
            dataset[i, :, :] = image / ion_bl_int[i]
            sys.stdout.write('\rprogress:%d%%' % ((i + 1) * 100.0 / self.npoints))
            sys.stdout.flush()

        print()
        if show_cen_image:
            plt.imshow(np.log10(dataset[int(self.npoints / 2), :, :] + 1.0), cmap='jet')
            plt.show()

        roi_int = np.sum(dataset[:, roi[0]:roi[1], roi[2]:roi[3]], axis=(1, 2))
        self.add_scan_data('%s_roi1' % self.detector, roi_int)
        self.add_motor_pos('%s_roi1' % self.detector, roi)

        pch = np.array([np.argmax(roi_int), np.argmax(np.sum(dataset[:, roi[0]:roi[1], roi[2]:roi[3]], axis=(0, 2))), np.argmax(np.sum(dataset[:, roi[0]:roi[1], roi[2]:roi[3]], axis=(0, 1)))], dtype=int) + np.array([0, roi[0], roi[2]])
        print("maximum intensity of the scan find at " + str(pch))

        if width is None:
            width = [400, 400]
        if (len(width) == 2):
            if (roi_order == 'XY'):
                width = np.array(width, dtype=int)
                width[[0, 1]] = width[[1, 0]]
            width = self.eiger_cut_check(pch[1:], width)
            dataset = dataset[:, (pch[1] - width[0]):(pch[1] + width[0]), (pch[2] - width[1]):(pch[2] + width[1])]
            mask_3D = np.repeat(self.mask[np.newaxis, (pch[1] - width[0]):(pch[1] + width[0]), (pch[2] - width[1]):(pch[2] + width[1])], self.npoints, axis=0)
        elif (len(width) == 4):
            if (roi_order == 'XY'):
                width = np.array(width, dtype=int)
                width[[0, 1, 2, 3]] = width[[2, 3, 0, 1]]
            width = self.eiger_cut_check(pch[1:], width)
            dataset = dataset[:, (pch[1] - width[0]):(pch[1] + width[1]), (pch[2] - width[2]):(pch[2] + width[3])]
            mask_3D = np.repeat(self.mask[np.newaxis, (pch[1] - width[0]):(pch[1] + width[1]), (pch[2] - width[2]):(pch[2] + width[3])], self.npoints, axis=0)

        return dataset, mask_3D, pch, width

    def eiger_roi_sum(self, rois, roi_order='YX', save_img_sum=True):
        """
        Calculate the integrated intensity in different region of interests.

        Parameters
        ----------
        rois : ndarray
            Region of interest to be integrated. Rois should be given in the form of [roi1, roi2, roi3].
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
        rois = np.array(rois, dtype=int)
        assert rois.ndim == 2, 'Region of interest should be described as two dimensional arrays! If only one roi is needed, then rois = [roi].'
        assert rois.shape[1] == 4, 'The roi must contain four integers!'
        num_of_rois = rois.shape[0]

        for i in range(num_of_rois):
            roi = rois[i, :]
            if roi_order == 'XY':
                roi = self.eiger_roi_converter(roi)
            rois[i, :] = self.eiger_roi_check(roi)

        img_sum = np.zeros((self.detector_size[0], self.detector_size[1]))
        rois_int = np.zeros((self.npoints, num_of_rois + 1))

        for i in range(self.npoints):
            pathimg = (self.path_eiger_img) % (self.p08_file, self.scan, i // 2000 + 1)
            f = h5py.File(pathimg, "r")
            dataset = f['entry/data/data']
            image = np.array(dataset[i % 2000, :, :], dtype=float)
            image = self.eiger_mask_correction(image)
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
            np.save(self.path_eiger_imgsum, img_sum)
        return

    def eiger_img_sum(self, sum_img_num=None, save_img_sum=True):
        """
        Integrate the diffraction patterns in the scan.

        Parameters
        ----------
        sum_img_num : int, optional
            The number of images to be summed up. The default is None.
        save_img_sum : bool, optional
            If true the integrate diffraction pattern will be saved. The default is True.

        Returns
        -------
        img_sum : ndarray
            The integrated diffraction intensity.

        """
        img_sum = np.zeros((self.detector_size[0], self.detector_size[1]))
        if sum_img_num is None:
            sum_img_num = self.npoints

        for i in range(sum_img_num):
            pathimg = (self.path_eiger_img) % (self.p08_file, self.scan, i // 2000 + 1)
            f = h5py.File(pathimg, "r")
            dataset = f['entry/data/data']
            image = np.array(dataset[i % 2000, :, :], dtype=float)
            image = self.eiger_mask_correction(image)
            img_sum += image
            sys.stdout.write('\rprogress:%d%%' % ((i + 1) * 100.0 / self.npoints))
            sys.stdout.flush()

        print('')
        if self.pathsave != '' and save_img_sum:
            np.save(self.path_eiger_imgsum, img_sum)
        return img_sum

    def eiger_peak_pos_per_frame(self, cut_width=[20, 20], save_img_sum=False):
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
        print('Loading data....')
        X_pos = np.zeros(self.npoints)
        Y_pos = np.zeros(self.npoints)
        int_ar = np.zeros(self.npoints)
        img_sum = np.zeros((2 * cut_width[0] + 1, 2 * cut_width[1] + 1))

        for i in range(self.npoints):
            pathimg = (self.path_eiger_img) % (self.p08_file, self.scan, i // 2000 + 1)
            f = h5py.File(pathimg, "r")
            dataset = f['entry/data/data']
            image = np.array(dataset[i % 2000, :, :], dtype=float)
            image = self.eiger_mask_correction(image)
            Y_pos[i], X_pos[i] = np.unravel_index(np.argmax(image), image.shape)
            Y_pos[i] = np.clip(Y_pos[i], cut_width[0], self.detector_size[0] - cut_width[0] - 1)
            X_pos[i] = np.clip(X_pos[i], cut_width[1], self.detector_size[1] - cut_width[1] - 1)
            Y_shift, X_shift = measurements.center_of_mass(image[int(Y_pos[i] - cut_width[0]):int(Y_pos[i] + cut_width[0] + 1), int(X_pos[i] - cut_width[1]):int(X_pos[i] + cut_width[1] + 1)])
            Y_pos[i] = Y_pos[i] + Y_shift - cut_width[0]
            X_pos[i] = X_pos[i] + X_shift - cut_width[1]
            if save_img_sum:
                img_sum += image[int(Y_pos[i] - cut_width[0]):int(Y_pos[i] + cut_width[0] + 1), int(X_pos[i] - cut_width[1]):int(X_pos[i] + cut_width[1] + 1)]
            int_ar[i] = np.sum(image[int(Y_pos[i] - cut_width[0]):int(Y_pos[i] + cut_width[0] + 1), int(X_pos[i] - cut_width[1]):int(X_pos[i] + cut_width[1] + 1)])
            sys.stdout.write('\rprogress:%d%%' % ((i + 1) * 100.0 / self.npoints))
            sys.stdout.flush()

        print()
        if self.pathsave != '' and save_img_sum:
            np.save(self.path_eiger_imgsum, img_sum)
        return X_pos, Y_pos, int_ar

    def eiger_find_peak_position(self, roi=None, roi_order='YX', cut_width=None):
        """
        Find the peak position in the scan.

        Parameters
        ----------
        roi : list, optional
            The region of interest. The default is None.
        roi_oder : str, optional
            If roi_order is 'XY', the roi is described in [Xmin, Xmax, Ymin, Ymax] order
            If roi_order is 'YX', the roi is described in [Ymin, Ymax, Xmin, Xmax] order
            The default order is 'YX'.
        cut_width : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        pch : list
            The position of the highest integrated diffraction intensity.
            pch[0] corresponds to the frame with the maximum intensity in the rocking curve
            pch[1] corresponds to the peak position Y on the detector
            pch[2] corresponds to the peak position X on the detector

        """
        print("Finding the frames with the highest intensity....")
        if roi is None:
            roi = [0, self.detector_size[0], 0, self.detector_size[1]]
        else:
            if roi_order == 'XY':
                roi = self.eiger_roi_converter(roi)
                cut_width = np.array(cut_width, dtype=int)
                cut_width[[0, 1]] = cut_width[[1, 0]]
            roi = self.eiger_roi_check(roi)

        roi_int = np.zeros(self.npoints)
        pch = np.zeros(3)
        ion_bl_int = self.get_scan_data('ion_bl')
        ion_bl_int = ion_bl_int / np.round(np.average(ion_bl_int))

        for i in range(self.npoints):
            pathimg = self.path_eiger_img % (self.p08_file, self.scan, i // 2000 + 1)
            f = h5py.File(pathimg, "r")
            image = np.array(f['entry/data/data'][i % 2000, :, :], dtype=float)
            image = self.eiger_mask_correction(image)
            roi_int[i] = np.sum(image[roi[0]:roi[1], roi[2]:roi[3]] / ion_bl_int[i])
            sys.stdout.write('\rprogress:%d%%' % ((i + 1) * 100.0 / self.npoints))
            sys.stdout.flush()

        pch[0] = int(np.argmax(roi_int))
        image = self.eiger_load_single_image(pch[0])
        pch[-2:] = measurements.center_of_mass(image[roi[0]:roi[1], roi[2]:roi[3]]) + np.array([roi[0], roi[2]])
        if cut_width is not None:
            pch[-2:] = measurements.center_of_mass(image[int(pch[1] - cut_width[0]):int(pch[1] + cut_width[0]), int(pch[2] - cut_width[1]):int(pch[2] + cut_width[1])]) + np.array([int(pch[1] - cut_width[0]), int(pch[2] - cut_width[1])])
        print("")
        print("peak position on the detector (Z, Y, X): " + str(pch))
        return pch

    def eiger_ptycho_cxi(self, cen, cut_width, detector_distance=5.0, pixel_size=75e-6, index_array=None):
        """
        Convert the 2D Scans with Eiger 4M to the CXI format to generate the input for PyNX.

        Generated according to the PyNX package (http://ftp.esrf.fr/pub/scisoft/PyNX/doc/).

        Parameters
        ----------
        cen : list
            The central position of the direct beam on the detector in the [Y, X] form.
        cut_width : list
            The half width for cutting around the direct beam position.
        detector_distance : float, optional
            The detector distance in meter. The default is 5.0.
        pixel_size : float, optional
            The pixel_size of the detector in meter. The default is 75e-6.
        index_array : ndarray, optional
            1D boolen array indicating which images should be used for the ptychography calculation. The default is None.

        Returns
        -------
        None.

        """
        path_save_cxi = os.path.join(self.pathsave, '%s_%05d.cxi' % (self.p08_file, self.scan))
        print('Create cxi file: %s' % path_save_cxi)
        f = h5py.File(path_save_cxi, "w")
        f.attrs['file_name'] = path_save_cxi
        f.attrs['file_time'] = self.get_start_time().strftime("%Y-%m-%dT%H:%M:%S")
        f.attrs['creator'] = 'p08'
        f.attrs['HDF5_Version'] = h5py.version.hdf5_version
        f.attrs['h5py_version'] = h5py.version.version
        f.attrs['default'] = 'entry_1'
        f.create_dataset("cxi_version", data=140)

        entry_1 = f.create_group("entry_1")
        entry_1.create_dataset('start_time', data=self.get_start_time().strftime("%Y-%m-%dT%H:%M:%S"))
        entry_1.attrs['NX_class'] = 'NXentry'
        entry_1.attrs['default'] = 'data_1'

        sample_1 = entry_1.create_group("sample_1")
        sample_1.attrs['NX_class'] = 'NXsample'

        command_infor = self.get_command_infor()
        geometry_1 = sample_1.create_group("geometry_1")
        sample_1.attrs['NX_class'] = 'NXgeometry'  # Deprecated NeXus class, move to NXtransformations

        if index_array is None:
            xyz = np.zeros((3, self.npoints), dtype=np.float32)
            xyz[0] = self.get_scan_data(command_infor['motor1_name']) * 1.0e-6
            xyz[1] = self.get_scan_data(command_infor['motor2_name']) * 1.0e-6
        else:
            xyz = np.zeros((3, np.sum(index_array)), dtype=np.float32)
            xyz[0] = (self.get_scan_data(command_infor['motor1_name']) * 1.0e-6)[index_array]
            xyz[1] = (self.get_scan_data(command_infor['motor2_name']) * 1.0e-6)[index_array]
        geometry_1.create_dataset("translation", data=xyz)

        data_1 = entry_1.create_group("data_1")
        data_1.attrs['NX_class'] = 'NXdata'
        data_1.attrs['signal'] = 'data'
        data_1.attrs['interpretation'] = 'image'
        data_1["translation"] = h5py.SoftLink('/entry_1/sample_1/geometry_1/translation')

        instrument_1 = entry_1.create_group("instrument_1")
        instrument_1.attrs['NX_class'] = 'NXinstrument'
        instrument_1.create_dataset("name", data='six_circle_diffractometer')

        source_1 = instrument_1.create_group("source_1")
        source_1.attrs['NX_class'] = 'NXsource'
        # Convert the energy to J
        nrj = self.get_motor_pos('fmbenergy') * 1.60218e-19
        source_1.create_dataset("energy", data=nrj)
        source_1["energy"].attrs['note'] = 'Incident photon energy (instead of source energy), for CXI compatibility'

        detector_1 = instrument_1.create_group("detector_1")
        detector_1.attrs['NX_class'] = 'NX_detector'
        if index_array is None:
            iobs = detector_1.create_dataset("data", (self.npoints, cut_width[0] * 2, cut_width[1] * 2), chunks=(1, cut_width[0] * 2, cut_width[1] * 2), shuffle=True, compression="gzip")
        else:
            iobs = detector_1.create_dataset("data", (int(np.sum(index_array)), cut_width[0] * 2, cut_width[1] * 2), chunks=(1, cut_width[0] * 2, cut_width[1] * 2), shuffle=True, compression="gzip")
        cut_width = self.eiger_cut_check(cen, cut_width)
        petra_current = self.get_scan_data('ion_bl')
        petra_current = petra_current / np.round(np.average(petra_current))
        if index_array is None:
            for i in range(self.npoints):
                image = self.eiger_load_single_image(i)
                print(i)
                iobs[i, :, :] = self.eiger_mask_correction(image)[(cen[0] - cut_width[0]):(cen[0] + cut_width[0]), (cen[1] - cut_width[1]):(cen[1] + cut_width[1])] / petra_current[i]
        else:
            for i, num in enumerate(np.arange(self.npoints)[index_array]):
                image = self.eiger_load_single_image(num)
                print(num)
                iobs[i, :, :] = self.eiger_mask_correction(image)[(cen[0] - cut_width[0]):(cen[0] + cut_width[0]), (cen[1] - cut_width[1]):(cen[1] + cut_width[1])] / petra_current[num]
        detector_1.create_dataset("distance", data=detector_distance)
        detector_1["distance"].attrs['units'] = 'm'
        detector_1.create_dataset("x_pixel_size", data=pixel_size)
        detector_1["x_pixel_size"].attrs['units'] = 'm'
        detector_1.create_dataset("y_pixel_size", data=pixel_size)
        detector_1["y_pixel_size"].attrs['units'] = 'm'
        mask_cut = self.mask[(cen[0] - cut_width[0]):(cen[0] + cut_width[0]), (cen[1] - cut_width[1]):(cen[1] + cut_width[1])]
        detector_1.create_dataset("mask", data=mask_cut, chunks=True, shuffle=True, compression="gzip")
        detector_1["mask"].attrs['note'] = "Mask of invalid pixels, applying to each frame"
        basis_vectors = np.zeros((2, 3), dtype=np.float32)
        basis_vectors[0, 1] = -pixel_size
        basis_vectors[1, 0] = -pixel_size
        detector_1.create_dataset("basis_vectors", data=basis_vectors)

        detector_1["translation"] = h5py.SoftLink('/entry_1/sample_1/geometry_1/translation')
        data_1["data"] = h5py.SoftLink('/entry_1/instrument_1/detector_1/data')

        process_1 = data_1.create_group("process_1")
        process_1.attrs['NX_class'] = 'NXprocess'
        process_1.create_dataset("program", data='PyNX')  # NeXus spec
        # process_1.create_dataset("version", data="%s" % __version__)  # NeXus spec
        # process_1.create_dataset("command", data=command)  # CXI spec
        config = process_1.create_group("configuration")
        config.attrs['NX_class'] = 'NXcollection'

        f.close()
        return
