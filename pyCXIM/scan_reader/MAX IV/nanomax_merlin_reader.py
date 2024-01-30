# -*- coding: utf-8 -*-
"""
Read and treat scans with merlin detector recorded at NanoMax beamline.
Created on Thu Jul  6 17:09:31 2023

@author: Ren Zhe
@email: renzhe@ihep.ac.cn
"""
import os
import numpy as np
import h5py
import hdf5plugin
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import sys
from .nanomax_scan_reader import NanoMaxScan


class NanoMaxMerlinScan(NanoMaxScan):
    """
    Read and treat the scans with merlin detector images recorded at NanoMax.

    Parameters
    ----------
    path : str
        The path for the raw file folder.
    sample_name : str
        The name of the sample defined by nanomax.
    scan : int
        The scan number.
    detector : str, optional
        The name of the detecter. The default is 'merlin'.
    pathsave : str, optional
        The folder to save the results, if not given no results will be saved. The default is ''.
    pathmask : str, optional
        The path of the detector mask. If not given, an empty mask will be generated. The default is ''.
    creat_save_folder : bool, optional
        Whether the save folder should be created. The default is True.

    Returns
    -------
    None.

    """

    def __init__(self, path, sample_name, scan, detector='merlin', pathsave='', pathmask='', creat_save_folder=True):
        super().__init__(path, sample_name, scan, pathsave, creat_save_folder)
        self.detector = detector
        self.path_merlin_imgsum = os.path.join(self.pathsave, '%s_scan%05d_%s_imgsum.npy' % (self.sample_name, self.scan, 'merlin'))
        self.add_header_infor('detector')

        scanfile = h5py.File(self.pathh5, 'r')
        assert ('entry/measurement/merlin/frames') in scanfile, 'Merlin detector data does not exists, please check it again!'
        scanfile.close()

        self.detector_size = (515, 515)
        self.pixel_size = 55e-3

        self.merlin_load_mask(pathmask)
        return

    def merlin_load_mask(self, pathmask):
        """
        Load the mask files defining the bad pixels on the detector.

        The mask file is padded with 0 and 1.
        0 means that the pixel is not masked and 1 means that the pixel is masked.
        If the mask file does not exist, an empty mask filled with zeros will be generated.

        Parameters
        ----------
        pathmask : str, optional
            The path for the mask file. The default is "".

        Returns
        -------
        ndarray
            The generated mask file.

        """
        if os.path.exists(pathmask):
            print('Predefined mask loaded')
            self.mask = np.load(pathmask)
        else:
            print('Could not find the predefined mask, generate an empty mask instead!')
            self.mask = np.zeros((self.detector_size[0], self.detector_size[1]))
        return self.mask

    def merlin_mask_correction(self, image, mode='constant'):
        """
        Correction of the intensities of the masked pixels.

        Parameters
        ----------
        image : ndarray
            The original images to be corrected.
        mode : str, optional
            If mode is 'constant', intensity of the masked pixels will be set to zero.
            If the mode is 'medianfilter', the intensity of the masked pixels will be set to the median filter value according the surrounding pixels.
            The default is 'constant'.

        Returns
        -------
        image : ndarray
            The corrected intensity.

        """
        if mode == 'constant':
            image[self.mask == 1] = 0
        elif mode == 'medianfilter':
            image_filtered = median_filter(image, size=3)
            image[self.mask == 1] = image_filtered[self.mask == 1]
        return image

    def merlin_load_single_image(self, img_index, mask_correction=True, correction_mode='constant'):
        """
        Load a single merlin detector image in the scan.

        Parameters
        ----------
        img_index : int
            The index of the image in the scan.
        mask_correction : bool, optional
            If true, the intensity of the masked pixels will be changed according to the correction mode given. The default is True.
        mode : str, optional
            If mode is 'constant', intensity of the masked pixels will be set to zero.
            If the mode is 'medianfilter', the intensity of the masked pixels will be set to the median filter value according the surrounding pixels.
            The default is 'constant'.

        Returns
        -------
        image : ndarray
            The result image in the scan.

        """
        assert img_index < self.npoints, 'The image number wanted is larger than the total image number in the scan!'
        img_index = int(img_index)

        scanfile = h5py.File(self.pathh5, 'r')

        dataset = scanfile['entry/measurement/merlin/frames']
        image = np.array(dataset[img_index, :, :], dtype=float)
        scanfile.close()
        if mask_correction:
            image = self.merlin_mask_correction(image, correction_mode)
        return image

    def merlin_load_rois(self, roi=None, show_cen_image=False):
        """
        Load the images with certain region of interest.

        Parameters
        ----------
        roi : list, optional
            The region of interest in [Ymin, Ymax, Xmin, Xmax] order.
            If not given, the complete detector image will be loaded. The default is None.
        show_cen_image : bool, optional
            If true the central image in the scan will be shown to help select the rois. The default is False.

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
            roi = self.merlin_roi_check(roi)

        roi = np.array(roi, dtype=int)
        if show_cen_image:
            plt.imshow(np.log10(self.merlin_load_single_image(int(self.npoints / 2)) + 1.0), cmap='jet')
            plt.show()

        dataset = np.zeros((self.npoints, roi[1] - roi[0], roi[3] - roi[2]))
        pch = np.zeros(3, dtype=int)
        normal_int = self.get_scan_data('1', detector_name='alba2')
        normal_int = normal_int / np.average(normal_int)
        scanfile = h5py.File(self.pathh5, 'r')
        h5dataset = scanfile['entry/measurement/merlin/frames']
        for img_index in range(self.npoints):
            image = np.array(h5dataset[img_index, :, :], dtype=float)
            image = self.merlin_mask_correction(image, 'medianfilter')
            dataset[img_index, :, :] = image[roi[0]:roi[1], roi[2]:roi[3]] / normal_int[img_index]
            sys.stdout.write('\rprogress:%d%%' % ((img_index + 1) * 100.0 / self.npoints))
            sys.stdout.flush()
        print()
        scanfile.close()

        roi_int = np.sum(dataset, axis=(1, 2))
        self.add_scan_data('%s_roi1' % self.detector, roi_int)
        self.add_motor_pos('%s_roi1' % self.detector, list(roi))

        pch = np.array([np.argmax(np.sum(dataset, axis=(1, 2))), np.argmax(np.sum(dataset, axis=(0, 2))), np.argmax(np.sum(dataset, axis=(0, 1)))], dtype=int) + np.array([0, roi[0], roi[2]])
        print("maximum intensity of the scan find at %s" % str(pch))

        mask_3D = np.repeat(self.mask[np.newaxis, roi[0]:roi[1], roi[2]:roi[3]], self.npoints, axis=0)
        return dataset, mask_3D, pch, roi

    def merlin_load_images(self, roi=None, width=None, show_cen_image=False):
        """
        Load the merlin images in the scan.

        The maximum integrated diffraction intensity in the region of interest will be located.
        The diffraction intensity will be cutted around the highest intensity.

        Parameters
        ----------
        roi : list, optional
            The region of interest on the detector. The roi is described in [Ymin, Ymax, Xmin, Xmax] order. The default is None.
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
        dataset = np.zeros((self.npoints, self.detector_size[0], self.detector_size[1]))
        if roi is None:
            roi = [0, self.detector_size[0], 0, self.detector_size[1]]
        else:
            roi = self.merlin_roi_check(roi)

        roi = np.array(roi, dtype=int)
        pch = np.zeros(3, dtype=int)
        normal_int = self.get_scan_data('1', detector_name='alba2')
        normal_int = normal_int / np.average(normal_int)
        scanfile = h5py.File(self.pathh5, 'r')
        h5dataset = scanfile['entry/measurement/merlin/frames']
        for img_index in range(self.npoints):
            image = np.array(h5dataset[img_index, :, :], dtype=float)
            image = self.merlin_mask_correction(image, 'medianfilter')
            dataset[img_index, :, :] = image / normal_int[img_index]
            sys.stdout.write('\rprogress:%d%%' % ((img_index + 1) * 100.0 / self.npoints))
            sys.stdout.flush()
        print()
        scanfile.close()

        if show_cen_image:
            plt.imshow(np.log10(dataset[int(self.npoints / 2), :, :] + 1.0), cmap='jet')
            plt.show()

        roi_int = np.sum(dataset[:, roi[0]:roi[1], roi[2]:roi[3]], axis=(1, 2))
        self.add_scan_data('%s_roi1' % self.detector, roi_int)
        self.add_motor_pos('%s_roi1' % self.detector, list(roi))

        pch = np.array([np.argmax(roi_int), np.argmax(np.sum(dataset[:, roi[0]:roi[1], roi[2]:roi[3]], axis=(0, 2))), np.argmax(np.sum(dataset[:, roi[0]:roi[1], roi[2]:roi[3]], axis=(0, 1)))], dtype=int) + np.array([0, roi[0], roi[2]])
        print("maximum intensity of the scan find at " + str(pch))

        if width is None:
            width = [400, 400]
        if (len(width) == 2):
            width = self.merlin_cut_check(pch[1:], width)
            dataset = dataset[:, (pch[1] - width[0]):(pch[1] + width[0]), (pch[2] - width[1]):(pch[2] + width[1])]
            mask_3D = np.repeat(self.mask[np.newaxis, (pch[1] - width[0]):(pch[1] + width[0]), (pch[2] - width[1]):(pch[2] + width[1])], self.npoints, axis=0)
        elif (len(width) == 4):
            width = self.merlin_cut_check(pch[1:], width)
            dataset = dataset[:, (pch[1] - width[0]):(pch[1] + width[1]), (pch[2] - width[2]):(pch[2] + width[3])]
            mask_3D = np.repeat(self.mask[np.newaxis, (pch[1] - width[0]):(pch[1] + width[1]), (pch[2] - width[2]):(pch[2] + width[3])], self.npoints, axis=0)

        return dataset, mask_3D, pch, width

    def merlin_img_sum(self, sum_img_num=None, save_img_sum=True):
        """
        Get the sum of the merlin detector images.

        Parameters
        ----------
        sum_img_num : int, optional
            The number of images to be summed up. The default is None.
        save_img_sum : bool, optional
            Determines whether the image will be s. The default is True.

        Returns
        -------
        image_sum : ndarray
            DESCRIPTION.

        """
        if sum_img_num is None:
            sum_img_num = self.npoints

        scanfile = h5py.File(self.pathh5, 'r')
        dataset = scanfile['entry/measurement/merlin/frames']
        image_sum = np.zeros((515, 515))
        for i in range(sum_img_num):
            image_sum = image_sum + np.array(dataset[i, :, :], dtype=float)

        if self.pathsave != '' and save_img_sum:
            np.save(self.path_merlin_imgsum, image_sum)
            self.add_scan_infor('path_merlin_imgsum')
        return image_sum

    def merlin_roi_check(self, roi):
        """
        Check the roi size for the merlin detector.

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

    def merlin_roi_converter(self, roi):
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

    def merlin_cut_check(self, cen, width):
        """
        Cut the maximum symmetric width that can be cutted around the peak position on the detector.

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

    def merlin_roi_sum(self, rois, roi_order='YX', save_img_sum=True, normalize=True):
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
                roi = self.merlin_roi_converter(roi)
            rois[i, :] = self.merlin_roi_check(roi)

        img_sum = np.zeros((515, 515))
        rois_int = np.zeros((self.npoints, num_of_rois + 1))
        scanfile = h5py.File(self.pathh5, 'r')
        assert ('entry/measurement/merlin/frames') in scanfile, 'Merlin detector data does not exists, please check it again!'

        h5dataset = scanfile['entry/measurement/merlin/frames']

        for i in range(self.npoints):
            image = np.array(h5dataset[i, :, :], dtype=float)
            image = self.merlin_mask_correction(image, 'constant')

            rois_int[i, 0] = np.sum(image[:, :])
            for j in range(num_of_rois):
                roi = rois[j, :]
                rois_int[i, j + 1] = np.sum(image[roi[0]:roi[1], roi[2]:roi[3]])
            img_sum += image
            sys.stdout.write('\rprogress:%d%%' % ((i + 1) * 100.0 / self.npoints))
            sys.stdout.flush()

        scanfile.close()
        if normalize:
            normal_int = self.get_scan_data('1', detector_name='alba2')
            normal_int = normal_int / np.average(normal_int)
            for j in range(num_of_rois + 1):
                rois_int[:, j] = rois_int[:, j] / normal_int

        print('')
        self.add_scan_data('%s_full' % self.detector, rois_int[:, 0])
        for j in range(num_of_rois):
            self.add_motor_pos('%s_roi%d' % (self.detector, (j + 1)), list(rois[j, :]))
            self.add_scan_data('%s_roi%d' % (self.detector, (j + 1)), rois_int[:, j + 1])

        if self.pathsave != '' and save_img_sum:
            np.save(self.path_merlin_imgsum, img_sum)
            self.add_scan_infor('path_merlin_imgsum')
        return
