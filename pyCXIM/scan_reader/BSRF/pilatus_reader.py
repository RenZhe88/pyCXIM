# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 10:28:55 2023

@author: ren zhe
@email: renzhe@ihep.ac.cn
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import center_of_mass
import os
import sys
from .spec_reader import BSRFScanImporter


class BSRFPilatusImporter(BSRFScanImporter):
    """
    Read and treat the scans with pilatus detector. It is a child class of BSRFScanImporter.

    Parameters
    ----------
    beamline : str
        The name of the beamline. Now only '1w1a' is supported.
    path : str
        The path for the raw file folder.
    sample_name : str
        The name of the sample defined by the spec_newfile name in the system.
    scan_num : int
        The scan number.
    detector : str, optional
        The name of the detecter. Now only pilatus '300K-A' is supported. The default is '300K-A'.
    pathsave : str, optional
        The folder to save the results, if not given no results will be saved. The default is ''.
    pathmask : str, optional
        The path of the detector mask. If not given, mask will be generated according to the hot pixels in the first image of the scan. The default is ''.
    creat_save_folder : bool, optional
        Whether the save folder should be created. The default is True.

    Raises
    ------
    KeyError
        If the detector type is not previously registered, raise KeyError.
    IOError
        If the image folder does not exist, raise IOError.

    Returns
    -------
    None.

    """

    def __init__(self, beamline, path, sample_name, scan_num, detector='300K-A', pathsave='', pathmask='', creat_save_folder=True):
        super().__init__(beamline, path, sample_name, scan_num, pathsave, creat_save_folder)

        self.detector = detector
        if self.detector == '300K-A':
            self.detector_size = (195, 487)
            self.pixel_size = 172e-3

        if beamline == '1w1a':
            self.detector_size = (487, 195)
            self.path_pilatus_folder = os.path.join(self.path, r'images\%s\S%03d' % (self.sample_name, self.scan))
            self.path_pilatus_img = os.path.join(self.path_pilatus_folder, "%s_S%03d_%05d.tif")
            self.path_pilatus_imgsum = os.path.join(self.pathsave, '%s_scan%05d_%s_imgsum.npy' % (self.sample_name, self.scan, self.detector))
        else:
            raise KeyError('Now this code is only develop for the 1w1a beamline. If you want to use this code for other BSRF bealines, please contact the author! renzhe@ihep.ac.cn')

        if not os.path.exists(self.path_pilatus_folder):
            raise IOError('The image folder for %s images %s does not exist, please check the path again!' % (self.detector, self.path_pilatus_folder))
        self.pilatus_load_mask(pathmask)
        return

    def get_detector_pixelsize(self):
        """
        Get the pixel_size of the detector.

        Returns
        -------
        float
            The pixel size of defined detector.

        """
        return self.pixel_size

    def pilatus_load_mask(self, pathmask=""):
        """
        Load the mask files defining the bad pixels on the detector.

        The mask file is padded with 0 and 1.
        0 means that the pixel is not masked and 1 means that the pixel is masked.
        If the mask file does not exist, no pixels will be masked.

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
            print('Could not find the predefined mask, no pixels will be masked!')
            self.mask = np.zeros(self.detector_size)
        self.img_correction = 1.0 - self.mask
        return self.mask

    def pilatus_mask_correction(self, image):
        """
        Correction of the images according to the mask generated.

        The value of the masked arrays are set to zero.

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

    def pilatus_img_reader(self, pathimg):
        """
        Read a single pilatus image stored in tif format.

        Parameters
        ----------
        pathimg : str
            The path of the pilatus image.

        Returns
        -------
        image : ndarray
            The image of the pilatus detector.

        """
        with open(pathimg, 'rb') as f:
            f.seek(4096)
            image = np.fromfile(f, dtype=np.int32).astype(float)

        if self.beamline == '1w1a':
            image = image.reshape(self.detector_size[1], self.detector_size[0])
            image = np.flip(image.T, axis=1)
        else:
            image = image.reshape(self.detector_size)
        return image

    def pilatus_roi_check(self, roi):
        """
        Check the roi size for the pilatus detector.

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

    def pilatus_cut_check(self, cen, width):
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

    def pilatus_load_single_image(self, img_index, mask_correction=True):
        """
        Load a single pilatus image in the scan.

        Parameters
        ----------
        img_index : int
            The index of the image in the scan.
        mask_correction : bool, optional
            If true, the masked pixels will be set to . The default is True.

        Returns
        -------
        image : ndarray
            The result image in the scan.

        """
        assert img_index < self.npoints, \
            'The image number wanted is larger than the total image number in the scan!'
        img_index = int(img_index)

        pathimg = (self.path_pilatus_img) % (self.sample_name, self.scan, img_index)
        image = self.pilatus_img_reader(pathimg)

        if mask_correction:
            image = self.pilatus_mask_correction(image)
        return image

    def pilatus_load_rois(self, roi=None, show_cen_image=False, normalize_signal=None):
        """
        Load the images with certain region of interest.

        Parameters
        ----------
        roi : list, optional
            The region of interest in [Ymin, Ymax, Xmin, Xmax] order.
            If not given, the complete detector image will be loaded. The default is None.
        show_cen_image : bool, optional
            If true the central image of the data will be shown to help select the rois. The default is False.
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
        print('Loading data....')
        if roi is None:
            roi = [0, self.detector_size[0], 0, self.detector_size[1]]
        else:
            roi = self.pilatus_roi_check(roi)

        if show_cen_image:
            plt.imshow(np.log10(self.pilatus_load_single_image(int(self.npoints / 2)) + 1.0), cmap='jet')
            plt.show()

        dataset = np.zeros((self.npoints, roi[1] - roi[0], roi[3] - roi[2]))
        pch = np.zeros(3, dtype=int)

        if (normalize_signal is None) and self.beamline == '1w1a':
            normalize_signal = 'Monitor'
        else:
            assert (normalize_signal in self.get_counter_names), "The given signal for the normalization does not exist in the scan!"

        normal_int = self.get_scan_data(normalize_signal)
        normal_int = normal_int / np.round(np.average(normal_int))

        for i in range(self.npoints):
            pathimg = (self.path_pilatus_img) % (self.sample_name, self.scan, i)
            image = self.pilatus_img_reader(pathimg)
            image = self.pilatus_mask_correction(image)
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

    def pilatus_load_images(self, roi=None, width=None, show_cen_image=False, normalize_signal=None):
        """
        Load the pilatus images in the scan.

        The maximum integrated diffraction intensity in the region of interest will be located.
        The diffraction intensity will be cutted around the highest intensity.

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
        print('Loading data....')
        dataset = np.zeros((self.npoints, self.detector_size[0], self.detector_size[1]))
        if roi is None:
            roi = [0, self.detector_size[0], 0, self.detector_size[1]]
        else:
            roi = self.pilatus_roi_check(roi)

        if (normalize_signal is None) and self.beamline == '1w1a':
            normalize_signal = 'Monitor'
        else:
            assert (normalize_signal in self.get_counter_names), "The given signal for the normalization does not exist in the scan!"
        normal_int = self.get_scan_data(normalize_signal)
        normal_int = normal_int / np.round(np.average(normal_int))

        pch = np.zeros(3, dtype=int)
        for i in range(self.npoints):
            pathimg = (self.path_pilatus_img) % (self.sample_name, self.scan, i)
            image = self.pilatus_img_reader(pathimg)
            image = self.pilatus_mask_correction(image)
            dataset[i, :, :] = image / normal_int[i]
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
            width = self.pilatus_cut_check(pch[1:], width)
            dataset = dataset[:, (pch[1] - width[0]):(pch[1] + width[0]), (pch[2] - width[1]):(pch[2] + width[1])]
            mask_3D = np.repeat(self.mask[np.newaxis, (pch[1] - width[0]):(pch[1] + width[0]), (pch[2] - width[1]):(pch[2] + width[1])], self.npoints, axis=0)
        elif (len(width) == 4):
            width = self.pilatus_cut_check(pch[1:], width)
            dataset = dataset[:, (pch[1] - width[0]):(pch[1] + width[1]), (pch[2] - width[2]):(pch[2] + width[3])]
            mask_3D = np.repeat(self.mask[np.newaxis, (pch[1] - width[0]):(pch[1] + width[1]), (pch[2] - width[2]):(pch[2] + width[3])], self.npoints, axis=0)

        return dataset, mask_3D, pch, width

    def pilatus_roi_sum(self, rois, save_img_sum=True):
        """
        Calculate the integrated intensity in different region of interests.

        Parameters
        ----------
        rois : ndarray
            Region of interest to be integrated. Rois should be given in the form of [roi1, roi2, roi3].
            Each roi is described in [Ymin, Ymax, Xmin, Xmax] order.
        save_img_sum : bool, optional
            If true, the integrated diffraction pattern of the entire scan will be saved. The default is True.

        Returns
        -------
        None.

        """
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
            rois[i, :] = self.pilatus_roi_check(rois[i, :])

        img_sum = np.zeros((self.detector_size[0], self.detector_size[1]))
        rois_int = np.zeros((self.npoints, num_of_rois + 1))

        for i in range(self.npoints):
            pathimg = (self.path_pilatus_img) % (self.sample_name, self.scan, i)
            image = self.pilatus_img_reader(pathimg)
            image = self.pilatus_mask_correction(image)
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
            np.save(self.path_pilatus_imgsum, img_sum)
        return

    def pilatus_peak_pos_per_frame(self, cut_width=[30, 30], save_img_sum=False):
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
        img_sum = np.zeros((2 * cut_width[0], 2 * cut_width[1]))

        for i in range(self.npoints):
            pathimg = (self.path_pilatus_img) % (self.sample_name, self.scan, i)
            image = self.pilatus_img_reader(pathimg)
            image = self.pilatus_mask_correction(image)
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
            np.save(self.path_pilatus_imgsum, img_sum)
        return X_pos, Y_pos, int_ar

    def pilatus_find_peak_position(self, roi=None, cut_width=None, normalize_signal=None):
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
        print("Finding the frames with the highest intensity....")
        if roi is None:
            roi = [0, self.detector_size[0], 0, self.detector_size[1]]
        else:
            roi = self.pilatus_roi_check(roi)

        roi_int = np.zeros(self.npoints)
        pch = np.zeros(3)

        if (self.beamline is None) or (self.beamline == '1w1a'):
            normalize_signal = 'Monitor'

        normal_int = self.get_scan_data(normalize_signal)
        normal_int = normal_int / np.round(np.average(normal_int))

        for i in range(self.npoints):
            pathimg = (self.path_pilatus_img) % (self.sample_name, self.scan, i)
            image = self.pilatus_img_reader(pathimg)
            image = self.pilatus_mask_correction(image)
            roi_int[i] = np.sum(image[roi[0]:roi[1], roi[2]:roi[3]] / normal_int[i])
            sys.stdout.write('\rprogress:%d%%' % ((i + 1) * 100.0 / self.npoints))
            sys.stdout.flush()

        plt.plot(roi_int)
        plt.show()

        pch[0] = int(np.argmax(roi_int))
        image = self.pilatus_load_single_image(pch[0])
        pch[-2:] = center_of_mass(image[roi[0]:roi[1], roi[2]:roi[3]]) + np.array([roi[0], roi[2]])
        if cut_width is not None:
            pch[-2:] = center_of_mass(image[int(pch[1] - cut_width[0]):int(pch[1] + cut_width[0]), int(pch[2] - cut_width[1]):int(pch[2] + cut_width[1])]) + np.array([int(pch[1] - cut_width[0]), int(pch[2] - cut_width[1])])
        print("")
        print("peak position on the detector (Z, Y, X): " + str(np.around(pch, 2)))
        return pch

    def load_6C_peak_infor(self, roi=None, cut_width=[50, 50]):
        """
        Load the motor positions of the six circle diffractometer.

        Parameters
        ----------
        roi : list, optional
            The region of interest. If not given, the complete detector image will be used. The default is None.
        cut_width : list, optional
            The cut width in Y, X direction. The default is [50, 50].

        Returns
        -------
        pixel_position : list
            The pixel position on the detector in [Y, X] order.
        motor_position : list
            motor positions in the order of [eta, delta, chi, phi, nu, energy]. The angles are in degree and the energy in eV.

        """
        pch = self.pilatus_find_peak_position(roi=roi, cut_width=cut_width)
        scan_motor = self.get_scan_motor()
        scan_motor_ar = self.get_scan_data(scan_motor)
        if scan_motor == 'eta':
            eta = scan_motor_ar[int(pch[0])]
            phi = self.get_motor_pos('phi')
        elif scan_motor == 'phi':
            eta = self.get_motor_pos('eta')
            phi = scan_motor_ar[int(pch[0])]
        delta = self.get_motor_pos('del')
        chi = self.get_motor_pos('chi')
        nu = self.get_motor_pos('nu')
        energy = self.get_motor_pos('energy')

        motor_position = np.array([eta, delta, chi, phi, nu, energy], dtype=float)
        pixel_position = np.array([pch[1], pch[2]])
        return pixel_position, motor_position
