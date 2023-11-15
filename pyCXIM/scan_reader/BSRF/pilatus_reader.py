# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 10:28:55 2023

@author: Lenovo
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt
from .spec_reader import BSRFScanImporter
import re
import ast

class BSRFPilatusImporter(BSRFScanImporter):

    def __init__(self, beamline, path, sample_name, scan_num, detector='300K-A', pathsave='', pathmask='', creat_save_folder=True):
        super().__init__(beamline, path, sample_name, scan_num, pathsave, creat_save_folder)

        self.detector = detector
        if self.detector == '300K-A':
            self.detector_size = (195, 487)
            self.pixel_size = 172e-3

        if beamline == '1W1A':
            self.data_format = 'tif'
            self.detector_size = (487, 195)
            self.pixel_size = 172e-3
            self.path_pilatus_folder = os.path.join(self.path, r'images\%s\S%03d' % (self.sample_name, self.scan))
            self.path_pilatus_img = os.path.join(self.path_pilatus_folder, "%s_S%03d_%05d.tif")

        assert os.path.exists(self.path_pilatus_folder), \
            'The image folder for %s images %s does not exist, please check the path again!' % (self.detector, self.path_pilatus_folder)
        self.pilatus_load_mask(pathmask)
        return

    def get_detector_pixelsize(self):
        return self.pixel_size

    def pilatus_img_reader(self, pathimg, data_format='tif'):
        if data_format == 'tif':
            with open(pathimg, 'rb') as f:
                f.seek(4096)
                image = np.fromfile(f, dtype=np.int32).astype(float)

        if self.beamline == '1W1A':
            image = image.reshape(self.detector_size[1], self.detector_size[0])
            image = np.flip(image.T, axis=1)
        else:
            image = image.reshape(self.detector_size)
        return image

    def pilatus_roi_check(self, roi):
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

    def pilatus_mask_correction(self, image):
        image = image * self.img_correction
        return image

    def pilatus_load_single_image(self, img_index, mask_correction=True):
        assert img_index < self.npoints, \
            'The image number wanted is larger than the total image number in the scan!'
        img_index = int(img_index)

        pathimg = (self.path_pilatus_img) % (self.sample_name, self.scan, img_index)
        image = self.pilatus_img_reader(pathimg, self.data_format)

        if mask_correction:
            image = self.pilatus_mask_correction(image)
        return image

    def pilatus_load_mask(self, pathmask="", threshold=1.0e7):
        self.mask = np.zeros(self.detector_size)
        self.img_correction = 1.0 - self.mask
        return self.mask

    def pilatus_peak_pos_per_frame(self, cut_width=[30, 30], save_img_sum=False):
        print('Loading data....')
        X_pos = np.zeros(self.npoints)
        Y_pos = np.zeros(self.npoints)
        int_ar = np.zeros(self.npoints)
        img_sum = np.zeros((2 * cut_width[0], 2 * cut_width[1]))

        for i in range(self.npoints):
            pathimg = (self.path_pilatus_img) % (self.sample_name, self.scan, i)
            image = self.pilatus_img_reader(pathimg, self.data_format)
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
        # if self.pathsave != '' and save_img_sum:
        #     np.save(self.path_pilatus_imgsum, img_sum)
        return X_pos, Y_pos, int_ar

    def pilatus_find_peak_position(self, roi=None, cut_width=None, normalize_signal=None):
        print("Finding the frames with the highest intensity....")
        if roi is None:
            roi = [0, self.detector_size[0], 0, self.detector_size[1]]
        else:
            roi = self.pilatus_roi_check(roi)

        roi_int = np.zeros(self.npoints)
        pch = np.zeros(3)

        normalize_signal = 'Monitor'

        normal_int = self.get_scan_data(normalize_signal)
        normal_int = normal_int / np.round(np.average(normal_int))

        for i in range(self.npoints):
            pathimg = (self.path_pilatus_img) % (self.sample_name, self.scan, i)
            image = self.pilatus_img_reader(pathimg, self.data_format)
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

def test():
    beamline = '1W1A'
    path = r'F:\Work place 4\sample\XRD\Additional Task\20231102 1W1A test data\RENZHE'
    sample_name = r'LVO_1'
    scan_num = 1
    pathsave = r'F:\Work place 4\sample\XRD\Additional Task\20231102 1W1A test data\RENZHE'
    pathmask = r''

    scan = BSRFPilatusImporter(beamline, path, sample_name, scan_num, 'pilatus', pathsave, pathmask, creat_save_folder=True)
    X_pos, Y_pos, int_ar = scan.pilatus_peak_pos_per_frame()
    plt.subplot(1, 3, 1)
    plt.plot(X_pos)
    plt.subplot(1, 3, 2)
    plt.plot(Y_pos)
    plt.subplot(1, 3, 3)
    plt.plot(int_ar)
    plt.show()

    img = scan.pilatus_load_single_image(21)
    print(scan.pilatus_find_peak_position())
    plt.imshow(np.log10(img + 1.0))
    plt.show()
    return

if __name__ == '__main__':
    test()