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
from ..general_detector import DetectorMixin

class NanoMaxMerlinScan(NanoMaxScan, DetectorMixin):
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

    def __init__(self, beamline, path, sample_name, scan, detector='merlin', pathsave='', pathmask='', creat_save_folder=True):
        super().__init__(beamline, path, sample_name, scan, pathsave, creat_save_folder)

        self.detector = detector
        self.path_imgsum = os.path.join(self.pathsave, '%s_scan%05d_%s_imgsum.npy' % (self.sample_name, self.scan, 'merlin'))
        self.add_header_infor('detector')

        scanfile = h5py.File(self.pathh5, 'r')
        assert ('entry/measurement/merlin/frames') in scanfile, 'Merlin detector data does not exists, please check it again!'
        scanfile.close()

        self.detector_size = (515, 515)
        self.pixel_size = 55e-3

        self.load_mask(pathmask)
        return

    def load_single_image(self, img_index, correction_mode='constant'):
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

        with h5py.File(self.pathh5, 'r') as scanfile:
            dataset = scanfile['entry/measurement/merlin/frames']
            image = np.array(dataset[img_index, :, :], dtype=float)
        image = self.image_mask_correction(image, correction_mode=correction_mode)
        return image