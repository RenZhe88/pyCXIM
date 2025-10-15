# -*- coding: utf-8 -*-
"""
Read and treat the scans recorded at 1w1a beamline BSRF with image detector.
Created on Wed Nov  1 10:28:55 2023

@author: Ren Zhe
@email: renzhe@ihep.ac.cn
"""
import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import center_of_mass
import os
import sys
import warnings
import gzip

from .spec_reader import ESRFScanImporter
from ..general_detector import DetectorMixin


class ESRFEdfImporter(ESRFScanImporter, DetectorMixin):
    """
    Read and treat the scans with image detector. It is a child class of ESRFScanImporter.

    Parameters
    ----------
    beamline : str
        The name of the beamline. Now only 'ID01' is supported.
    path : str
        The path for the raw file folder.
    sample_name : str
        The name of the sample defined by the spec_newfile name in the system.
    scan_num : int
        The scan number.
    detector : str, optional
        The name of the detecter. Now only image 'mixipix' is supported. The default is 'mixipix'.
    prefix : str, optional
        The prefix of the edf detector images. The default is ''.
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

    def __init__(self, beamline, path, sample_name, scan_num, detector='maxipix', prefix='', pathsave='', pathmask='', creat_save_folder=True):
        super().__init__(beamline, path, sample_name, scan_num, pathsave, creat_save_folder)

        self.detector = detector
        self.prefix = prefix

        if beamline == 'ID01':
            if self.detector == 'maxipix':
                self.detector_size = (516, 516)
                self.pixel_size = 55e-3
            elif self.detector == 'andor':
                self.detector_size = (2160, 2560)
                self.pixel_size = 6.5e-3
            elif self.detector == 'eiger2M':
                self.detector_size = (2164, 1030)
                self.pixel_size = 75e-3

            self.path_image_folder = os.path.join(path, 'images', '%s' % self.sample_name)
            if not os.path.exists(self.path_image_folder):
                warnings.warn('Default image folder does not exist, please change it with image folder function!', category=RuntimeWarning)
            if self.start_time < datetime.datetime(2016, 10, 1):
                self.path_img = os.path.join(self.path_image_folder, prefix + "_%04d.edf")
            else:
                self.path_img = os.path.join(self.path_image_folder, prefix + "_%05d.edf.gz")
            self.path_imgsum = os.path.join(self.pathsave, '%s_scan%05d_%s_imgsum.npy' % (self.sample_name, self.scan, self.detector))
        else:
            raise KeyError('Now this code is only develop for the ID01 beamline. If you want to use this code for other ESRF bealines, please contact the author! renzhe@ihep.ac.cn')
        self.load_mask(pathmask)
        return

    def change_image_infor(self, path_image_folder):
        """
        Change the image folder for the edf files.

        Parameters
        ----------
        path_image_folder : str
            The new path for the image folder.

        Raises
        ------
        IOError
            If the path does not exist, raise io error.

        Returns
        -------
        None.

        """
        self.path_image_folder = path_image_folder
        if not os.path.exists(self.path_image_folder):
            raise IOError('The image folder for %s images %s does not exist, please check the path again!' % (self.detector, self.path_image_folder))
        if self.start_time < datetime.datetime(2016, 10, 1):
            self.path_img = os.path.join(self.path_image_folder, self.prefix + "_%04d.edf")
        else:
            self.path_img = os.path.join(self.path_image_folder, self.prefix + "_%05d.edf.gz")
        return

    def load_single_image(self, img_index, correction_mode='constant'):
        """
        Read a single image stored in edf or edf.gz format.

        Parameters
        ----------
        img_index : int
            The index of the single image in the scan.
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
            The image of the pilatus detector.

        """
        assert img_index < self.npoints, \
            'The image number wanted is larger than the total image number in the scan!'
        img_index = int(img_index)

        img_index = int(self.get_scan_data('mpx4inr')[img_index])
        pathimg = self.path_img % img_index
        if pathimg[-4:] == '.edf':
            with open(pathimg, 'r') as f:
                f.seek(1024)
                image = np.fromfile(f, dtype=np.int32).astype(float)
                image = image.reshape(self.detector_size)
        elif pathimg[-3:] == '.gz':
            with gzip.open(pathimg, 'rb') as f:
                f.seek(1024)
                image = np.frombuffer(f.read(), dtype=np.int32).astype(float)
                image = image.reshape(self.detector_size)
        image = self.image_mask_correction(image, correction_mode=correction_mode)
        return image

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
        pch = self.image_find_peak_position(roi=roi, cut_width=cut_width)
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
        mu = self.get_motor_pos('mu')
        energy = self.get_motor_pos('nrj')

        motor_position = np.array([eta, delta, chi, phi, nu, mu, energy], dtype=float)
        pixel_position = np.array([pch[1], pch[2]])
        return pixel_position, motor_position
