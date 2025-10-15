# -*- coding: utf-8 -*-
"""
Read and treat the scans recorded at ID01 beamline ESRF with image detector.
Created on Wed Aug 20 15:52:00 2025

@author: Ren Zhe
@email: renzhe@ihep.ac.cn
"""
import matplotlib.pyplot as plt
import numpy as np
import h5py
import hdf5plugin
import os

from .nexus_reader import ESRFScanImporter
from ..general_detector import DetectorMixin


class ESRFH5Importer(ESRFScanImporter, DetectorMixin):
    """
    Read and treat the scans with image detector. It is a child class of ESRFScanImporter.

    Parameters
    ----------
    beamline : str
        The name of the beamline. Please chose between 'p10' and 'p08'.
    path : str
        The path for the raw file folder.
    beamtimeID : str
        The beamtime id of this experiment.
    sample_name : str
        The name of the sample defined by the p10_newfile or spec_newfile name in the system.
    experimental_method : str
        The experimental method performed.
    scan_num : int
        The scan number.
    detector : str, optional
        The name of the detecter. Now only image 'mpx1x4' is supported. The default is 'mpx1x4'.
    pathsave : str, optional
        The folder to save the results, if not given no results will be saved. The default is ''.
    creat_save_folder : bool, optional
        Whether the save folder should be created. The default is True.

    Raises
    ------
    IOError
        If the code could not locate the fio file, then the IOError is reportted.
    KeyError
        Now the code only support beamline p10 or p08 if other beamlines are selected, then KeyError is reportted.

    Returns
    -------
    None.

    """

    def __init__(self, beamline, path, beamtimeID, sample_name, experimental_method, scan_num, detector='mpx1x4', pathsave='', creat_save_folder=True):
        super().__init__(beamline, path, beamtimeID, sample_name, experimental_method, scan_num, pathsave, creat_save_folder)

        self.detector = detector

        if beamline == 'id01':
            with h5py.File(self.pathh5, 'r') as scanfile:
                aimed_group = scanfile['%s_%s_%d.1/instrument/%s' % (self.sample_name, self.experimental_method, self.scan, self.detector)]
                dimi = int(aimed_group['dim_i'][()])
                dimj = int(aimed_group['dim_j'][()])
                self.detector_size = (dimi, dimj)
                self.pixel_size = aimed_group['x_pixel_size'][()] * 1.0e3

            self.path_imgsum = os.path.join(self.pathsave, '%s_scan%05d_%s_imgsum.npy' % (self.sample_name, self.scan, self.detector))
        else:
            raise KeyError('Now this code is only develop for the ID01 beamline. If you want to use this code for other ESRF bealines, please contact the author! renzhe@ihep.ac.cn')
        self.load_mask()
        return

    def load_mask(self):
        with h5py.File(self.pathh5, 'r') as scanfile:
            aimed_group = scanfile['%s_%s_%d.1/instrument/%s/pixel_mask' % (self.sample_name, self.experimental_method, self.scan, self.detector)]
            self.mask = np.array(aimed_group)
        if self.mask.shape != self.detector_size:
            raise ValueError('The mask size does not match with the detector size, please check it again!')

        self.img_correction = 1.0 - self.mask
        return

    def load_single_image(self, img_index, correction_mode='constant'):
        assert img_index < self.npoints, \
            'The image number wanted is larger than the total image number in the scan!'
        img_index = int(img_index)

        with h5py.File(self.pathh5, 'r') as scanfile:
            aimed_group = scanfile['%s_%s_%d.1/measurement/%s' % (self.sample_name, self.experimental_method, self.scan, self.detector)]
            image = np.array(aimed_group[img_index, :, :])
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
        delta = self.get_motor_pos('delta')
        chi = 0
        nu = self.get_motor_pos('nu')
        mu = self.get_motor_pos('mu')
        energy = self.get_motor_pos('nrj') * 1000.0

        motor_position = np.array([eta, delta, chi, phi, nu, mu, energy], dtype=float)
        pixel_position = np.array([pch[1], pch[2]])
        return pixel_position, motor_position
