# -*- coding: utf-8 -*-
"""
Description
Created on Thu May  4 17:23:02 2023

@author: renzhe
"""

import os
import sys
import numpy as np
from .p10_scan_reader import P10Scan


class P10FluoScan(P10Scan):
    """
    The read and treat the scans with eiger4M detector.

    Parameters
    ----------
    path : str
        The path for the raw file folder.
    p10_file : str
        The name of the sample defined by the p10_newfile name in the system.
    scan : int
        The scan number.
    pathsave : str, optional
        The folder to save the results, if not given no results will be saved. The default is ''.
    creat_save_folder : boolen, optional
        Whether the save folder should be created. The default is True.

    Returns
    -------
    None.

    """

    def __init__(self, path, p10_file, scan, pathsave='', creat_save_folder=True):
        super().__init__(path, p10_file, scan, pathsave, creat_save_folder)
        self.path_fluo_folder = os.path.join(self.path, "%s_%05d" % (p10_file, scan))
        self.path_fluo_spec = os.path.join(self.path_fluo_folder, "%s_%05d_mca_s%d.fio")
        assert os.path.exists(self.path_fluo_folder), 'The image folder for fluorescence spectrums %s does not exist, please check whether the spectrum saving option is enabled!' % (self.path_fluo_folder)

    def fluo_spec_reader(self, fluo_channels):
        """
        Sum up fluorescence channels at each positionin the scan.

        Parameters
        ----------
        fluo_channels : list
            Channels to be integrated. Channels should be given in the form of [channel1, channel2, channle3].
            Each channel must contain two integers, such as channel1 = [1000, 1500].

        Returns
        -------
        fluo_sum : ndarray
            The summed fluorescence spectrum.

        """

        fluo_channels = np.array(fluo_channels, dtype=int)
        assert fluo_channels.ndim == 2, 'Fluorescence channels should be described as two dimensional arrays! If only one roi is needed, then rois = [roi].'
        assert fluo_channels.shape[1] == 2, 'Each channels must be described with two integers!'
        num_of_fluo_rois = fluo_channels.shape[0]
        fluo_ar = np.zeros((self.npoints, num_of_fluo_rois))
        fluo_sum = np.zeros(2048)

        if 'Data' not in self.section_names:
            self.npoints = len(os.listdir(self.path_fluo_folder))

        for i in range(self.npoints):
            spec_file = self.path_fluo_spec % (self.p10_file, self.scan, i + 1)
            fluo_spec = np.array([])
            counters = []
            f = open(spec_file, 'r')
            markd = False
            for line in f:
                if line[0] == "!":
                    markd = False
                elif markd and (line[0:5] == " Col "):
                    counters.append(line.split()[2])
                elif markd:
                    value = np.fromstring(line.replace('None', 'nan'), dtype=float, sep=" ")
                    fluo_spec = np.append(fluo_spec, value, axis=0)
                if line[0:2] == "%d":
                    markd = True
            f.close()
            fluo_sum = fluo_sum + fluo_spec
            for j in range(num_of_fluo_rois):
                Channel_range = fluo_channels[j, :]
                fluo_ar[i, j] = np.append(fluo_ar, np.sum(fluo_spec[Channel_range[0]: Channel_range[1]]))
        for j in range(num_of_fluo_rois):
            self.add_scan_data('fluo_%02d' % (j + 5), fluo_ar[:, j])
        return fluo_sum
