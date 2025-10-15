# -*- coding: utf-8 -*-
"""
Read and treat the nexus files for the scans recorded at ESRF.
Created on Wed Aug 27 16:10:27 2025

@author: Ren Zhe
@email: renzhe@ihep.ac.cn
"""

import os
import numpy as np
import h5py
import hdf5plugin
from datetime import datetime
import pandas as pd

from ..general_scan import GeneralScanStructure


class ESRFScanImporter(GeneralScanStructure):
    """
    Read and write fio files for the scan recorded at ESRF beamlines. It is a child class of general scan structures.

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

    def __init__(self, beamline, path, beamtimeID, sample_name, experimental_method, scan_num, pathsave='', creat_save_folder=True):
        super().__init__(beamline, path, sample_name, scan_num, pathsave, creat_save_folder)
        self.add_section_func('Scan Information', self.load_scan_infor, self.write_scan_infor)
        self.add_section_func('Motor Positions', self.load_motor_pos, self.write_motor_pos)
        self.add_section_func('Scan Data', self.load_scan_data, self.write_scan_data)

        self.beamtimeID = beamtimeID
        self.experimental_method = experimental_method
        self.path = os.path.join(path, sample_name)
        # Try to locate the fio file, first look at the folder to save the results, then try to look at the folder in the raw data.
        if os.path.exists(os.path.join(self.path, r"%s_%s.h5" % (beamtimeID, sample_name))):
            self.pathh5 = os.path.join(self.path, r"%s_%s.h5" % (beamtimeID, sample_name))
        else:
            raise IOError('Could not find the scan files please check the path, sample name, and the scan number again!')
        self.add_header_infor('beamtimeID')
        self.add_header_infor('experimental_method')
        self.add_header_infor('pathh5')

        if os.path.exists(self.save_infor_path):
            self.load_scan()
        else:
            self.read_nexus()
        return

    def read_nexus(self):
        """
        Read the h5 files from the new ID01 beamline, load the scan information.

        Returns
        -------
        None.

        """
        with h5py.File(self.pathh5, 'r') as scanfile:
            if not '%s_%s_%d.1' % (self.sample_name, self.experimental_method, self.scan) in scanfile:
                raise KeyError('The scan does not exist! Please check the scan number again!')
            aimed_group = scanfile['%s_%s_%d.1' % (self.sample_name, self.experimental_method, self.scan)]
            # Read basic scan information
            self.command = aimed_group['title'][()].decode('UTF-8')
            self.start_time = aimed_group['start_time'][()].decode('UTF-8')
            self.start_time = datetime.fromisoformat(self.start_time)
            self.end_time = aimed_group['end_time'][()].decode('UTF-8')
            self.end_time = datetime.fromisoformat(self.end_time)
            self.end_reason = aimed_group['end_reason'][()].decode('UTF-8')
            self.add_scan_infor('end_reason')
            self.add_scan_section('Scan Information')
            # Read motor positions
            self.motor_position = {}
            positioners = aimed_group['instrument/positioners_start']
            for motor_name in positioners.keys():
                self.motor_position[motor_name] = positioners[motor_name][()]
            self.add_scan_section('Motor Positions')
            # Read scan data
            counters = []
            data = aimed_group['measurement']
            for counter in data.keys():
                if data[counter].ndim == 1:
                    counters.append(counter)

            self.npoints = data[counter].shape[0]
            scan_data = np.zeros((self.npoints, len(counters)))
            for i in range(len(counters)):
                counter = counters[i]
                scan_data[:, i] = np.array(data[counter])
            self.scan_infor = pd.DataFrame(scan_data, columns=counters)
            self.add_scan_section('Scan Data')
        return
