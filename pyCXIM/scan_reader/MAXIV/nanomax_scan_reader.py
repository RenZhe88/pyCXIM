# -*- coding: utf-8 -*-
"""
Read and treat the h5 file for NanomMax scans.
Created on Thu Jul 6 17:09:31 2023

@author: Ren Zhe
@email: renzhe@ihep.ac.cn
"""
import os
import numpy as np
import h5py
import hdf5plugin

from ..general_scan import GeneralScanStructure


class NanoMaxScan(GeneralScanStructure):
    """
    Read and write h5 files for the scan recorded at NanoMax beamlines. It is a child class of general scan structures.

    Parameters
    ----------
    beamline : str
        The name of the beamline.
    path : str
        The path for the raw file folder.
    sample_name : str
        The name of the sample defined by the p10_newfile or spec_newfile name in the system.
    scan : int
        The scan number.
    pathsave : str, optional
        The folder to save the results, if not given no results will be saved. The default is ''.
    creat_save_folder : boolen, optional
        Whether the save folder should be created. The default is True.

    Raises
    ------
    IOError
        If the code could not locate the h5 file, then the IOError is reportted.

    Returns
    -------
    None.
    """

    def __init__(self, beamline, path, sample_name, scan, pathsave='', creat_save_folder=True):
        super().__init__(beamline, path, sample_name, scan, pathsave, creat_save_folder)
        self.add_section_func('Scan Information', self.load_scan_infor, self.write_scan_infor)
        self.add_section_func('Motor Position Prescan', self.load_motor_pos_prescan, self.write_motor_pos_prescan)
        self.add_section_func('Motor Position Postscan', self.load_motor_pos_postscan, self.write_motor_pos_postscan)
        self.add_section_func('Data', self.load_scan_data, self.write_scan_data)
        self.add_header_infor('pathh5')
        self.add_command_description('fermatscan', ('scan_type', 'motor1_name', 'motor1_start_pos', 'motor1_end_pos', 'motor2_name', 'motor2_start_pos', 'motor2_end_pos', 'step_size', 'exposure'))
        self.add_command_description('npointflyscan', ('scan_type', 'motor1_name', 'motor1_start_pos', 'motor1_end_pos', 'motor1_step_num', 'motor2_name', 'motor2_start_pos', 'motor2_end_pos', 'motor2_step_num', 'exposure', 'time_interval'))

        self.path = os.path.join(path, sample_name)
        # Try to locate the fio file, first look at the folder to save the results, then try to look at the folder in the raw data.
        if os.path.exists(os.path.join(self.path, r"%06d.h5" % (scan))):
            self.pathh5 = os.path.join(self.path, r"%06d.h5" % scan)
        else:
            raise IOError('Could not find the scan files please check the path, sample name, and the scan number again!')

        with h5py.File(self.pathh5, 'r') as scanfile:
            self.command = scanfile['entry/description'][0].decode('UTF-8')
            self.motor_position_pre_scan = {}
            self.motor_position_post_scan = {}
            for parameter_name in scanfile['entry/snapshots/pre_scan'].keys():
                self.motor_position_pre_scan[parameter_name] = scanfile['entry/snapshots/pre_scan/%s' % parameter_name][0]
            for parameter_name in scanfile['entry/snapshots/post_scan'].keys():
                self.motor_position_post_scan[parameter_name] = scanfile['entry/snapshots/post_scan/%s' % parameter_name][0]

        self.add_scan_section = 'Data'
        self.load_default_data()
        self.load_pseudo_data()
        self.load_alba2_data()
        return

    def load_default_data(self):
        """
        Load the default data section in the h5 file.

        Returns
        -------
        None.

        """
        if self.get_scan_type() != 'npointflyscan':
            with h5py.File(self.pathh5, 'r') as scanfile:
                counters = ('dt', 'ring_current')
                if type(self.get_scan_motor()) is tuple:
                    counters = counters + self.get_scan_motor()
                else:
                    counters = counters + (self.get_scan_motor(), )

                self.npoints = scanfile['entry/measurement/ring_current'][()].shape[0]
                for counter in counters:
                    assert ('entry/measurement/%s' % counter) in scanfile, 'Counter %s does not exists, please check it again!' % counter
                    self.scan_infor[counter] = np.array(scanfile['entry/measurement/%s' % counter])
        return

    def load_pseudo_data(self):
        """
        Load the data from pseudo detector in the h5 file.

        This data represents the position of the specimen meausred by the interferometer.

        Returns
        -------
        None.

        """
        with h5py.File(self.pathh5, 'r') as scanfile:
            counters = ('analog_x', 'analog_y', 'analog_z', 'x', 'y', 'z')
            self.npoints = scanfile['entry/measurement/pseudo/x'].shape[0]
            for counter in counters:
                assert ('entry/measurement/pseudo/%s' % counter) in scanfile, 'Counter pseudo %s does not exists, please check it again!' % counter
                self.scan_infor['pseudo_%s' % counter] = np.array(scanfile['entry/measurement/pseudo/%s' % counter])
        return

    def load_alba2_data(self):
        """
        Load the data from alba2 detector in the h5 file.

        The counter 1 repesents the intensity measured by the ion chamber in front of the specimen.

        Returns
        -------
        None.

        """
        with h5py.File(self.pathh5, 'r') as scanfile:
            counters = ('1', '2', '3', '4', 't')
            for counter in counters:
                assert ('entry/measurement/alba2/%s' % counter) in scanfile, 'Counter alba2 %s does not exists, please check it again!' % counter
                self.scan_infor['alba2_%s' % counter] = np.ravel(np.array(scanfile['entry/measurement/alba2/%s' % counter]))
        return

    def write_motor_pos_prescan(self):
        """
        Generate the motor position section.

        Returns
        -------
        list_of_lines : list
            The list of lines to be writen in the file.

        """
        return self.position_dict_to_text(self.motor_position_pre_scan)

    def load_motor_pos_prescan(self, text):
        """
        Load the section of motor positions.

        Parameters
        ----------
        text : str
            The text of section.

        Returns
        -------
        None.

        """
        self.motor_position_pre_scan = self.position_dict_from_text(text)
        return

    def write_motor_pos_postscan(self):
        """
        Generate the motor position section.

        Returns
        -------
        list_of_lines : list
            The list of lines to be writen in the file.

        """
        return self.position_dict_to_text(self.motor_position_post_scan)

    def load_motor_pos_postscan(self, text):
        """
        Load the section of motor positions.

        Parameters
        ----------
        text : str
            The text of section.

        Returns
        -------
        None.

        """
        self.motor_position_post_scan = self.position_dict_from_text(text)
        return

    def get_scan_data(self, counter_name, detector_name=None):
        """
        Get the counter values in the scan.

        Parameters
        ----------
        counter_name : str
            The name of the counter, e.g. roi1.
        detector_name : str, optional
            The name of the detector, which can be merlin, eiger1m, pseudo and alba2. The default is None.

        Returns
        -------
        ndarray
            The values of the counter in the scan.
        """
        if detector_name is None:
            counter_name = counter_name
        else:
            counter_name = '%s_%s' % (detector_name, counter_name)
        if counter_name in self.scan_infor.keys():
            return np.array(self.scan_infor[counter_name])
        else:
            print('counter %s does not exist!' % counter_name)
            nan_array = np.empty(self.scan_infor.shape[0])
            nan_array[:] = np.NaN
            return nan_array

    def get_motor_pos(self, motor_name, acquisition_stage='pre_scan'):
        """
        Get the motor positions in the scan file.

        Parameters
        ----------
        motor_name : str
            The name of the motor.
        acquisition_stage : str
            The acquisition time can be 'pre_scan' or 'post_scan'. The default is 'pre_scan'

        Returns
        -------
        object
            The value of the motors. If the motor does not exist in the fio file, return None.

        """
        try:
            if acquisition_stage == 'pre_scan':
                return self.motor_position_pre_scan[motor_name]
            elif acquisition_stage == 'post_scan':
                return self.motor_position_post_scan[motor_name]
        except KeyError:
            print('motor %s does not exist in the scan file!' % motor_name)
            return None

    def add_motor_pos(self, motor_name, position):
        """
        Add motor position information in the fio file.

        Parameters
        ----------
        motor_name : str
            The name of the motors to be added.
        position : object
            The poisiotn of the aimed motor.

        Returns
        -------
        None.

        """
        self.motor_position_pre_scan[motor_name] = position
        self.motor_position_post_scan[motor_name] = position
        return