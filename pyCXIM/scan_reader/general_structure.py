# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 21:53:27 2023

@author: Lenovo
"""

import os
import numpy as np
import re
import pandas as pd

class GeneralScanStructure(object):
    def __init__(self, beamline, path, sample_name, scan, pathsave='', creat_save_folder=True):
        self.beamline = beamline
        self.sample_name = sample_name
        self.scan = scan
        assert os.path.exists(path), "The scan folder %s does not exist, please check it again!" % path

        if pathsave != '':
            assert os.path.exists(pathsave), \
                "The save folder %s does not exist, please check it again!" % self.pathsave
            self.pathsave = os.path.join(pathsave, '%s_%05d' % (sample_name, scan))
            if (not os.path.exists(self.pathsave)) and creat_save_folder:
                os.mkdir(self.pathsave)
        else:
            self.pathsave = pathsave

        self.command = ''
        self.motor_position = {}
        self.scan_infor = pd.DataFrame()
        self.npoints = 0
        return

    def __str__(self):
        """
        Print the scan number, sample_name name and the command.

        Returns
        -------
        str
            The string containing the information of the scan.

        """
        return '%s_%05d: %s' % (self.sample_name, self.scan, self.get_command())

    def __repr__(self):
        """
        Print the scan number, sample_name name and the command.

        Returns
        -------
        str
            The string containing the information of the scan.

        """
        return '%s_%05d: %s' % (self.sample_name, self.scan, self.get_command())

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
        self.motor_position[motor_name] = position
        return

    def add_scan_data(self, counter_name, data_value):
        """
        Add a scan counter and data in the fio file.

        Parameters
        ----------
        counter_name : str
            The name of the counter to be added.
        data_value : ndarray
            The value of the counters.

        Returns
        -------
        None.

        """
        assert len(data_value) == self.npoints, \
            'The scan counter to be added has different dimension as in the original scan file! Please check it again!'
        self.scan_infor[counter_name] = data_value
        return

    def get_pathsave(self):
        """
        Get the path to save the results.

        Returns
        -------
        str
            The path to save the results.

        """
        return self.pathsave

    def get_sample_name(self):
        """
        Get the sample name, which would be the name of p10_newfile or spec_newfile.

        Returns
        -------
        str
            The sample name.

        """
        return self.sample_name

    def get_command(self):
        """
        Get the command of the scan.

        Returns
        -------
        str
            the command of the scan.

        """
        return self.command

    def get_num_points(self):
        """
        Get the number of points in the scan.

        Returns
        -------
        int
            The number of points in the scan.

        """
        return self.npoints

    def get_motor_names(self):
        """
        Get the motor names in the scan.

        Returns
        -------
        list
            The motor names that exists in the scan.

        """
        return self.motor_position.keys()

    def get_counter_names(self):
        """
        Get the counter names in the scan.

        Returns
        -------
        list
            The counter names that exists in the scan.

        """
        return list(self.scan_infor.columns)

    def get_motor_pos(self, motor_name):
        """
        Get the motor positions in the fio file.

        Parameters
        ----------
        motor_name : str
            The name of the motor.

        Returns
        -------
        object
            The value of the motors. If the motor does not exist in the fio file, return None.

        """
        try:
            return self.motor_position[motor_name]
        except KeyError:
            print('motor %s does not exist in the fio file!' % motor_name)
            return None

    def get_motor_pos_list(self, motor_name_list):
        """
        Given a list of motor names, get their values.

        Parameters
        ----------
        motor_name_list : list
            List of the parameter names.

        Returns
        -------
        motor_pos_list : list
            List of the corresponding parameter values.

        """
        motor_pos_list = []
        for motor_name in motor_name_list:
            motor_pos_list.append(self.get_motor_pos(motor_name))
        return motor_pos_list

    def get_scan_data(self, counter_name):
        """
        Get the counter values in the scan.

        If the counter does not exist, return an array with NaN.

        Parameters
        ----------
        counter_name : str
            The name of the counter.

        Returns
        -------
        ndarray
            The values of the counter in the scan.

        """
        try:
            return np.array(self.scan_infor[counter_name])
        except KeyError:
            print('counter %s does not exist!' % counter_name)
            nan_array = np.empty(self.scan_infor.shape[0])
            nan_array[:] = np.NaN
            return nan_array

    def get_scan_type(self):
        """
        Get the scan type.

        Returns
        -------
        str
            The scan type information in the command.

        """
        command = self.command.split()
        return command[0]
