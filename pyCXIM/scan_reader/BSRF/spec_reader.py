# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:40:19 2023

@author: Lenovo
"""

import re
import os
import numpy as np
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt

class BSRFScanImporter:
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

        # Try to locate the fio file, first look at the folder to save the results, then try to look at the folder in the raw data.
        if beamline == '1W1A':
            self.path = path
            self.pathspec = os.path.join(self.path, r"%s.spec" % sample_name)

        self.spec_reader()
        return

    def spec_reader(self):
        pattern0 = r'#S \d+  .+\n'
        pattern1 = r'#S \d+  (.+)\n'
        pattern2 = r'#O\d (.+)\n'
        pattern3 = r'#o\d (.+)\n'
        pattern4 = r'#P\d (.+)\n'
        pattern5 = r'#L (.+)\n'
        pattern6 = r'#L .+\n([^#]+\n)(?:#C|\n)'

        specfile = open(self.pathspec, 'r')
        spectext = specfile.read()

        header = re.split(pattern0, spectext)[0]
        scan_command_list = re.findall(pattern1, spectext)
        scan_text_list = re.split(pattern0, spectext)[1:]

        full_motor_name_lines = re.findall(pattern2, header)
        short_motor_name_lines = re.findall(pattern3, header)
        full_motor_name_list = []
        short_motor_name_list = []
        for i in range(len(short_motor_name_lines)):
            full_motor_name_list += full_motor_name_lines[i].split()
            short_motor_name_list += short_motor_name_lines[i].split()

        self.motor_name_full_to_short = dict(zip(full_motor_name_list, short_motor_name_list))
        self.motor_name_short_to_full = dict(zip(short_motor_name_list, full_motor_name_list))

        self.command = scan_command_list[self.scan - 1]
        scantext = scan_text_list[self.scan - 1]

        motor_value_lines = re.findall(pattern4, scantext)
        motor_value_list = []
        for i in range(len(motor_value_lines)):
            motor_value_list += motor_value_lines[i].split()
        motor_value_list = list(map(float, motor_value_list))
        self.motor_position = dict(zip(short_motor_name_list, motor_value_list))

        counters = re.findall(pattern5, scantext)[0].split()
        scan_data = np.loadtxt(StringIO(re.findall(pattern6, scantext)[0]))
        self.scan_infor = pd.DataFrame(scan_data, columns=counters)
        self.npoints = scan_data.shape[0]
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

    def name_converter_short_to_full(self, motor_name):
        if motor_name in self.motor_name_short_to_full.keys():
            motor_name = self.motor_name_short_to_full[motor_name]
        return motor_name

    def name_converter_full_to_short(self, motor_name):
        if motor_name in self.motor_name_full_to_short.keys():
            motor_name = self.motor_name_full_to_short[motor_name]
        return motor_name

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
        if 'Data' not in self.section_names:
            self.section_names.append('Data')
            data_value = data_value[:, np.newaxis]
            self.scan_infor = pd.DataFrame(data_value, columns=[counter_name])
        else:
            assert len(data_value) == self.npoints, 'The scan counter to be added has different dimension as in the original scan file! Please check it again!'
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

    def get_num_points(self):
        """
        Get the number of points in the scan.

        Returns
        -------
        int
            The number of points in the scan.

        """
        return self.npoints

    def get_scan_shape(self):
        """
        Get the shape of the real scan data.

        Returns
        -------
        tuple of ints
            The shape of the scan data extracted from the command.

        """
        scan_type = self.get_scan_type()
        command = self.command.split()
        if scan_type == 'dscan' or scan_type == 'ascan':
            return (int(command[4]) + 1,)
        elif scan_type == 'd2scan' or scan_type == 'a2scan':
            return (int(command[7]) + 1,)
        elif scan_type == 'dmesh' or scan_type == 'mesh':
            return (int(command[4]) + 1, int(command[8]) + 1)
        elif scan_type == 'time_series':
            return (self.npoints,)
        else:
            raise RuntimeError('Unrecognized scan type!')

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
            counter_name = self.name_converter_short_to_full(counter_name)
            return np.array(self.scan_infor[counter_name])
        except:
            print('counter %s does not exist!' % counter_name)
            nan_array = np.empty(self.scan_infor.shape[0])
            nan_array[:] = np.NaN
            return nan_array


def test():
    beamline = '1W1A'
    path = r'F:\Work place 4\sample\XRD\Additional Task\20231020 Special request 1W1A'
    p10_newfile = r'lxd_59_1'
    scan_num = 22
    pathsave = r'F:\Work place 4\sample\XRD\Additional Task\20231020 Special request 1W1A'

    scan = BSRFScanImporter(beamline, path, p10_newfile, scan_num, pathsave=pathsave, creat_save_folder=True)
    print(scan.get_motor_pos('del'))
    return

if __name__ == '__main__':
    test()