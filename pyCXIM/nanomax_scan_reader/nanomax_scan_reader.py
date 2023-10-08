# -*- coding: utf-8 -*-
"""
Description
Created on Thu Jul  6 17:09:31 2023

@author: renzhe
"""
import os
import numpy as np
import h5py
import hdf5plugin
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
import sys

class NanoMaxScan:
    def __init__(self, path, sample_name, scan, pathsave='', creat_save_folder=True):
        self.sample_name = sample_name
        self.scan = scan
        self.path = os.path.join(path, sample_name)
        assert os.path.exists(path), "The scan folder %s does not exist, please check it again!" % path

        if pathsave != '':
            assert os.path.exists(pathsave), "The save folder %s does not exist, please check it again!" % pathsave
            self.pathsave = os.path.join(pathsave, '%s_%05d' % (sample_name, scan))
            if (not os.path.exists(self.pathsave)) and creat_save_folder:
                os.mkdir(self.pathsave)
        else:
            self.pathsave = pathsave

        # Try to locate the fio file, first look at the folder to save the results, then try to look at the folder in the raw data.
        if os.path.exists(os.path.join(self.path, r"%06d.h5" % (scan))):
            self.pathh5 = os.path.join(self.path, r"%06d.h5" % scan)
        else:
            assert False, 'Could not find the scan files please check the path, sample name, and the scan number again!'

        # self.path_merlin_imgsum = os.path.join(self.pathsave, '%s_scan%05d_%s_imgsum.npy' % (self.sample_name, self.scan, 'merlin'))

        scanfile = h5py.File(self.pathh5, 'r')
        self.command = scanfile['entry/description'][0].decode('UTF-8')
        self.motor_position_pre_scan = {}
        self.motor_position_post_scan = {}
        for parameter_name in scanfile['entry/snapshots/pre_scan'].keys():
            self.motor_position_pre_scan[parameter_name] = scanfile['entry/snapshots/pre_scan/%s' % parameter_name][0]
        for parameter_name in scanfile['entry/snapshots/post_scan'].keys():
            self.motor_position_post_scan[parameter_name] = scanfile['entry/snapshots/post_scan/%s' % parameter_name][0]
        scanfile.close()
        self.load_default_data()
        self.load_pseudo_data()
        self.load_alba2_data()
        return

    def save_hdf5(self):
        assert os.path.exists(self.pathsave), "The save folder %s does not exist, please check it again!" % self.pathsave
        self.pathsaveh5 = os.path.join(self.pathsave, r"%06d.h5" % self.scan)
        scanfile = h5py.File(self.pathsaveh5, 'w')
        main_group = scanfile.create_group('entry')
        main_group['description'] = self.command

        for parameter_name in self.motor_position_pre_scan.keys():
            scanfile['entry/snapshots/pre_scan/%s' % parameter_name] = self.motor_position_pre_scan[parameter_name]
        for parameter_name in self.motor_position_post_scan.keys():
            scanfile['entry/snapshots/post_scan/%s' % parameter_name] = self.motor_position_post_scan[parameter_name]

        if hasattr(self, 'alba2_infor'):
            scanfile.create_dataset('entry/measurement/alba2/1', data=self.get_scan_data('1', 'alba2'))
            scanfile.create_dataset('entry/measurement/alba2/2', data=self.get_scan_data('2', 'alba2'))
            scanfile.create_dataset('entry/measurement/alba2/3', data=self.get_scan_data('3', 'alba2'))
            scanfile.create_dataset('entry/measurement/alba2/4', data=self.get_scan_data('4', 'alba2'))
            scanfile.create_dataset('entry/measurement/alba2/t', data=self.get_scan_data('t', 'alba2'))
        if hasattr(self, 'pseudo_infor'):
            scanfile.create_dataset('entry/measurement/pseudo/x', data=self.get_scan_data('x', 'pseudo'))
            scanfile.create_dataset('entry/measurement/pseudo/y', data=self.get_scan_data('y', 'pseudo'))
            scanfile.create_dataset('entry/measurement/pseudo/z', data=self.get_scan_data('z', 'pseudo'))
            scanfile.create_dataset('entry/measurement/pseudo/analog_x', data=self.get_scan_data('analog_x', 'pseudo'))
            scanfile.create_dataset('entry/measurement/pseudo/analog_y', data=self.get_scan_data('analog_y', 'pseudo'))
            scanfile.create_dataset('entry/measurement/pseudo/analog_z', data=self.get_scan_data('analog_z', 'pseudo'))
        if hasattr(self, 'merlin_roi_pos') and hasattr(self, 'merlin_roi_infor'):
            scanfile.create_dataset('entry/measurement/merlin_roi/roi_position', data=self.merlin_roi_pos.to_numpy())
            scanfile['entry/measurement/merlin_roi/roi_position'].attrs['columns_names'] = list(self.merlin_roi_pos.columns)
            scanfile.create_dataset('entry/measurement/merlin_roi/roi_intensity', data=self.merlin_roi_infor.to_numpy())
            scanfile['entry/measurement/merlin_roi/roi_intensity'].attrs['column_names'] = list(self.merlin_roi_infor.columns)
        scanfile.close()
        return

    def load_default_data(self):
        scanfile = h5py.File(self.pathh5, 'r')
        counters = ('dt', 'ring_current')
        if self.get_scan_type() == 'npointflyscan':
            counters = counters + (self.get_scan_motor()[1], )
        elif type(self.get_scan_motor()) is tuple:
            counters = counters + self.get_scan_motor()
        else:
            counters = counters + (self.get_scan_motor(), )

        data_length = scanfile['entry/measurement/ring_current'][()].shape[0]
        scan_data = np.zeros((data_length, len(counters)))
        for i, counter in enumerate(counters):
            assert ('entry/measurement/%s' % counter) in scanfile, 'Counter %s does not exists, please check it again!' % counter
            scan_data[:, i] = scanfile['entry/measurement/%s' % counter][()]
        self.scan_infor = pd.DataFrame(scan_data, columns=counters)
        scanfile.close()
        return

    def load_pseudo_data(self):
        scanfile = h5py.File(self.pathh5, 'r')
        counters = ('analog_x', 'analog_y', 'analog_z', 'x', 'y', 'z')
        self.npoints = scanfile['entry/measurement/pseudo/x'].shape[0]
        pseudo_data = np.zeros((self.npoints, len(counters)))
        for i, counter in enumerate(counters):
            assert ('entry/measurement/pseudo/%s' % counter) in scanfile, 'Counter pseudo %s does not exists, please check it again!' % counter
            pseudo_data[:, i] = scanfile['entry/measurement/pseudo/%s' % counter][()]
        self.pseudo_infor = pd.DataFrame(pseudo_data, columns=counters)
        scanfile.close()
        return

    def load_alba2_data(self):
        scanfile = h5py.File(self.pathh5, 'r')
        counters = ('1', '2', '3', '4', 't')
        alba2_data = np.zeros((self.npoints, len(counters)))
        for i, counter in enumerate(counters):
            assert ('entry/measurement/alba2/%s' % counter) in scanfile, 'Counter alba2 %s does not exists, please check it again!' % counter
            alba2_data[:, i] = np.ravel(np.array(scanfile['entry/measurement/alba2/%s' % counter]))
        self.alba2_infor = pd.DataFrame(alba2_data, columns=counters)
        scanfile.close()
        return

    def get_command(self):
        """
        Get the command of the scan.

        Returns
        -------
        str
            the command of the scan.

        """
        return self.command

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
        Get the nanomax new_sample name.

        Returns
        -------
        str
            The nanomax new_sample name.

        """
        return self.sample_name

    def get_scan_data(self, counter_name, detector_name=None):
        scanfile = h5py.File(self.pathh5, 'r')
        if detector_name is None:
            assert ('entry/measurement/%s' % counter_name) in scanfile, 'Counter %s does not exists, please check it again!' % counter_name
            return np.array(self.scan_infor[counter_name])
        elif detector_name == 'pseudo':
            assert ('entry/measurement/pseudo/%s' % counter_name) in scanfile, 'Counter pseudo %s does not exists, please check it again!' % counter_name
            return np.array(self.pseudo_infor[counter_name])
        elif detector_name == 'alba2':
            assert ('entry/measurement/alba2/%s' % counter_name) in scanfile, 'Counter alba2 %s does not exists, please check it again!' % counter_name
            return np.array(self.alba2_infor[counter_name])
        elif detector_name == 'merlin_roi':
            assert hasattr(self, 'merlin_roi_infor'), 'The roi intensity for the merlin detecor has not been calculated! Please try to calculate it first!'
            assert counter_name in self.merlin_roi_infor.columns, 'The roi %s does not exist for the merlin detector.' % counter_name
            return np.array(self.merlin_roi_infor[counter_name])
        elif detector_name == 'eiger1m':
            assert hasattr(self, 'eiger1m_infor'), 'The roi intensity for the eiger1m detecor has not been calculated! Please try to calculate it first!'
            assert counter_name in self.eiger1m_infor.columns, 'The roi %s does not exist for the eiger1m detector.' % counter_name
            return np.array(self.eiger1m_infor[counter_name])
        else:
            raise KeyError('Unkown detector type!')

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

    def get_scan_motor(self):
        """
        Get the motor names of the scan.

        Returns
        -------
        object
            The value of the corresponding motor.

        """
        command = self.command.split()
        if command[0] == 'dscan' or command[0] == 'ascan':
            return command[1]
        elif command[0] == 'dmesh' or command[0] == 'mesh' or command[0] == 'npointflyscan':
            return command[1], command[5]
        elif command[0] == 'fermatscan':
            return command[1], command[4]
        else:
            print('Please check the scan type!')
            return ''

    def get_scan_shape(self):
        """
        Get the shape of the real scan data.

        Returns
        -------
        tuple of ints
            The shape of the scan data extracted from the command.

        """
        command = self.command.split()
        if command[0] == 'dscan' or command[0] == 'ascan':
            return (int(command[4]) + 1,)
        elif command[0] == 'dmesh' or command[0] == 'mesh':
            return (int(command[4]) + 1, int(command[8]) + 1)
        elif command[0] == 'npointflyscan':
            return (int(command[8]) + 1, int(command[4]) + 1)
        elif command[0] == 'fermatscan':
            return (self.npoints,)
        else:
            print('Please check the scan type!')
            return 0

    def get_command_infor(self):
        """
        Get the information of the command line.

        Returns
        -------
        command_infor : dict
            command line information in the dictionary form.

        """
        command = self.command.split()
        command_infor = {}
        if command[0] == 'dscan' or command[0] == 'ascan':
            command_infor['scan_type'] = command[0]
            command_infor['motor1_name'] = command[1]
            command_infor['motor1_start_pos'] = command[2]
            command_infor['motor1_end_pos'] = command[3]
            command_infor['motor1_step_num'] = command[4]
            command_infor['exposure'] = command[5]
            return command_infor
        elif command[0] == 'd2scan' or command[0] == 'a2scan':
            command_infor['scan_type'] = command[0]
            command_infor['motor1_name'] = command[1]
            command_infor['motor1_start_pos'] = command[2]
            command_infor['motor1_end_pos'] = command[3]
            command_infor['motor2_name'] = command[4]
            command_infor['motor2_start_pos'] = command[5]
            command_infor['motor2_end_pos'] = command[6]
            command_infor['motor1_step_num'] = command[7]
            command_infor['exposure'] = command[8]
            return command_infor
        elif command[0] == 'dmesh' or command[0] == 'mesh' or command[0] == 'npointflyscan':
            command_infor['scan_type'] = command[0]
            command_infor['motor1_name'] = command[1]
            command_infor['motor1_start_pos'] = command[2]
            command_infor['motor1_end_pos'] = command[3]
            command_infor['motor1_step_num'] = command[4]
            command_infor['motor2_name'] = command[5]
            command_infor['motor2_start_pos'] = command[6]
            command_infor['motor2_end_pos'] = command[7]
            command_infor['motor2_step_num'] = command[8]
            command_infor['exposure'] = command[9]
            return command_infor


    def get_num_points(self):
        """
        Get the number of points in the scan according to the command.

        Returns
        -------
        int
            The number of points in the scan.

        """
        return self.npoints

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

    def __str__(self):
        """
        Print the scan number, p10_newfile name and the command.

        Returns
        -------
        str
            The string containing the information of the scan.

        """
        return '%s_%05d: %s' % (self.sample_name, self.scan, self.get_command())

def test():       
    path = r'E:\Data2\XRD raw\20230623_PTO_STO_NanoMax\raw'
    nanomax_newfile = r'PTO_STO_DSO_28'
    scan_num = 288
    pathmask = r'E:\Work place 3\testprog\X-ray diffraction\Common functions\nanomax_merlin_mask.npy'
    pathsave = r'E:\Work place 3\sample\XRD\Test'

    scan = NanoMaxScan(path, nanomax_newfile, scan_num, pathsave)
    scan.merlin_load_mask(pathmask)
    roi = [100, 400, 100, 400]
    scan.merlin_roi_sum([roi])
    intensity = scan.get_scan_data('merlin_roi1', detector_name='merlin_roi')
    plt.plot(intensity)
    plt.show()
    return

if __name__ == '__main__':
    test()
