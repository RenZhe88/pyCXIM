# -*- coding: utf-8 -*-
"""
Read and treat the spec files recorded at 1w1a beamline BSRF.

Created on Mon Nov 27 21:53:27 2023

@author: ren zhe
@email: renzhe@ihep.ac.cn
"""

import ast
import datetime
from io import StringIO
import numpy as np
import os
import pandas as pd
import re

from ..general_scan import GeneralScanStructure


class HEPSScanImporter(GeneralScanStructure):
    """
    Read and write the spec files generated at BSRF, beamlines.

    Parameters
    ----------
    beamline : str
        The name of the beamline. Now only '1w1a' is supported.
    path : str
        The path for the raw file folder.
    sample_name : str
        The name of the sample defined by the spec_newfile name in the system.
    scan : int
        The scan number.
    pathsave : str, optional
        The folder to save the results, if not given no results will be saved. The default is ''.
    creat_save_folder : bool, optional
        Whether the save folder should be created. The default is True.

    Raises
    ------
    KeyError
        Now the code only support 1w1a beamline, if other beamlines are selected, then KeyError is reportted.

    Returns
    -------
    None.

    """

    def __init__(self, beamline, path, sample_name, scan, pathsave='', creat_save_folder=True):
        super().__init__(beamline, path, sample_name, scan, pathsave, creat_save_folder)
        self.add_section_func('Scan Information', self.load_scan_infor, self.write_scan_infor)
        self.add_section_func('Name conversion', self.load_name_converter, self.write_name_converter)
        self.add_section_func('Motor Positions', self.load_motor_pos, self.write_motor_pos)
        self.add_section_func('Scan Data', self.load_scan_data, self.write_scan_data)

        # Try to locate the fio file, first look at the folder to save the results, then try to look at the folder in the raw data.
        if (beamline == 'id05_4c' or beamline == 'id05_6c'):
            self.path = path
            self.pathspec = os.path.join(self.path, r"%s.spec" % sample_name)
        else:
            raise KeyError('Now this code is developed for only 1w1a beamline! Please contact the author if you want to use data from other beamlines! email: renzhe@ihep.ac.cn')

        if os.path.exists(self.save_infor_path):
            self.load_scan()
        else:
            self.spec_reader()
        return

    def spec_reader(self):
        """
        Load the spec file information of corresponding scan into the memory.

        Returns
        -------
        None.

        """
        pattern0 = r'^#S \d+  .*$'
        pattern1 = r'^#S (\d+)  .*$'
        pattern2 = r'^#S \d+  (.*)$'
        pattern3 = r'^#O\d+ (.+)$'
        pattern4 = r'^#o\d+ (.+)$'
        pattern5 = r'^#P\d+ (.+)$'
        pattern6 = r'^#L (.+)$'
        pattern7 = r'^#L .*?\n(.*?)(?=\n#|\Z)'
        pattern8 = r'^#G4 (.+)$'
        pattern9 = r'^#D (\w+ \w+ \d+ [\d|\:]+ \d+)'

        with open(self.pathspec, 'r') as specfile:
            spectext = specfile.read()

        header = re.split(pattern0, spectext, flags=re.M)[0]
        scan_num_list = re.findall(pattern1, spectext, flags=re.M)
        scan_command_list = re.findall(pattern2, spectext, flags=re.M)
        scan_text_list = re.split(pattern0, spectext, flags=re.M)[1:]

        full_motor_name_lines = re.findall(pattern3, header, flags=re.M)
        short_motor_name_lines = re.findall(pattern4, header, flags=re.M)
        full_motor_name_list = []
        short_motor_name_list = []
        for i in range(len(short_motor_name_lines)):
            full_motor_name_list += full_motor_name_lines[i].split()
            short_motor_name_list += short_motor_name_lines[i].split()

        self.motor_name_full_to_short = dict(zip(full_motor_name_list, short_motor_name_list))
        self.motor_name_short_to_full = dict(zip(short_motor_name_list, full_motor_name_list))
        self.add_scan_section('Name conversion')

        scan_idx = scan_num_list.index(str(self.scan))
        self.command = scan_command_list[scan_idx]
        scantext = scan_text_list[scan_idx]
        self.extract_roi_infor(scantext)
        self.add_scan_section('Scan Information')

        motor_value_lines = re.findall(pattern5, scantext, flags=re.M)
        motor_value_list = []
        for i in range(len(motor_value_lines)):
            motor_value_list += motor_value_lines[i].split()
        motor_value_list = list(map(float, motor_value_list))
        self.motor_position = dict(zip(short_motor_name_list, motor_value_list))

        time_matches = re.findall(pattern9, scantext, flags=re.M)
        if time_matches:
            self.start_time = datetime.datetime.strptime(time_matches[0], '%a %b %d %H:%M:%S %Y')

        wavelength = float(re.findall(pattern8, scantext, flags=re.M)[0].split()[3])
        hc = 1.23984 * 10000.0
        energy = hc / wavelength
        self.add_motor_pos('energy', energy, 'Energy')
        self.add_scan_section('Motor Positions')

        old_counters = re.findall(pattern6, scantext, flags=re.M)[0].split()
        counters = []
        for element in old_counters:
            if element.isdigit():
                counters[-1] += element
            else:
                counters += [element] 
        scan_data = np.loadtxt(StringIO(re.findall(pattern7, scantext, flags=re.M | re.S)[0]))
        self.scan_infor = pd.DataFrame(scan_data, columns=counters)
        self.npoints = scan_data.shape[0]
        self.add_scan_section('Scan Data')
        return

    def extract_roi_infor(self, scan_text):
        """
        Extracting the roi information used in the experiment

        Parameters
        ----------
        scan_text : str
            Information of the corresponding scan in text format.

        Returns
        -------
        None.

        """
        roi_blocks = re.findall(r'<group name="(roi\d+)".*?</group>', scan_text, re.DOTALL)
        
        self.roi_dict = {}
        
        for roi_name in roi_blocks:
            block = re.search(fr'<group name="{roi_name}".*?</group>', scan_text, re.DOTALL).group()
        
            min_x = re.search(r'<dataset name="min_x".*?>(\d+)</dataset>', block).group(1)
            min_y = re.search(r'<dataset name="min_y".*?>(\d+)</dataset>', block).group(1)
            size_x = re.search(r'<dataset name="size_x".*?>(\d+)</dataset>', block).group(1)
            size_y = re.search(r'<dataset name="size_y".*?>(\d+)</dataset>', block).group(1)

            self.roi_dict[roi_name] = [int(min_y), int(min_y) + int(size_y), int(min_x), int(min_x) + int(size_x)]
        return

    def write_name_converter(self, section_name='Name conversion'):
        """
        Generate the name conversion section.

        Parameters
        ----------
        section_name : str, optional
            The name of the section. The default is 'Name conversion'.

        Returns
        -------
        list_of_lines : list
            The list of lines to be writen in the file.

        """
        list_of_lines = []
        list_of_lines.append(str(list(self.motor_name_short_to_full.keys())) + '\n')
        list_of_lines.append(str(list(self.motor_name_short_to_full.values())) + '\n')
        return list_of_lines

    def load_name_converter(self, text):
        """
        Load the name conversion section.

        Parameters
        ----------
        text : str
            The text of the section.

        Returns
        -------
        None.

        """
        short_motor_name_list = ast.literal_eval(text.splitlines()[0])
        full_motor_name_list = ast.literal_eval(text.splitlines()[1])
        self.motor_name_short_to_full = dict(zip(short_motor_name_list, full_motor_name_list))
        self.motor_name_full_to_short = dict(zip(full_motor_name_list, short_motor_name_list))
        return

    def name_converter_short_to_full(self, short_motor_name):
        """
        Convert the short filename in the spec file to its corresponding fullnames.

        Parameters
        ----------
        short_motor_name : str
            The short name of the given motor.

        Returns
        -------
        full_motor_name : str
            The full name of the given motor.

        """
        if short_motor_name in self.motor_name_short_to_full.keys():
            full_motor_name = self.motor_name_short_to_full[short_motor_name]
        else:
            full_motor_name = short_motor_name
        return full_motor_name

    def name_converter_full_to_short(self, full_motor_name):
        """
        Convert the short filename in the full file to its corresponding shortnames.

        Parameters
        ----------
        full_motor_name : str
            The full name of the given motor.

        Returns
        -------
        short_motor_name : TYPE
            The short name of the given motor.

        """
        if full_motor_name in self.motor_name_full_to_short.keys():
            short_motor_name = self.motor_name_full_to_short[full_motor_name]
        else:
            short_motor_name = full_motor_name
        return short_motor_name

    def add_motor_pos(self, short_motor_name, position, full_motor_name=None):
        """
        Add motor position information in the spec file.

        Parameters
        ----------
        short_motor_name : str
            The name of the motors to be added.
        position : object
            The poisiotn of the aimed motor.
        full_motor_name : str, optional
            The full name of the motor. The default is None.

        Returns
        -------
        None.

        """
        self.motor_position[short_motor_name] = position
        if full_motor_name is None:
            full_motor_name = short_motor_name
        self.motor_name_full_to_short[full_motor_name] = short_motor_name
        self.motor_name_short_to_full[short_motor_name] = full_motor_name
        return

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
        except KeyError:
            print('counter %s does not exist!' % counter_name)
            nan_array = np.empty(self.scan_infor.shape[0])
            nan_array[:] = np.NaN
            return nan_array
    
    def get_imgsum(self, det_type='e4m'):
        """
        Get the sum fo the corresponding images.

        Parameters
        ----------
        det_type : str, optional
            The type of the detect chosen, can be 'e4m', 'pilatus'. The default is 'e4m'.

        Returns
        -------
        ndarray
            The sum of the corresponding detector images.

        """
        if det_type == 'e4m' or det_type == 'pilatus':
            self.path_eiger_imgsum = os.path.join(self.pathsave, '%s_scan%05d_%s_imgsum.npy' % (self.sample_name, self.scan, det_type))
            if os.path.exists(self.path_eiger_imgsum):
                return np.load(self.path_eiger_imgsum)
            else:
                raise RuntimeError('Could not find the integrated detector images')
