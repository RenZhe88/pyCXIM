# -*- coding: utf-8 -*-
"""
Defining a general scan structure to be the bases for reading the spec files
Created on Mon Nov 27 21:53:27 2023

@author: Ren Zhe
@email: renzhe@ihep.ac.cn
"""

import ast
import datetime
from io import StringIO
import numpy as np
import os
import pandas as pd
import re


class GeneralScanStructure(object):
    """
    A generalized scan structure, which should be the parent class of all the scan readers.

    Parameters
    ----------
    beamline : str
        The beamline, where the experiment is performed.
    path : str
        The path of the folder, where all the experiment data and spec files are saved.
    sample_name : str
        The sample name, which is given to the spec system.
    scan : int
        The scan number.
    pathsave : str, optional
        The path to save the general spec file. The default is ''.
    creat_save_folder : bool, optional
        The folder to save the results will be created, if the folder does not already exist. The default is True.

    Raises
    ------
    IOError
        If the folder of the spec file does not exist, raise IOError.

    Returns
    -------
    None.

    """

    def __init__(self, beamline, path, sample_name, scan, pathsave='', creat_save_folder=True):
        self.beamline = beamline
        self.sample_name = sample_name
        self.scan = scan
        if not os.path.exists(path):
            raise IOError("The scan folder %s does not exist, please check it again!" % path)
        self.path = path
        self.save_infor_path = ''

        if pathsave != '':
            assert os.path.exists(pathsave), \
                "The save folder %s does not exist, please check it again!" % pathsave
            self.pathsave = os.path.join(pathsave, '%s_%05d' % (sample_name, scan))
            if (not os.path.exists(self.pathsave)) and (not creat_save_folder):
                self.pathsave = ''
            elif not os.path.exists(self.pathsave):
                os.mkdir(self.pathsave)
                self.save_infor_path = os.path.join(self.pathsave, '%s_%05d.txt' % (self.sample_name, self.scan))
            else:
                self.save_infor_path = os.path.join(self.pathsave, '%s_%05d.txt' % (self.sample_name, self.scan))
        else:
            self.pathsave = ''

        self.header_infor = ['beamline', 'path', 'sample_name', 'scan']
        self.sections = []
        self.section_names = []
        self.section_writers = []
        self.section_loaders = []

        self.scan_infor_paras = ["command", "npoints", "start_time", "end_time"]

        self.scan_infor = pd.DataFrame()

        self.scan_types = {}
        self.add_command_description('ascan', ('scan_type', 'motor_name', 'start_pos', 'end_pos', 'step_num', 'exposure'))
        self.add_command_description('dscan', ('scan_type', 'motor_name', 'start_pos', 'end_pos', 'step_num', 'exposure'))
        self.add_command_description('a2scan', ('scan_type', 'motor1_name', 'motor1_start_pos', 'motor1_end_pos', 'motor2_name', 'motor2_start_pos', 'motor2_end_pos', 'step_num', 'exposure'))
        self.add_command_description('d2scan', ('scan_type', 'motor1_name', 'motor1_start_pos', 'motor1_end_pos', 'motor2_name', 'motor2_start_pos', 'motor2_end_pos', 'step_num', 'exposure'))
        self.add_command_description('mesh', ('scan_type', 'motor1_name', 'motor1_start_pos', 'motor1_end_pos', 'motor1_step_num', 'motor2_name', 'motor2_start_pos', 'motor2_end_pos', 'motor2_step_num', 'exposure'))
        self.add_command_description('dmesh', ('scan_type', 'motor1_name', 'motor1_start_pos', 'motor1_end_pos', 'motor1_step_num', 'motor2_name', 'motor2_start_pos', 'motor2_end_pos', 'motor2_step_num', 'exposure'))
        return

    def write_scan(self):
        """
        Write scan with general scan structures.

        The file will be a text file with different sections.
        Each section starts with the pattern '#########section name#########'.
        The section content will be writed by the function defined.
        The section ends with a new line.

        Raises
        ------
        IOError
            If the save folder has not been defined or does not exist, raise IOError.

        Returns
        -------
        None.

        """
        list_of_lines = []
        list_of_lines += self.write_header_infor()
        list_of_lines += ["\n"]

        sections_writers_dict = dict(zip(self.section_names, self.section_writers))
        for section_name in self.sections:
            if section_name in self.section_names:
                list_of_lines += ["#########%s#########\n" % section_name]
                list_of_lines += sections_writers_dict[section_name]()
                list_of_lines += ["\n"]
            else:
                print('The writing function for section %s is not defined, the section will not be written!!!' % section_name)

        if self.pathsave != '':
            with open(self.save_infor_path, 'w') as f:
                f.writelines(list_of_lines)
        else:
            raise IOError('The saving folder is not defined, please define it first before saveing the information file!')
        return

    def load_scan(self):
        """
        Load scan with general scan structures.

        The file will be a text file with different sections.
        The section name is defined by the pattern '#########section name#########'.
        Each section will be readed by the function defined.
        The loading funcitons should only have one parameter, which is used to receive the text of the section.

        Returns
        -------
        None.

        """
        sections_loads_dict = dict(zip(self.section_names, self.section_loaders))
        pattern0 = r'\#{9}([^\#\n]+)\#{9}\n'
        pattern1 = r'\#{9}[^\#\n]+\#{9}\n'

        with open(self.save_infor_path, 'r') as inforfile:
            text = inforfile.read()

        self.sections = re.findall(pattern0, text)
        # header = re.split(pattern1, text)[1:]
        section_texts_dict = dict(zip(self.sections, re.split(pattern1, text)[1:]))

        for section in self.sections:
            if section in self.section_names:
                section_text = section_texts_dict[section].rstrip()
                sections_loads_dict[section](section_text)
            else:
                print('section "%s" not found!' % section)
        return

    def add_section_func(self, section_name, section_load_func, section_write_func):
        """
        Adding reading or writing methods for a specific section.

        Parameters
        ----------
        section_name : str
            The name of the section.
        section_load_func : callable
            Function which is used to load the section, with signature fun(text).
        section_write_func : callable
            Function which is used to write the section, with signature fun().

        Raises
        ------
        TypeError
            If the section_load_func or the section_write_func is not callable functions, raise error.

        Returns
        -------
        None.

        """
        if not callable(section_load_func):
            raise TypeError('section_load_function must be a callable funciton!')
        if not callable(section_write_func):
            raise TypeError('section_write_function must be a callable funciton!')

        self.section_names.append(section_name)
        self.section_loaders.append(section_load_func)
        self.section_writers.append(section_write_func)
        return

    def add_scan_section(self, section_name):
        """
        Adding scan sections of the spec file.

        Parameters
        ----------
        section_name : str
            The name of the sections.

        Returns
        -------
        None.

        """
        if section_name not in self.sections:
            self.sections.append(section_name)
        return

    def attr_list_from_text(self, text):
        """
        Tranform the text content to the attributes of the class, and return their names as a list.

        E.g. start_time = Thu Nov 03 09:45:15 2022
        command = "dscan del 0.0 1.0 50 0.5"
        npoints = 51

        Parameters
        ----------
        text : str
            The text of the section.

        Returns
        -------
        paralist : list
            The list of the parameter names.

        """
        lines = text.splitlines()
        paralist = []
        for line in lines:
            if line.split(" ", 2)[0] in ["start_time", "end_time"]:
                setattr(self, line.split(" ", 2)[0], datetime.datetime.strptime(line.split(" ", 2)[-1].rstrip(), '%a %b %d %H:%M:%S %Y'))
            else:
                setattr(self, line.split(" ", 2)[0], ast.literal_eval(line.split(" ", 2)[-1].rstrip()))
            paralist.append(line.split(" ", 2)[0])
        return paralist

    def attr_list_to_text(self, paralist):
        """
        Tranform the attributes of the class contained in a list to the text file.

        Parameters
        ----------
        paralist : list
            The list of the parameter names of the attributes to be transformed.

        Returns
        -------
        list_of_lines : list
            The list of the lines of the section.

        """
        list_of_lines = []
        for parameter in paralist:
            if hasattr(self, parameter):
                paravalue = getattr(self, parameter)
                if isinstance(paravalue, datetime.datetime):
                    list_of_lines.append("%s = %s\n" % (parameter, paravalue.strftime('%a %b %d %H:%M:%S %Y')))
                elif isinstance(getattr(self, parameter), str):
                    if '\\' in repr(getattr(self, parameter)):
                        list_of_lines.append('%s = r"%s"\n' % (parameter, getattr(self, parameter)))
                    else:
                        list_of_lines.append('%s = "%s"\n' % (parameter, getattr(self, parameter)))
                else:
                    list_of_lines.append("%s = %s\n" % (parameter, getattr(self, parameter)))
        return list_of_lines

    def position_dict_from_text(self, text):
        """
        Transfrom text files to a dictionary.

        Used for reading motor positions.

        Parameters
        ----------
        text : str
            The text of section.

        Returns
        -------
        motor_position : dict
            The dictionary containing the name and values of the motors.

        """
        lines = text.splitlines()
        position_dict = {}
        for line in lines:
            if line != "\n":
                try:
                    position_dict[line.split(" ", 2)[0]] = ast.literal_eval(line.split(" ", 2)[-1])
                except (ValueError, SyntaxError):
                    position_dict[line.split(" ", 2)[0]] = line.split(" ", 2)[-1].rstrip()
        return position_dict

    def position_dict_to_text(self, position_dict):
        """
        Transfrom the dictionary to a list of lines.

        Used for writing the motor positions.

        Parameters
        ----------
        position_dict : dict
            The motor position dictionary.

        Returns
        -------
        list_of_lines : list
            The list of lines to be writen in the file.

        """
        list_of_lines = []

        for pos_name, pos_value in position_dict.items():
            if isinstance(pos_value, str):
                if '\\' in repr(pos_value):
                    list_of_lines.append('%s = r"%s"\n' % (pos_name, pos_value))
                else:
                    list_of_lines.append('%s = "%s"\n' % (pos_name, pos_value))
            else:
                list_of_lines.append("%s = %s\n" % (pos_name, pos_value))
        return list_of_lines

    def dataframe_from_text(self, text):
        """
        Transform the pandas dataframe to a list of lines.

        Parameters
        ----------
        text : str
            The text of section.

        Returns
        -------
        None.

        """
        text = text.rstrip()
        f = StringIO(text)
        header = f.readline().rstrip()
        header = re.split('\\s+', header)[1:]
        data = np.loadtxt(f)
        dataframe = pd.DataFrame(data, columns=header)
        return dataframe

    def dataframe_to_text(self, dataframe):
        """
        Generate the text of the scan data section.

        Returns
        -------
        list_of_lines : list
            The list of lines to be writen in the file.

        """
        list_of_lines = []
        data = dataframe.to_string(header=True, index=False).split('\n')
        for line in data:
            list_of_lines.append(line + '\n')
        return list_of_lines

    def write_header_infor(self):
        """
        Generate header text.

        The header text describes the basic informaiton of the scan.
        It is always saved in this type of information file.
        It is used to indicated which scan this new spec file is based on.

        Returns
        -------
        list_of_lines : list
            The list of lines to be writen in the file.

        """
        return self.attr_list_to_text(self.header_infor)

    def add_header_infor(self, parameter_name):
        """
        Add a attribute in this class to the headers.

        Parameters
        ----------
        parameter_name : str
            The name of the parameters.

        Returns
        -------
        None.

        """
        self.header_infor.append(parameter_name)
        return

    def load_scan_infor(self, text):
        """
        Load the scan information section.

        This section is a common section, which contains information about the scan command, the number of points, the start time and the end time.
        Several parameters could be missing in spec file.

        Parameters
        ----------
        text : str
            The text of section.

        Returns
        -------
        None.

        """
        self.scan_infor_paras = self.attr_list_from_text(text)
        return

    def write_scan_infor(self):
        """
        Generate the scan information section.

        Returns
        -------
        list_of_lines : list
            The list of lines to be writen in the file.

        """
        return self.attr_list_to_text(self.scan_infor_paras)

    def add_scan_infor(self, parameter_name):
        """
        Add an attribute to the scan information section.

        Parameters
        ----------
        parameter_name : str
            The name of the parameter.

        Returns
        -------
        None.

        """
        # setattr(self, parameter_name, parameter_value)
        if parameter_name not in self.scan_infor_paras:
            self.scan_infor_paras.append(parameter_name)
        return

    def load_motor_pos(self, text):
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
        self.motor_position = self.position_dict_from_text(text)
        return

    def write_motor_pos(self):
        """
        Generate the motor position section.

        Returns
        -------
        list_of_lines : list
            The list of lines to be writen in the file.

        """
        return self.position_dict_to_text(self.motor_position)

    def load_scan_data(self, text):
        """
        Load the scan data section.

        Parameters
        ----------
        text : str
            The text of section.

        Returns
        -------
        None.

        """
        self.scan_infor = self.dataframe_from_text(text)
        return

    def write_scan_data(self):
        """
        Generate the text of the scan data section.

        Returns
        -------
        list_of_lines : list
            The list of lines to be writen in the file.

        """
        return self.dataframe_to_text(self.scan_infor)

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

    def get_scan_type(self):
        """
        Get the scan type.

        Returns
        -------
        str
            The scan type information in the command.

        """
        return self.command.split()[0]

    def add_command_description(self, scan_type, command_description):
        """
        Add a scan type and its descriptions in the scan structure.

        e.g.
        self.add_command_description('dscan', ('scan_type', 'motor_name', 'start_pos', 'end_pos', 'step_num', 'exposure'))
        self.add_command_description('d2scan', ('scan_type', 'motor1_name', 'motor1_start_pos', 'motor1_end_pos', 'motor2_name', 'motor2_start_pos', 'motor2_end_pos', 'step_num', 'exposure'))
        self.add_command_description('dmesh', ('scan_type', 'motor1_name', 'motor1_start_pos', 'motor1_end_pos', 'motor1_step_num', 'motor2_name', 'motor2_start_pos', 'motor2_end_pos', 'motor2_step_num', 'exposure'))

        Parameters
        ----------
        scan_type : str
            The type of the scan, e.g. ascan, dscan.
        command_description : tuple
            The description of the command.

        Returns
        -------
        None.

        """
        self.scan_types[scan_type] = command_description
        return

    def get_command_infor(self):
        """
        Analysis the command, and return its information is the scan type is recognized.

        Raises
        ------
        RuntimeError
            If the scan type has not been previously added to the scan structure, raise RuntimeEorror.

        Returns
        -------
        command_infor : dict
            The informaiton of the command.

        """
        command = []
        for command_part in self.command.split():
            try:
                command.append(ast.literal_eval(command_part))
            except (ValueError, SyntaxError):
                command.append(command_part)
        if command[0] in self.scan_types.keys():
            command_infor = dict(zip(self.scan_types[command[0]], command))
            return command_infor
        else:
            raise RuntimeError('The scan type is not recognized, please contact the authors to add the scan type!')

    def get_scan_shape(self):
        """
        Get the shape of the real scan data.

        Returns
        -------
        tuple or ints
            The shape of the scan data extracted from the command.

        """
        scan_shape = []
        command_infor = self.get_command_infor()
        for key in command_infor.keys():
            if 'step_num' in key:
                scan_shape.append(command_infor[key] + 1)
        if len(scan_shape) == 0:
            print('Reading the scan shape is not sucessful!!!')
            return None
        elif len(scan_shape) == 1:
            return scan_shape[0]
        else:
            return tuple(scan_shape)

    def get_scan_motor(self):
        """
        Get the motor names of the scan.

        Returns
        -------
        tuple
            The names of the scanning motor.

        """
        scan_motor = []
        command_infor = self.get_command_infor()
        for key in command_infor.keys():
            if ('motor' in key) and ('name' in key):
                scan_motor.append(command_infor[key])
        if len(scan_motor) == 0:
            print('scan does not use any motor!!!')
            return None
        elif len(scan_motor) == 1:
            return scan_motor[0]
        else:
            return tuple(scan_motor)

    def get_num_points(self):
        """
        Get the number of points in the scan.

        Returns
        -------
        int
            The number of points in the scan.

        """
        return self.npoints

    def get_start_time(self):
        """
        Get the start time of the scan.

        Returns
        -------
        time : datetime
            The starting time of the scan.

        """
        if self.start_time is not None:
            return self.start_time
        else:
            raise LookupError('The start time of the scan does not exist!')

    def get_end_time(self):
        """
        Get the end time of the scan.

        Returns
        -------
        time : datetime
            The end time of the scan.

        """
        if self.end_time is not None:
            return self.end_time
        else:
            raise LookupError('The end time of the scan does not exist!')

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
