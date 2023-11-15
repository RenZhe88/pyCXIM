# -*- coding: utf-8 -*-
"""
Reading, writing and modifying the information file for a typical BCDI scan and the phase retrieveal.
"""
import os
import ast
import pandas as pd


class InformationFileIO():
    """
    Generate the information file to store the parameters during the BCDI data treatment.

    Parameters
    ----------
    pathinfor : string
        The path for the information file.

    Returns
    -------
    None.

    """

    def __init__(self, pathinfor):
        self.pathinfor = pathinfor
        self.para_list = None
        return

    def infor_writer(self):
        """
        Save the information file as txt file in the aimed position.

        Returns
        -------
        None.

        """
        list_of_lines = []
        grouped_para = self.para_list.groupby(level='section', sort=False)
        for section, para_section in grouped_para:
            list_of_lines.append("#########%s#########\n" % section)
            for para_name, para in para_section.iterrows():
                if type(para['value']) == str:
                    if '\\' in repr(para['value']):
                        list_of_lines.append("%s = r'%s'\n" % (para_name[1], para['value']))
                    else:
                        list_of_lines.append("%s = '%s'\n" % (para_name[1], para['value']))
                else:
                    list_of_lines.append("%s = %s\n" % (para_name[1], para['value']))
            list_of_lines.append("\n")

        with open(self.pathinfor, 'w') as f:
            f.writelines(list_of_lines)
        return

    def infor_reader(self):
        """
        Read existing information file and load parameters into memory.

        Returns
        -------
        None.

        """
        index = []
        para_value = []
        if os.path.exists(self.pathinfor):
            f = open(self.pathinfor, 'r')
            for line in f:
                if line[0] == "#":
                    section = line.rstrip('\n').strip('#')
                elif line != "\n":
                    index.append((section, line.split(" ", 2)[0]))
                    try:
                        para_value.append(ast.literal_eval(line.split(" ", 2)[-1]))
                    except (ValueError, SyntaxError):
                        para_value.append(line.split(" ", 2)[-1].rstrip())
            f.close()
            index = pd.MultiIndex.from_tuples(index, names=['section', 'paraname'])
            self.para_list = pd.DataFrame({'value': para_value}, index=index)
        return

    def get_para_value(self, para_name, section=''):
        """
        Get the parameter value from the parameter list, the section name can be provided if the parameter is not unique.

        Parameters
        ----------
        para_name : string
            The name of the aimed parameter.
        section : string, optional
            The section name to specify the non-unique parameter. The default is ''.

        Returns
        -------
        object
            The value of aimed parameter if it exists in the information file.

        """
        if para_name not in self.para_list.index.get_level_values('paraname'):
            print('Could not find the desired parameter in the information file! Please check the parameter name %s again!' % para_name)
            return None
        elif list(self.para_list.index.get_level_values('paraname')).count(para_name) > 1 and section == '':
            print('More than one paramter with the same name! Please specify the sections of the desired parameter!')
            return None
        elif section == '':
            return self.para_list.xs(para_name, level='paraname').iloc[0, 0]
        else:
            return self.para_list.loc[(section, para_name), 'value']

    def add_para(self, para_name, section, para_value):
        """
        Add the parameter value to the parameter list.

        Parameters
        ----------
        para_name : string
            The name of the parameter.
        section : string
            The section of the parameter.
        para_value : object
            The value of the parameter.

        Returns
        -------
        None.

        """
        if self.para_list is None:
            index = pd.MultiIndex.from_tuples([(section, para_name)], names=['section', 'paraname'])
            self.para_list = pd.DataFrame({'value': 'to be filled'}, index=index)
        else:
            self.para_list.at[(section, para_name), 'value'] = 'to be filled'
        self.para_list.at[(section, para_name), 'value'] = para_value
        return

    def del_para_section(self, section):
        """
        Delete parameters from a section.

        Works only if the section name exists.

        Parameters
        ----------
        section : str
            The name of the section.

        Returns
        -------
        None.

        """
        if self.para_list is not None:
            if section in self.para_list.index.get_level_values('section'):
                self.para_list = self.para_list.drop(section, level='section')
        return

    def gen_empty_para_file(self, para_name_list, section):
        """
        Create an empty parameter file with the aimed parameter names, so that the users can fill it by themselves.

        Parameters
        ----------
        para_name_list : list
            The list of the parameter names.
        section : str
            The name of the section.

        Returns
        -------
        None.

        """
        for para_name in para_name_list:
            self.add_para(para_name, section, '')
        self.infor_writer()
        return

    def copy_para_file(self, other_para_file, para_name_list, section):
        """
        Copy the parameter from another parameter file.

        Parameters
        ----------
        other_para_file : Instants of BCDI information
            The information file, where the parameters should be copied from.
        para_name_list : list
            List of parameters to be copied.
        section : str
            The name of the section to be saved.

        Returns
        -------
        None.

        """
        other_para_file.infor_reader()
        for para_name in para_name_list:
            para_value = other_para_file.get_para_value(para_name)
            self.add_para(para_name, section, para_value)
        self.infor_writer()
        return
