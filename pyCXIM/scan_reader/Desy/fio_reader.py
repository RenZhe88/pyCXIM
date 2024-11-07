# -*- coding: utf-8 -*-
"""
Read and treat the fio files for the scans recorded at DESY.
Created on Mon Nov 27 21:53:27 2023

@author: Ren Zhe
@email: renzhe@ihep.ac.cn
"""

import ast
import datetime
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

from ..general_scan import GeneralScanStructure


class DesyScanImporter(GeneralScanStructure):
    """
    Read and write fio files for the scan recorded at Desy beamlines. It is a child class of general scan structures.

    Parameters
    ----------
    beamline : str
        The name of the beamline. Please chose between 'p10' and 'p08'.
    path : str
        The path for the raw file folder.
    sample_name : str
        The name of the sample defined by the p10_newfile or spec_newfile name in the system.
    scan : int
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

    def __init__(self, beamline, path, sample_name, scan, pathsave='', creat_save_folder=True):
        super().__init__(beamline, path, sample_name, scan, pathsave, creat_save_folder)
        self.add_section_func('Comments', self.load_scan_infor, self.write_scan_infor)
        self.add_section_func('Parameter', self.load_motor_pos, self.write_motor_pos)
        self.add_section_func('Data', self.load_scan_data, self.write_scan_data)
        self.add_header_infor('pathfio')
        self.add_command_description('time_series', ('scan_type', 'step_num', 'exposure'))

        # Try to locate the fio file, first look at the folder to save the results, then try to look at the folder in the raw data.
        if beamline == 'p10':
            if os.path.exists(os.path.join(self.pathsave, r"%s_%05d.fio" % (sample_name, scan))):
                self.path = os.path.join(path, r"%s_%05d" % (sample_name, scan))
                self.pathfio = os.path.join(self.pathsave, r"%s_%05d.fio" % (sample_name, scan))
            elif os.path.exists(os.path.join(path, r"%s_%05d" % (sample_name, scan), "%s_%05d.fio" % (sample_name, scan))):
                self.path = os.path.join(path, r'%s_%05d' % (sample_name, scan))
                self.pathfio = os.path.join(self.path, r"%s_%05d.fio" % (sample_name, scan))
            elif os.path.exists(os.path.join(path, sample_name, "%s_%05d.fio" % (sample_name, scan))):
                self.path = os.path.join(path, r'%s' % sample_name)
                self.pathfio = os.path.join(self.path, r"%s_%05d.fio" % (sample_name, scan))
            else:
                raise IOError('Could not find the fio files please check the beamline, p10_newfile name, and the scan number again!')
        elif beamline == 'p08':
            if os.path.exists(os.path.join(self.pathsave, r"%s_%05d.fio" % (sample_name, scan))):
                self.path = os.path.join(path, r"%s_%05d" % (sample_name, scan))
                self.pathfio = os.path.join(self.pathsave, r"%s_%05d.fio" % (sample_name, scan))
            elif os.path.exists(os.path.join(path, r"%s_%05d.fio" % (sample_name, scan))):
                self.path = os.path.join(path, r'%s_%05d' % (sample_name, scan))
                self.pathfio = os.path.join(path, r"%s_%05d.fio" % (sample_name, scan))
            else:
                raise IOError('Could not find the fio files please check the beamline, p08_newfile name, and the scan number again!')
        else:
            raise KeyError('Now the code only support two beamlines, please chose from p10 and p08! If you want to work with data from other beamlines, please contact the author! Email: renzhe@ihep.ac.cn')

        if os.path.exists(self.save_infor_path):
            self.load_scan()
        else:
            self.read_fio()
        return

    def read_fio(self):
        """
        Read the fio files, load the scan information.

        Returns
        -------
        None.

        """
        fiofile = open(self.pathfio, 'r')
        fiotext = fiofile.read()
        pattern0 = r'!\n! \w+\n!\n%\w\n'
        section_texts = re.split(pattern0, fiotext)[1:]
        pattern0 = r'!\n! (\w+)\n!\n%\w\n'
        self.sections = re.findall(pattern0, fiotext)

        for section_name, section_infor in list(zip(self.sections, section_texts)):
            if section_name == 'Comments':
                if section_infor != '':
                    self.command = section_infor.splitlines()[0]
                    pattern1 = 'user %suser Acquisition started at (.+)' % self.beamline
                    self.start_time = re.findall(pattern1, section_infor.splitlines()[1])[0]
                    self.start_time = datetime.datetime.strptime(self.start_time, '%a %b %d %H:%M:%S %Y')
                else:
                    self.command = 'time_series'
            elif section_name == 'Parameter':
                pattern2 = r'(\w+) = (.+)\n'
                self.motor_position = dict(re.findall(pattern2, section_infor))
                for parameter_name in self.motor_position:
                    try:
                        self.motor_position[parameter_name] = ast.literal_eval(self.motor_position[parameter_name])
                    except (ValueError, SyntaxError):
                        self.motor_position[parameter_name] = self.motor_position[parameter_name]
            elif section_name == 'Data':
                pattern3 = r' Col \d+ (\S+) \w+\n'
                counters = re.findall(pattern3, section_infor)
                section_infor = re.sub(pattern3, '', section_infor)
                pattern4 = r'! Acquisition ended at (.+)\n'
                self.end_time = re.findall(pattern4, section_infor)
                if self.end_time != []:
                    self.end_time = datetime.datetime.strptime(self.end_time[0], '%a %b %d %H:%M:%S %Y')
                else:
                    delattr(self, 'end_time')
                section_infor = re.sub(pattern4, '', section_infor)
                scan_data = np.loadtxt(StringIO(section_infor))
                self.scan_infor = pd.DataFrame(scan_data, columns=counters)
                self.npoints = scan_data.shape[0]

        fiofile.close()
        return

    def write_fio(self):
        """
        Write the fio file in the folder defined by pathsave.

        Returns
        -------
        None.

        """
        list_of_lines = []
        for section_name in self.sections:
            list_of_lines.append("!\n")
            list_of_lines.append("! %s\n" % section_name)
            list_of_lines.append("!\n")
            if section_name == 'Comments':
                list_of_lines.append("%c\n")
                list_of_lines.append(self.command + '\n')
                if hasattr(self, 'start_time'):
                    list_of_lines.append('user %suser Acquisition started at %s\n' % (self.beamline, self.start_time.strftime('%a %b %d %H:%M:%S %Y')))
            elif section_name == 'Parameter':
                list_of_lines.append("%p\n")
                for para_name in self.motor_position:
                    list_of_lines.append('%s = %s\n' % (para_name, str(self.motor_position[para_name])))
            elif section_name == 'Data':
                list_of_lines.append("%d\n")
                for i, data_name in enumerate(self.scan_infor):
                    list_of_lines.append(" Col %d %s DOUBLE\n" % (i + 1, data_name))
                data = self.scan_infor.to_string(header=False, index=False).split('\n')
                for line in data:
                    list_of_lines.append(line + '\n')
                if hasattr(self, 'end_time'):
                    list_of_lines.append('! Acquisition ended at %s\n' % self.end_time.strftime('%a %b %d %H:%M:%S %Y'))

        if self.pathsave != '':
            pathsave = os.path.join(self.pathsave, "%s_%05d.fio" % (self.sample_name, self.scan))
            with open(pathsave, 'w') as f:
                f.writelines(list_of_lines)
        else:
            print('The path for saving is not specified! Please specify this!')
        return

    def get_imgsum(self, det_type='e4m'):
        """
        Get the sum fo the corresponding images.

        Parameters
        ----------
        det_type : str, optional
            The type of the detect chosen, can be 'e4m', 'p300', 'e500'. The default is 'e4m'.

        Returns
        -------
        ndarray
            The sum of the corresponding detector images.

        """
        if det_type == 'e4m' or det_type == 'e500':
            self.path_eiger_imgsum = os.path.join(self.pathsave, '%s_scan%05d_%s_imgsum.npy' % (self.sample_name, self.scan, det_type))
            if os.path.exists(self.path_eiger_imgsum):
                return np.load(self.path_eiger_imgsum)
            else:
                raise RuntimeError('Could not find the aimed eiger detector images')

    def get_absorber(self):
        """
        Print the absorber used in the scan.

        The corresponding transmission can be found at https://henke.lbl.gov/optical_constants/filter2.html.

        Returns
        -------
        None.

        """
        assert self.beamline == 'p10', 'This function is only applied to beamline P10.'
        abs1z = int(round(self.get_motor_pos('abs1z')) + 4)
        abs2z = int(round(self.get_motor_pos('abs2z')) + 4)
        if self.get_start_time() > datetime.datetime(2020, 11, 6):
            abs1_Si = np.array([0, 2000, 500, 125, 0, 25, 100, 400, 1600])
            abs2_Si = np.array([0, 4000, 1000, 250, 0, 50, 200, 800, 3200])
            abs1_Ag = np.array([450, 0, 0, 0, 0, 0, 0, 0, 0])
            abs2_Ag = np.array([900, 0, 0, 0, 0, 0, 0, 0, 0])
        else:
            abs1_Si = np.array([0, 0, 0, 0, 0, 25, 100, 400, 1600])
            abs2_Si = np.array([0, 0, 0, 0, 0, 50, 200, 800, 3200])
            abs1_Ag = np.array([800, 200, 50, 12.5, 0, 0, 0, 0, 0])
            abs2_Ag = np.array([1600, 400, 100, 25, 0, 0, 0, 0, 0])
        total_Si_thickness = abs1_Si[abs1z] + abs2_Si[abs2z]
        total_Ag_thickness = abs1_Ag[abs1z] + abs2_Ag[abs2z]
        print('Absorber used is %.1f um thick Si and %.1f um thick Ag' % (total_Si_thickness, total_Ag_thickness))
        return total_Si_thickness, total_Ag_thickness

    def cal_diode_flux(self, counts, amplification=1.0e4, fmbenergy=None, thickness_Si=300.0):
        """
        Calculate the flux based on the Si diode counts in P10.

        Calculation performed according to paper:
            Determination of X-ray flux using silicon pin diodes.
            Holton, et al (2009).Journal of Synchrotron Radiation, Vol 16, Issue (2), page 143–151.

        Parameters
        ----------
        counts : Union[ndarray, float]
            The counter of the Si diode per second.
        amplification : float, optional
            The amplification of the diode output current. The default is 1e4.
        fmbenergy : float, optional
            The eneryg of the X-ray beam in eV.
            If None, the energy is imported from the fio file. The default is None.
        thickness_Si : float, optional
            The thickness of the Si diode in um. The default is 300.

        Returns
        -------
        flux : Union[ndarray, float]
            The flux calculated.

        """
        if fmbenergy is None:
            fmbenergy = self.get_motor_pos('fmbenergy')

        # Some constants
        # Electron charge [As]
        e_charge = 1.6022e-19
        # Required energy to create a silicon electron-hole pair [VAs]
        ehp_si = 3.66 * 1.6022e-19
        # silicon density [g/cm**3]
        silicon_rho = 2.33
        # Convert thicknesses to centimeter
        thickness_Si = thickness_Si * 1.0e-04

        # Convert energy to keV
        fmbenergy = fmbenergy * 1e-03

        # Photoelectric crosssections
        log10_silicon_pcs = 4.1580 - 2.2380 * np.log10(fmbenergy) - 0.4770 * (np.log10(fmbenergy) ** 2) + 0.0789 * (np.log10(fmbenergy) ** 3)
        silicon_pcs = 10 ** log10_silicon_pcs

        # Photoinduced current [mA]
        # voltage detected by the VFC [V]
        voltage_vfc = counts / 1000000 * 10
        # current created at silicon diode [A]
        photocurrent_si = voltage_vfc / amplification

        # X-ray flux
        flux = photocurrent_si * (ehp_si / (e_charge * fmbenergy * 1e03 * 1.6022e-19)) * (1 - np.exp(-silicon_pcs * thickness_Si * silicon_rho)) ** -1
        return flux

    def cal_cyberstar_flux(self, counts, pinhole_size=1.0, fmbenergy=None, distance=100, thickness=0.025, energy_corr=True):
        """
        Calculate the flux based on the cyberstar counts in P10.

        Parameters
        ----------
        counts : Union[ndarray, float]
            The counter of the cyberstar per second.
        pinhole_size : float, optional
            The openning size of the pinhole. The default is 1.0.
        fmbenergy : float, optional
            The energy of the X-ray beam in eV.
            If None, the energy is imported from fio file. The default is None.
        distance : float, optional
            Distance between Kapton foil and pinhole in mm. The default is 100.
        thickness : float, optional
            The . The default is 0.025.
        energy_corr : boolen, optional
            If true, the flux calculated for energy above 10 eV will be corrected with some experimental factor. The default is True.

        Returns
        -------
        flux : Union[ndarray, float]
            The flux calculated.

        """
        assert self.beamline == 'p10', 'This function is only applied to beamline P10.'
        if fmbenergy is None:
            fmbenergy = self.get_motor_pos('fmbenergy')

        fmbenergy = fmbenergy * 1.0e-3

        # Thickness correction to account for the incident angle of 45deg
        thickness = thickness / np.cos(np.deg2rad(45))

        # Kapton cross section (5-25keV)
        # these are materials constants
        a0 = 6.21567e-07
        a1 = -9.62581e-08
        a2 = 7.86926e-09
        a3 = -2.85194e-10
        a4 = 3.78401e-12

        b0 = -6.33669e-08
        b1 = 0.629331
        b2 = 3.25478
        b3 = 0.110262
        b4 = 2.72988

        b5 = 7.8529e-09
        b6 = 1.07227
        b7 = -294.699
        b8 = 0.0543647
        b9 = 18.5207

        C = 21175.44
        # some shortcuts
        A = a0 * fmbenergy ** 0 + a1 * fmbenergy ** 1 + a2 * fmbenergy ** 2 + a3 * fmbenergy ** 3 + a4 * fmbenergy ** 4
        B1 = b0 * np.cos((fmbenergy / b1) - b2) * np.exp(-(b3 * fmbenergy) ** b4)
        B2 = b5 * np.cos((fmbenergy / b6) - b7) * np.exp(-(b8 * fmbenergy) ** b9)
        KCS = C * (A + B1 + B2)

        # (half) cone opening angle
        Omega = np.arctan(1.0 / 2.0 * pinhole_size * 1.0e-03 / (distance * 1.0e-03))
        # solid angle of pinhole opening
        dOmega = 2 * np.pi * (1 - np.cos(Omega))

        efficiency = dOmega * KCS * thickness
        flux = counts / efficiency

        if energy_corr:
            if fmbenergy >= 10.0:
                flux = 1.152 * flux
            elif fmbenergy >= 5.0 and fmbenergy < 10.0:
                flux = (-0.1934 * fmbenergy + 3.045) * flux
        return flux

    def Gaussian_estimation(self, counter_name, sigma=1, normalize=True, plot=False):
        """
        Fit the diffraction intensity with gaussian function.

        Parameters
        ----------
        counter_name : str
            Name of the intensity counter.
        sigma : float, optional
            The starting sigma value for the fitting. The default is 1.
        normalize : bool, optional
            If true, the intensity will be normalized before fitting. The default is True.
        plot : bool, optional
            If true, the fitted diffraction intensiy will be plotted. The default is False.

        Returns
        -------
        amp : float
            The amplitude of the gaussian.
        cen : float
            The center position of the gaussian.
        FWHM : float
            The FWHM of the gaussian function.

        """
        motor = self.command.split()[1]
        motor_scan_value = self.get_scan_data(motor)
        counter_scan_value = self.get_scan_data(counter_name) / self.get_scan_data('curpetra')
        if normalize:
            counter_scan_value = (counter_scan_value - np.amin(counter_scan_value)) / ((np.amax(counter_scan_value) - np.amin(counter_scan_value)))
        p0 = [np.amax(counter_scan_value), motor_scan_value[np.argmax(counter_scan_value)], sigma]
        popt, pcov = curve_fit(self.gaussian, motor_scan_value, counter_scan_value, p0=p0)
        amp = popt[0]
        cen = popt[1]
        FWHM = 2.35482 * popt[2]
        if plot:
            plt.plot(motor_scan_value, counter_scan_value, 'o', label='scan' + str(self.scan))
            plt.plot(motor_scan_value, self.gaussian(motor_scan_value, popt[0], popt[1], popt[2]), color=plt.gca().lines[-1].get_color())
            plt.ylabel('Intensity (a.u.)')
            plt.xlabel(motor)
        return amp, cen, FWHM

    def knife_edge_estimation(self, counter_name, sigma=1.0, display_range=0.2, smooth=True, plot=False):
        """
        Fit the knife scan and estimate the knife edge position and the FWHM.

        Parameters
        ----------
        counter_name : str
            The name of the counter.
        sigma : float, optional
            The initial sigma for the fitting of the diffraction peak. The default is 1.0.
        display_range : float, optional
            The percentage of the scan to be displayed in the fitted images. The default is 0.2.
        smooth : bool, optional
            If true, the . The default is True.
        plot : bool, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        cen : float
            The position of the edge.
        FWHM : float
            The FWHM of the beam.

        """
        motor = self.get_scan_motor()
        motor_scan_value = np.array(self.scan_infor[motor])
        counter_scan_value = np.array(self.scan_infor[counter_name] / self.scan_infor['curpetra'])
        # Normaize data
        counter_scan_value = (counter_scan_value - np.amin(counter_scan_value)) / ((np.amax(counter_scan_value) - np.amin(counter_scan_value)))
        diff_motor = (motor_scan_value[:-1] + motor_scan_value[1:]) / 2
        diff_counter = np.diff(counter_scan_value)
        if smooth:
            diff_counter = savgol_filter(diff_counter, 5, 2)
        if 0.5 * np.abs(np.amin(diff_counter)) > np.abs(np.amax(diff_counter)):
            diff_counter = diff_counter * -1.0
        p0 = [np.amax(diff_counter), diff_motor[np.argmax(diff_counter)], sigma]
        popt, pcov = curve_fit(self.gaussian, diff_motor, diff_counter, p0=p0)
        cen = popt[1]
        FWHM = 2.35482 * popt[2]
        if plot:
            plt.subplot(1, 2, 1)
            scan_range = np.ptp(motor_scan_value)
            range_index = np.logical_and(motor_scan_value > cen - display_range * scan_range, motor_scan_value < cen + display_range * scan_range)
            plt.plot(motor_scan_value[range_index], counter_scan_value[range_index], "x-")
            plt.ylabel('%s (a.u.)' % counter_name)
            plt.xlabel(motor)
            plt.subplot(1, 2, 2)
            range_index = np.logical_and(diff_motor > cen - display_range * scan_range, diff_motor < cen + display_range * scan_range)
            plt.plot(diff_motor[range_index], diff_counter[range_index], 'o', label=str(self.get_motor_pos('_scan')))
            plt.plot(diff_motor[range_index], self.gaussian(diff_motor[range_index], popt[0], popt[1], popt[2]), color=plt.gca().lines[-1].get_color(), label='FWHM %0.2f' % FWHM)
            plt.ylabel('Intensity (a.u.)')
            plt.xlabel(motor)
        return cen, FWHM

    def tophat_estimation(self, counter_name, sigma=1.0, smooth=True, plot=False):
        """
        Fit the scan with tophat function and return its center.

        Used to find the center of the tungsten wire for the correction of the center of rotation.

        Parameters
        ----------
        counter_name : str
            The name of the counter to be used.
        sigma : float, optional
            The sigma value of the gaussian estimation. The default is 1.0.
        smooth : bool, optional
            If true, the diffraction signal will be smoothed before the fit. The default is True.
        plot : bool, optional
            If true, the positon of the two edges found for the tophat function will be plotted. The default is False.

        Returns
        -------
        float
            The center of the tophat function.

        """
        motor = self.get_scan_motor()
        motor_scan_value = self.get_scan_data(motor)
        counter_scan_value = self.get_scan_data(counter_name) / self.get_scan_data('curpetra')

        counter_scan_value = (counter_scan_value - np.amin(counter_scan_value)) / ((np.amax(counter_scan_value) - np.amin(counter_scan_value)))
        diff_motor = (motor_scan_value[:-1] + motor_scan_value[1:]) / 2
        diff_counter = np.diff(counter_scan_value)
        if smooth:
            diff_counter = savgol_filter(diff_counter, 5, 2)
        p0 = [np.amax(diff_counter), diff_motor[np.argmax(diff_counter)], sigma, np.amin(diff_counter), diff_motor[np.argmin(diff_counter)], sigma]
        popt, pcov = curve_fit(self.gaussian_two_peaks, diff_motor, diff_counter, p0=p0)
        edge1 = popt[1]
        edge2 = popt[4]
        if plot:
            plt.plot(motor_scan_value, counter_scan_value, "x-", label='scan %d' % self.scan)
            plt.vlines(edge1, 0, 1, "r")
            plt.vlines(edge2, 0, 1, "r")
            plt.ylabel('%s (a.u.)' % counter_name)
            plt.xlabel(motor)
        return (edge1 + edge2) / 2.0

    def gaussian(self, x, amp, cen, sigma):
        """
        Generate the gaussian function.

        Parameters
        ----------
        x : ndarray
            x poisitions of the gaussian function.
        amp : float
            The amplitude of the gaussian function.
        cen : float
            The center of the gaussian function.
        sigma : float
            The sigma value of the gaussian function.

        Returns
        -------
        ndarray
            Calculated Gaussian function.

        """
        return amp * np.exp(-(x - cen) ** 2.0 / (2.0 * sigma ** 2.0))

    def gaussian_two_peaks(self, x, amp1, cen1, sigma1, amp2, cen2, sigma2):
        """
        Generate two gaussian functions.

        Parameters
        ----------
        x : ndarray
            x poisitions of the gaussian function.
        amp1 : float
            The amplitude of the first gaussian function.
        cen1 : float
            The center of the first gaussian function.
        sigma1 : float
            The sigma value of the first gaussian function.
        amp2 : float
            The amplitude of the second gaussian function.
        cen2 : float
            The center of the second gaussian function.
        sigma2 : float
            The sigma value of the second gaussian function.

        Returns
        -------
        ndarray
            Calculated Gaussian function.

        """
        return amp1 * np.exp(-(x - cen1) ** 2.0 / (2.0 * sigma1 ** 2.0)) + amp2 * np.exp(-(x - cen2) ** 2.0 / (2.0 * sigma2 ** 2.0))

    def fio_to_spec(self, list_of_motors):
        """
        Convert the fio files to the spec format for a single scan, used in combination with spec writer function.

        Parameters
        ----------
        list_of_motors : list
            List of motors to be recorded in the spec file.

        Returns
        -------
        list_of_lines : TYPE
            DESCRIPTION.

        """
        list_of_lines = []
        list_of_lines.append('#S %d %s\n' % (self.scan, self.get_command()))
        list_of_lines.append('#D %s\n' % (self.get_start_time().strftime('%c')))
        list_of_lines.append('#T %s  (Seconds)\n' % self.get_command().split()[-1])
        for i in range(len(list_of_motors) // 8 + 1):
            s = '#P%d ' % i
            for j in range(len(list_of_motors[i * 8:(i + 1) * 8])):
                if self.get_motor_pos(list_of_motors[i * 8 + j]) is not None:
                    s += '%f ' % self.get_motor_pos(list_of_motors[i * 8 + j])
                else:
                    s += 'NaN  '
            s = s[:-1] + '\n'
            list_of_lines.append(s)
        for i in range(10):
            if self.get_motor_pos('e4m_roi%d' % i) is not None:
                roi = self.get_motor_pos('e4m_roi%d' % i)
                list_of_lines.append('#Ulima %s 0 %d %d %d %d\n' % ('e4m_roi%d' % i, roi[0], roi[1], roi[2], roi[3]))
        s = '#L '
        for counter_name in self.scan_infor.columns:
            s += '%s  ' % counter_name
        s = s[:-2] + '\n'
        list_of_lines .append(s)
        list_of_lines.append(self.scan_infor.to_string(header=False, index=False).replace('\n ', '\n'))
        list_of_lines.append('\n')
        return list_of_lines


def spec_writer(beamline, beamtimeID, path, sample_name, pathsave):
    """
    Generate the spec file for all the scans with the same sample_name name.

    Parameters
    ----------
    beamline : str
        The name of the beamline. Please chose between 'p10' and 'p08'.
    beamtimeID : int
        The beamtimeID of the scan.
    path : str
        The path for the raw file folder.
    sample_name : str
        The name of the sample defined by the sample_name name in the system.
    pathsave : str, optional
        The folder to save the results.

    Returns
    -------
    None.

    """
    assert os.path.exists(pathsave), "The folder for the spec file %s does not exist, please check it again!" % pathsave
    pathspec = os.path.join(pathsave, '%s.spec' % sample_name)
    list_of_motors = ['abs1z', 'abs2z', 'apiny', 'apinz', 'bpm_mon', 'bpmy', 'bpmz', 'chi', 'cryox', 'cryoy', 'cryoz', 'del', 'diffy', 'diffz', 'fmbenergy', 'fsz', 'gam', 'gslt1cx', 'gslt1cy', 'gslt1dx', 'gslt1dy', 'hexary', 'hexarz', 'hexax', 'hexay', 'hexaz', 'ipetra', 'mir1rz', 'mir1y', 'mir2rz', 'mir2y', 'mirz', 'mon1y', 'mu', 'om', 'ot1y', 'ot1z', 'phi', 'tomox', 'tomoy', 'undulator', 'undulatorgap', 'usamx', 'usamy', 'usamz', 'uslt1cx', 'uslt1cy', 'uslt1dx', 'uslt1dy', 'uslt2cx', 'uslt2cy', 'uslt2dx', 'uslt2dy', 'uty', 'utz']
    list_of_lines = []
    list_of_lines.append('#F %s' % (pathspec))
    list_of_lines.append('#E %d\n' % beamtimeID)
    list_of_lines.append('#D %s\n' % (datetime.datetime.now().strftime('%c')))
    list_of_lines.append('#C User = %s_user\n' % beamline)
    list_of_lines.append('\n')
    list_of_lines.append('\n')

    for i in range(len(list_of_motors) // 8 + 1):
        s = '#O%d ' % i
        for j in range(len(list_of_motors[i * 8:(i + 1) * 8])):
            s += '%s  ' % list_of_motors[i * 8 + j]
        s = s[:-2] + '\n'
        list_of_lines.append(s)
    list_of_lines.append('\n')
    list_of_lines.append('\n')

    for scan in range(1, 10000):
        if os.path.exists(os.path.join(pathsave, r"%s_%05d.fio" % (sample_name, scan))) or os.path.exists(os.path.join(path, "%s_%05d" % (sample_name, scan), "%s_%05d.fio" % (sample_name, scan))) or os.path.exists(os.path.join(path, sample_name, "%s_%05d.fio" % (sample_name, scan))):
            scan_data = DesyScanImporter(beamline, path, sample_name, scan, pathsave, creat_save_folder=False)
            if scan_data.get_command is not None:
                list_of_lines = list_of_lines + (scan_data.fio_to_spec(list_of_motors))
        else:
            break
    with open(pathspec, 'w', newline='\n') as f:
        f.writelines(list_of_lines)
    return
