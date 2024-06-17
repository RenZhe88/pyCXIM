# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
sys.path.append(r'E:\Work place 3\testprog\pyCXIM_master')
from pyCXIM.scan_reader.Desy.fio_reader import DesyScanImporter
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def start_alignment():
    path = r'F:\Raw Data\20221103_P10_BFO_PTO\raw'
    p10_newfile = r'align_02'
    pathsave = r'F:\Work place 4\Temp'
    align = alignment(path, p10_newfile, pathsave)

    # # ###align hexary####
    # align.scripts_align_hexary()
    # scan_num = [34, 42]
    # align.align_hexary(scan_num)

    # ####align hexarz####
    # align.scripts_align_hexarz()
    # scan_num = [45, 54]
    # align.align_hexarz(scan_num)

    # ####align phi center of rotation: Step1####
    # align.scripts_align_phi()
    # scan_num = [60, 62]
    # align.align_phi_rotation_step1(scan_num)

    # ####align phi center of rotation: Step2####
    # scan_num = [64]
    # align.align_phi_rotation_step2(scan_num)

    # ####align omega center of rotation####
    # hpz = 1864.5
    # align.scripts_align_om(hpz)
    # scan_num = [83, 88]
    # align.align_om_rotation(scan_num)

    # ####Energy vs Beamsize####
    # hpz_edge = [897, 1000.5, 1631.5]
    # hpy_edge = [559, 807.4, -244.5]
    # align.scripts_energy_vs_beamsize(hpz_edge, hpy_edge, 3)
    # scan_num = [115, 132]
    # align.energy_vs_beamsize(scan_num)

    # ####Measurement of the beamsize####
    # hpz_edge = [897, 1000.5, 1631.5]
    # hpy_edge = [559, 807.4, -244.5]
    # align.scripts_measure_beamsize(hpz_edge, hpy_edge)
    # scan_num = [135, 138]
    # align.measure_beamsize(scan_num)

#    ####hexary and the beamsize####
#    hpz_edge = [559, 807.4, -244.5]
#    align.scripts_hexary_vs_beamsize(hpy_edge, hexarz_start=-0.1,step_num=11, hexarz_step=0.02):
#    scan_num = [179, 200]
#    align.hexary_vs_beamsize(scan_num)

#    ####hexarz and the beamsize####
#    hpy_edge = [559, 807.4, -244.5]
#    align.scripts_hexarz_vs_beamsize(hpy_edge, hexarz_start=-0.1,step_num=11, hexarz_step=0.02):
#    scan_num = [264, 283]
#    align.hexarz_vs_beamsize(scan_num)

#    ####Measurement of the beamsize####
#    hpz_edge = [897, 1000.5, 1631.5]
#    hpy_edge = [559, 807.4, -244.5]
#    align.scripts_depth_of_focus(hpz_edge, hpy_edge, hpx_start=-500, hpx_end=500, hpx_step=100)
    return


class alignment():
    def __init__(self, path, p10_newfile, pathsave):
        self.path = path
        self.p10_newfile = p10_newfile
        self.pathsave = pathsave
        self.path_save_corrrection = os.path.join(pathsave, 'correction.txt')

    def gaussian(self, x, amp, cen, sigma):
        """
        Generate Gaussian function for the fitting proposes.

        Parameters
        ----------
        x : ndarray
            The x values for the gaussian function calculation.
        amp : float
            The amplitude of the gaussian function.
        cen : float
            The center position of the gaussian function.
        sigma : float
            The sigma value of the gaussian function.

        Returns
        -------
        ndarray
            The calculated gaussian function.

        """
        return amp * np.exp(-(x - cen) ** 2 / (2 * sigma ** 2))

    def scripts_align_hexary(self, hexary_start=-.2, step_num=9, hexary_step=0.05):
        """
        Generate the script for the hexary alignment.

        The alignment script name generated will be "sequence_01.txt".

        Parameters
        ----------
        hexary_start : float, optional
            The starting position of the hexary motor relative to the current position. The default is -.2.
        step_num : int, optional
            The total step num to be used for the hexaray motor. The default is 9.
        hexary_step : float, optional
            The step size for the hexary motor. The default is 0.05.

        Returns
        -------
        None.

        """
        pathsavetmp = os.path.join(self.pathsave, 'sequence_01.txt')
        text = '#The scripts to check the len rotation hexary\n\nsenv SignalCounter diffdio\n'
        rep_txt = '\n#delta Hexary = %.02f\numvr hexary %.2f\ndscan hexaz -0.4 0.4 80 .2\nmvsa peak 0\n'
        for i in range(step_num):
            if i == 0:
                text += rep_txt % (hexary_start, hexary_start)
            else:
                text += rep_txt % (hexary_start + i * hexary_step, hexary_step)
        with open(pathsavetmp, 'w') as f:
            f.write(text)
        return

    def align_hexary(self, scan_num_range):
        """
        Align the hexary motor.

        To finish the alignment, run the 'correction.txt' script generated.

        Parameters
        ----------
        scan_num_range : list
            The hexary alignment scan number range.

        Returns
        -------
        None.

        """
        scan_num_ar = range(scan_num_range[0], scan_num_range[1] + 1)
        Motor_name = 'hexary'
        Counter_name = 'diffdio'
        pos_ar = np.array([])
        Int_ar = np.array([])
        for scan_num in scan_num_ar:
            scan = DesyScanImporter('p10', self.path, self.p10_newfile, scan_num, creat_save_folder=False)
            pos_ar = np.append(pos_ar, scan.get_motor_pos(Motor_name))
            Int_ar = np.append(Int_ar, np.amax(scan.get_scan_data(Counter_name) / scan.get_scan_data('curpetra')))

        p0 = [np.argmax(Int_ar), pos_ar[np.argmax(Int_ar)], 0.1]
        popt, pcov = curve_fit(self.gaussian, pos_ar, Int_ar, p0=p0)
        plt.plot(pos_ar, Int_ar, "x")
        plt.plot(np.sort(pos_ar), self.gaussian(np.sort(pos_ar), popt[0], popt[1], popt[2]), "-", label='peak at %0.4f' % popt[1])
        plt.legend()
        plt.ylabel(Counter_name)
        plt.xlabel(Motor_name)
        plt.show()

        text = 'umv hexary %f\ndscan hexaz -.4 .4 80 .5\nmvsa peak 0\nwm hexary hexaz\n' % popt[1]
        with open(self.path_save_corrrection, 'w') as f:
            f.write(text)
        return

    def scripts_align_hexarz(self, hexarz_start=-0.2, step_num=9, hexarz_step=0.05):
        """
        Generate the script for the hexarz alignment.

        The alignment script name generated will be "sequence_02.txt".

        Parameters
        ----------
        hexarz_start : float, optional
            The starting position of the hexary motor relative to the current position. The default is -.2.
        step_num : int, optional
            The total step num to be used for the hexaray motor. The default is 9.
        hexarz_step : float, optional
            The step size for the hexary motor. The default is 0.05.

        Returns
        -------
        None.

        """
        pathsavetmp = os.path.join(self.pathsave, 'sequence_02.txt')
        text = '#The scripts to check the len rotation hexarz\n\nsenv SignalCounter diffdio\n'
        rep_txt = '\n#delta Hexarz = %.02f\numvr hexarz %.2f\ndscan hexay -0.4 0.4 80 .2\nmvsa peak 0\n'
        for i in range(step_num):
            if i == 0:
                text += rep_txt % (hexarz_start, hexarz_start)
            else:
                text += rep_txt % (hexarz_start + i * hexarz_step, hexarz_step)
        with open(pathsavetmp, 'w') as f:
            f.write(text)
        return

    def align_hexarz(self, scan_num_range):
        """
        Align the hexarz motor.

        To finish the alignment, run the 'correction.txt' script generated.

        Parameters
        ----------
        scan_num_range : list
            The hexarz alignment scan number range.

        Returns
        -------
        None.

        """
        scan_num_ar = range(scan_num_range[0], scan_num_range[1] + 1)
        Motor_name = 'hexarz'
        Counter_name = 'diffdio'
        pos_ar = np.array([])
        Int_ar = np.array([])
        for scan_num in scan_num_ar:
            scan = DesyScanImporter('p10', self.path, self.p10_newfile, scan_num, creat_save_folder=False)
            pos_ar = np.append(pos_ar, scan.get_motor_pos(Motor_name))
            Int_ar = np.append(Int_ar, np.amax(scan.get_scan_data(Counter_name) / scan.get_scan_data('curpetra')))

        p0 = [np.argmax(Int_ar), pos_ar[np.argmax(Int_ar)], 0.1]
        popt, pcov = curve_fit(self.gaussian, pos_ar, Int_ar, p0=p0)
        plt.plot(pos_ar, Int_ar, "x")
        plt.plot(np.sort(pos_ar), self.gaussian(np.sort(pos_ar), popt[0], popt[1], popt[2]), "-", label='peak at %0.4f' % popt[1])
        plt.legend()
        plt.ylabel(Counter_name)
        plt.xlabel(Motor_name)
        plt.show()

        text = 'umv hexarz %f\ndscan hexay -.4 .4 80 .5\nmvsa peak 0\nwm hexarz hexay\n' % popt[1]
        with open(self.path_save_corrrection, 'w') as f:
            f.write(text)
        return

    def scripts_align_phi(self):
        """
        Generate the script for the alignment of the phi motor.

        Motor used will be hpx, hpy, hpz. The script name would be 'sequence_03.txt'.

        Returns
        -------
        None.

        """
        pathsavetmp = os.path.join(self.pathsave, 'sequence_03.txt')
        text = '#The scripts to align phi the center of rotation\n\nsenv SignalCounter diffdio\n'
        rep_text = '\n#phi=%.2f\numvr phi %.2f\ndscan hpy -200 200 50 .3\nmvsa dip 0\ndscan hpy -20 20 160 .3\nmvsa dip 0\n'
        for phi in [0, 180]:
            text += rep_text % (phi, phi)
        with open(pathsavetmp, 'w') as f:
            f.write(text)
        return

    def align_phi_rotation_step1(self, scan_num_range, sigma=1.0):
        """
        Align the phi center of rotation with the absorption of the tungsten cross.

        To finish the alignment, run the 'correction.txt' script generated.

        Parameters
        ----------
        scan_num_range : list
            The two fine scans across the vertical tungsten wire at phi = 0 and 180.
        sigma : float, optional
            The estimated width of the direct beam. The default is 1.0.

        Returns
        -------
        None.

        """
        cen_ar = np.array([])
        for scan_num in scan_num_range:
            scan = DesyScanImporter('p10', self.path, self.p10_newfile, scan_num, creat_save_folder=False)
            cen = scan.tophat_estimation('diffdio', sigma=sigma, smooth=True, plot=True)
            cen_ar = np.append(cen_ar, cen)
        plt.legend()
        plt.show()
        print('umvr hpy %.1f, umvr diffy %.4f' % ((cen_ar[1] - cen_ar[0]) / 2.0, (cen_ar[0] - cen_ar[1]) / 2.0 / 1000.0))
        text = 'umv hpy %.1f\numvr diffy %.4f\numvr phi -90\ndscan hpx -200 200 100 .3\n' % ((cen_ar[1] + cen_ar[0]) / 2.0, (cen_ar[0] - cen_ar[1]) / 2.0 / 1000.0)
        with open(self.path_save_corrrection, 'w') as f:
            f.write(text)
        return

    def align_phi_rotation_step2(self, scan_num_range, sigma=1.0):
        """
        Align the tungsten wire to the center of the phi rotation at phi = 90.

        To finish the alignment, run the 'correction.txt' script generated.

        Parameters
        ----------
        scan_num_range : list
            The fine scan across the vertical tungsten wire at phi = 90.
        sigma : float, optional
            The estimated width of the direct beam. The default is 1.0.

        Returns
        -------
        None.

        """
        for scan_num in scan_num_range:
            scan = DesyScanImporter('p10', self.path, self.p10_newfile, scan_num, creat_save_folder=False)
            cen = scan.tophat_estimation('diffdio', sigma=sigma, plot=True)
        plt.legend()
        plt.show()
        text = 'umv hpx %.1f\ndscan hpx -20 20 160 .3\nmvsa dip 0\n' % cen
        print('umv hpx %.1f' % cen)
        with open(self.path_save_corrrection, 'w') as f:
            f.write(text)
        return

    def scripts_align_om(self, hpz, rel_hpz=100.0, rel_hpx=-100.0, rel_om=4.0):
        """
        Generate the script for the alignment of the omega center of rotation.

        Motor used will be hpx, hpy, hpz. The script name would be 'sequence_04.txt'.

        Parameters
        ----------
        hpz : float
            The initial center position of the horizontal wire.
        rel_hpz : float, optional
            Shift in hpz direction away from the assumed center of rotation. The default is 100.0.
        rel_hpx : float, optional
            Shift in hpx direction away from the assumed center of rotation. The default is -100.0.
        rel_om : float, optional
            The range of the omega for the alignment. The default is 4.0.

        Returns
        -------
        hpz : float
            The new hpz position after the correction..
        rel_hpz : float
            The .

        """
        pathsavetmp = os.path.join(self.pathsave, 'sequence_04.txt')
        text = '#The scripts to align om the center of rotation\n\nsenv SignalCounter diffdio\n\numvr diffz %.4f\numvr hpz %.1f\numvr hpx %.1f\n' % (-rel_hpz / 1000.0, rel_hpz, rel_hpx)
        rep_text = '\n#delta om = %.2f\numv om %.2f\ndscan hpz -200 200 50 .3\nmvsa dip 0\numvr hpz 12.5\ndscan hpz -10 10 80 .3\numv hpz %.1f\n'
        for om in [-rel_om, 0, rel_om]:
            text += rep_text % (om, om, hpz + rel_hpz)
        with open(pathsavetmp, 'w') as f:
            f.write(text)
        return hpz, rel_hpz

    def expected_shift(self, angle, alpha, radius, ycen):
        """
        Calculate the expected wire position for the fitting preposes.

        Parameters
        ----------
        angle : ndarray
            The omega angles.
        alpha : float
            The angle of the wire and the center of rotation between the z and x direction.
        radius : float
            The distance between the wire and the center of rotation.
        ycen : float
            The y position for the center of rotation.

        Returns
        -------
        ndarray
            The calculated shift.

        """
        return (np.sin(angle + alpha) - np.sin(alpha)) * radius / np.cos(angle) + ycen

    def align_om_rotation(self, scan_num_range, sigma=1.0):
        """
        Align the omega center of rotation with the tungsten cross.

        To finish the alignment, run the 'correction.txt' script generated.

        Parameters
        ----------
        scan_num_range : list
            The scan range of the omega alignment.
        sigma : float, optional
            The width of the direction beam. The default is 1.0.

        Returns
        -------
        None.

        """
        scan_num_ar = range(scan_num_range[0] + 1, scan_num_range[-1] + 1, 2)
        cen_ar = np.array([])
        angle_ar = np.array([])
        scan = DesyScanImporter('p10', self.path, self.p10_newfile, scan_num_range[0], creat_save_folder=False)
        hpz = scan.get_motor_pos('hpz')

        for scan_num in scan_num_ar:
            scan = DesyScanImporter('p10', self.path, self.p10_newfile, scan_num, creat_save_folder=False)
            angle = scan.get_motor_pos('om')
            angle_ar = np.append(angle_ar, angle)
            cen, FWHM = scan.knife_edge_estimation('diffdio', sigma=sigma, smooth=True, plot=True)
            cen_ar = np.append(cen_ar, cen)

        plt.show()
        if len(scan_num_ar) == 3:
            if np.abs(angle_ar[0]) == np.abs(angle_ar[2]) and int(np.abs(angle_ar[1])) == 0:
                delta_angle = np.radians(np.abs(angle_ar[2]))
                shift_x = (cen_ar[2] - cen_ar[0]) / (2.0 * np.sin(delta_angle))
                shift_y = (cen_ar[0] + cen_ar[2] - 2.0 * cen_ar[1]) / (2.0 * (np.cos(delta_angle) - 1))
                print(shift_x, shift_y)
            else:
                print('For three point correction, the rotation angle has to be symmetric against zero.')
        p0 = [np.pi / 4.0, 100, 0]
        popt, pcov = curve_fit(self.expected_shift, np.radians(angle_ar), cen_ar, p0=p0)
        shift_x = np.cos(popt[0]) * popt[1]
        shift_y = np.sin(popt[0]) * popt[1]
        print('umvr hpx %.1f, umvr hpz %.1f, umvr diffz %.4f' % (shift_x, shift_y, -shift_y / 1000.0))
        text = 'umv om 0\numvr hpx %.1f\numvr hpz %.1f\numvr diffz %.4f' % (shift_x, shift_y, -shift_y / 1000.0)
        with open(self.path_save_corrrection, 'w') as f:
            f.write(text)
        self.scripts_align_om(hpz + shift_y)
        return

    def scripts_energy_vs_beamsize(self, hpz_edge, hpy_edge, harmonic_order=1):
        """
        Generate the script for the measurement of energy vs beamsize scans.

        Motor used will be hpx, hpy, hpz. The script name would be 'sequence_05.txt'.

        Parameters
        ----------
        hpz_edge : list
            The position of the wire edge to measure the beam size in hpz direction in the order of hpx, hpy, hpz.
        hpy_edge : list
            The position of the wire edge to measure the beam size in hpy direction in the order of hpx, hpy, hpz.
        harmonic_order : int, optional
            The harmonic order of the undulator to be used. The default is 1.

        Returns
        -------
        None.

        """
        pathsavetmp = os.path.join(self.pathsave, 'sequence_05.txt')
        if harmonic_order == 3:
            energy_ar = np.array([-180.0, 60.0, 60.0, 30.0, 30.0, 30.0, 30.0, 60.0, 60.0])
            undulator_ar = energy_ar / 3.0
        elif harmonic_order == 1:
            energy_ar = np.array([-150.0, 50.0, 50.0, 25.0, 25.0, 25.0, 25.0, 50.0, 50.0])
            undulator_ar = energy_ar
        text = '#The scripts to check the beamsize with respect to the energy\n'
        rep_text = '\n#Energy = %.0f\numvr undulator %.0f\numvr fmbenergy %.0f\np10_pause 3\n\numv hpx %.1f\numv hpy %.1f\numv hpz %.1f\ndscan hpz -10 10 80 0.2\n\numv hpx %.1f\numv hpy %.1f\numv hpz %.1f\ndscan hpy -10 10 80 0.2\n'
        rel_energy = 0.0
        for energy, undulator in list(zip(energy_ar, undulator_ar)):
            rel_energy += energy
            text += rep_text % (rel_energy, undulator, energy, hpz_edge[0], hpz_edge[1], hpz_edge[2], hpy_edge[0], hpy_edge[1], hpy_edge[2])
        with open(pathsavetmp, 'w') as f:
            f.write(text)
        return

    def energy_vs_beamsize(self, scan_num_range, sigma=1.0):
        """
        Plot beamsize with respect to the energy.

        Parameters
        ----------
        scan_num_range : list
            The scan num range for the energy vs beamsize scans.
        sigma : float, optional
            The estimated size of the direct beam. The default is 1.0.

        Returns
        -------
        None.

        """
        hpz_scan_num_ar = range(scan_num_range[0], scan_num_range[1], 2)
        hpy_scan_num_ar = range(scan_num_range[0] + 1, scan_num_range[1] + 1, 2)
        energy_ar = np.array([])
        FWHM_ar = np.array([])
        for scan_num in hpz_scan_num_ar:
            scan = DesyScanImporter('p10', self.path, self.p10_newfile, scan_num, creat_save_folder=False)
            energy = scan.get_motor_pos('fmbenergy')
            energy_ar = np.append(energy_ar, energy)
            cen, FWHM = scan.knife_edge_estimation('diffdio', sigma=sigma, smooth=True, plot=False)
            FWHM_ar = np.append(FWHM_ar, FWHM)

        plt.plot(energy_ar, FWHM_ar, 'x-', label='hpz')

        energy_ar = np.array([])
        FWHM_ar = np.array([])
        for scan_num in hpy_scan_num_ar:
            scan = DesyScanImporter('p10', self.path, self.p10_newfile, scan_num, creat_save_folder=False)
            energy = scan.get_motor_pos('fmbenergy')
            energy_ar = np.append(energy_ar, energy)
            cen, FWHM = scan.knife_edge_estimation('diffdio', sigma=sigma, smooth=True, plot=False)
            FWHM_ar = np.append(FWHM_ar, FWHM)

        plt.plot(energy_ar, FWHM_ar, 'x-', label='hpy')
        plt.xlabel('energy (eV)')
        plt.ylabel('FWHM')
        plt.legend()
        plt.show()
        return

    def scripts_measure_beamsize(self, hpz_edge, hpy_edge):
        """
        Generate the script for the measurement of the beamsize.

        The script name would be 'sequence_06.txt'.

        Parameters
        ----------
        hpz_edge : list
            The position of the wire edge to measure the beam size in hpz direction in the order of hpx, hpy, hpz.
        hpy_edge : list
            The position of the wire edge to measure the beam size in hpy direction in the order of hpx, hpy, hpz.

        Returns
        -------
        None.

        """
        pathsavetmp = os.path.join(self.pathsave, 'sequence_06.txt')
        text = '#The scripts to measure the beamsize\n'
        rep_text = '\n####vertical beamsize####\numv hpx %.1f\numv hpy %.1f\numv hpz %.1f\ndscan hpz -10 10 100 0.4\n\n####horizontal beamsize####\numv hpx %.1f\numv hpy %.1f\numv hpz %.1f\ndscan hpy -10 10 100 0.4\n'
        for i in range(2):
            text += rep_text % (hpz_edge[0], hpz_edge[1], hpz_edge[2], hpy_edge[0], hpy_edge[1], hpy_edge[2])
        with open(pathsavetmp, 'w') as f:
            f.write(text)
        return

    def measure_beamsize(self, scan_num_range, sigma=1.0):
        """
        Calculate the measured beamsize.

        Parameters
        ----------
        scan_num_range : list
            The scan num range for calculating the measured beamsize.
        sigma : float, optional
            The estimated size of the direct beam. The default is 1.0.

        Returns
        -------
        None.

        """
        for scan_num in range(scan_num_range[0], scan_num_range[-1] + 1):
            scan = DesyScanImporter('p10', self.path, self.p10_newfile, scan_num, creat_save_folder=False)
            cen, FWHM = scan.knife_edge_estimation('diffdio', sigma=sigma, smooth=True, plot=False)
            print('FWHM in %s direction: %.2f' % (scan.get_scan_motor(), FWHM))
        return

    def scripts_hexary_vs_beamsize(self, hpz_edge, hexary_start=-0.2, step_num=9, hexary_step=0.05):
        """
        Generate the script for the hexary vs beamsize scans.

        The script name would be 'sequence_07.txt'.

        Parameters
        ----------
        hpz_edge : list
            The position of the wire edge to measure the beam size in hpz direction in the order of hpx, hpy, hpz.
        hexary_start : float, optional
            The starting position of the hexary motor relative to the current position. The default is -.2.
        step_num : int, optional
            The total step num to be used for the hexaray motor. The default is 9.
        hexary_step : float, optional
            The step size for the hexary motor. The default is 0.05.

        Returns
        -------
        None.

        """
        pathsavetmp = os.path.join(self.pathsave, 'sequence_07.txt')
        text = ''
        start_text = '#The scripts to check the beamsize with respect to hexary\nsenv SignalCounter diffdio\numv hpx %.1f\numv hpy %.1f\numv hpz %.1f\numvr hpz -500\n'
        rep_text = '\n#hexary = %.2f\numvr hexary %.2f\ndscan hexaz -0.25 0.25 50 .2\nmvsa cen 0\numvr hpz 500\ndscan hpz -7.5 7.5 150 0.2\numvr hpz -500\n'
        end_text = '\numvr hpz 500\n'
        text += start_text % (hpz_edge[0], hpz_edge[1], hpz_edge[2])
        for i in range(step_num):
            if i == 0:
                text += rep_text % (hexary_start, hexary_start)
            else:
                text += rep_text % (hexary_start + i * hexary_step, hexary_step)
        text += end_text
        with open(pathsavetmp, 'w') as f:
            f.write(text)
        return

    def hexary_vs_beamsize(self, scan_num_range, sigma=1.0):
        """
        Calculate the beamsize with respect to hexary motor.

        Parameters
        ----------
        scan_num_range : list
            The scan number range.
        sigma : float, optional
            The expected beamsize. The default is 1.0.

        Returns
        -------
        None.

        """
        scan_num_ar = range(scan_num_range[0] + 1, scan_num_range[1] + 1, 2)
        Motor_name = 'hexary'
        Counter_name = 'diffdio'
        pos_ar = np.array([])
        FWHM_ar = np.array([])
        for scan_num in scan_num_ar:
            scan = DesyScanImporter('p10', self.path, self.p10_newfile, scan_num, creat_save_folder=False)
            pos_ar = np.append(pos_ar, scan.get_motor_pos(Motor_name))
            cen, FWHM = scan.knife_edge_estimation('diffdio', sigma=sigma, smooth=True, plot=False)
            FWHM_ar = np.append(FWHM_ar, FWHM)

        plt.plot(pos_ar, FWHM_ar, "x-")
        plt.legend()
        plt.ylabel(Counter_name)
        plt.xlabel(Motor_name)
        plt.show()
        return

    def scripts_hexarz_vs_beamsize(self, hpy_edge, hexarz_start=-0.2, step_num=9, hexarz_step=0.05):
        """
        Generate the script for the hexarz vs beamsize scans.

        The script name would be 'sequence_08.txt'.

        Parameters
        ----------
        hpy_edge : list
            The position of the wire edge to measure the beam size in hpy direction in the order of hpx, hpy, hpz.
        hexarz_start : float, optional
            The starting position of the hexarz motor relative to the current position. The default is -.2.
        step_num : int, optional
            The total step num to be used for the hexaraz motor. The default is 9.
        hexarz_step : float, optional
            The step size for the hexarz motor. The default is 0.05.

        Returns
        -------
        None.

        """
        pathsavetmp = os.path.join(self.pathsave, 'sequence_08.txt')
        text = ''
        start_text = '#The scripts to check the beamsize with respect to hexarz\nsenv SignalCounter diffdio\numv hpx %.1f\numv hpy %.1f\numv hpz %.1f\numvr hpy -500\n'
        rep_text = '\n#hexarz = %.2f\numvr hexarz %.2f\ndscan hexay -0.25 0.25 50 .2\nmvsa cen 0\numvr hpy 500\ndscan hpy -7.5 7.5 150 0.2\numvr hpy -500\n'
        end_text = '\numvr hpy 500\n'
        text += start_text % (hpy_edge[0], hpy_edge[1], hpy_edge[2])
        for i in range(step_num):
            if i == 0:
                text += rep_text % (hexarz_start, hexarz_start)
            else:
                text += rep_text % (hexarz_start + i * hexarz_step, hexarz_step)
        text += end_text
        with open(pathsavetmp, 'w') as f:
            f.write(text)
        return

    def hexarz_vs_beamsize(self, scan_num_range, sigma=1.0):
        """
        Calculate the beamsize with respect to hexarz motor.

        Parameters
        ----------
        scan_num_range : list
            The scan number range.
        sigma : float, optional
            The expected beamsize. The default is 1.0.

        Returns
        -------
        None.

        """
        scan_num_ar = range(scan_num_range[0] + 1, scan_num_range[1] + 1, 2)
        Motor_name = 'hexarz'
        Counter_name = 'diffdio'
        pos_ar = np.array([])
        FWHM_ar = np.array([])
        for scan_num in scan_num_ar:
            scan = DesyScanImporter('p10', self.path, self.p10_newfile, scan_num, creat_save_folder=False)
            pos_ar = np.append(pos_ar, scan.get_motor_pos(Motor_name))
            cen, FWHM = scan.knife_edge_estimation('diffdio', sigma=sigma, smooth=True, plot=False)
            FWHM_ar = np.append(FWHM_ar, FWHM)

        plt.plot(pos_ar, FWHM_ar, "x-")
        plt.legend()
        plt.ylabel(Counter_name)
        plt.xlabel(Motor_name)
        plt.show()
        return

    def scripts_depth_of_focus(self, hpz_edge, hpy_edge, hpx_start=-500, hpx_end=500, hpx_step=10):
        """
        Generate the script to measure the depth of focus using the hpx motor.

        Parameters
        ----------
        hpz_edge : list
            The position of the wire edge to measure the beam size in hpz direction in the order of hpx, hpy, hpz.
        hpy_edge : list
            The position of the wire edge to measure the beam size in hpy direction in the order of hpx, hpy, hpz.
        hpx_start : float, optional
            The start position of the hpx motor relative to the current position. The default is -500.
        hpx_end : float, optional
            The end position of the hpx motor relative to the current position. The default is 500.
        hpx_step : int, optional
            The number of steps to be taken. The default is 10.

        Returns
        -------
        None.

        """
        pathsavetmp = os.path.join(self.pathsave, 'sequence_09.txt')
        text = ''
        start_text = '#The scripts to check the depth of focus\nsenv SignalCounter diffdio\n'
        rep_text = '\n#hpx = %.1f\np10_pause 3\n\numv hpx %.1f\numv hpy %.1f\numv hpz %.1f\numvr hpx %.1f\ndscan hpz -50 50 50 .2\nmvsa dip 0\numvr hpz 12.5\ndscan hpz -10 10 80 0.2\n\numv hpx %.1f\numv hpy %.1f\numv hpz %.1f\numvr hpx %.1f\ndscan hpy -50 50 50 .2\nmvsa dip 0\numvr hpy 12.5\ndscan hpy -10 10 80 0.2\n'
        end_text = '\numvr hpx %.1f\n'
        text += start_text
        for i in np.linspace(hpx_start, hpx_end, hpx_step):
            text += rep_text % (i, hpz_edge[0], hpz_edge[1], hpz_edge[2], i, hpy_edge[0], hpy_edge[1], hpy_edge[2], i)
        text += end_text % (-hpx_end)
        with open(pathsavetmp, 'w') as f:
            f.write(text)
        return

    def depth_of_focus(self, scan_num_range, sigma=1.0):
        """
        Plot the beamsize with respect to the hpx motor.

        Parameters
        ----------
        scan_num_range : list
            The scan number range.
        sigma : float, optional
            The expected beamsize. The default is 1.0.

        Returns
        -------
        None.

        """
        hpz_scan_num_ar = np.arange(scan_num_range[0], scan_num_range[1], 4)
        hpy_scan_num_ar = hpz_scan_num_ar + 2.0
        hpx_ar = np.array([])
        FWHM_ar = np.array([])
        for scan_num in hpz_scan_num_ar:
            scan = DesyScanImporter('p10', self.path, self.p10_newfile, scan_num, creat_save_folder=False)
            hpx_ar = np.append(hpx_ar, scan.get_motor_pos('hpx'))
            cen, FWHM = scan.knife_edge_estimation('diffdio', sigma=sigma, smooth=True, plot=False)
            FWHM_ar = np.append(FWHM_ar, FWHM)
        hpx_ar = hpx_ar - np.averge(hpx_ar)
        plt.plot(hpx_ar, FWHM_ar, 'x-', label='hpz')

        hpx_ar = np.array([])
        FWHM_ar = np.array([])
        for scan_num in hpy_scan_num_ar:
            scan = DesyScanImporter('p10', self.path, self.p10_newfile, scan_num, creat_save_folder=False)
            hpx_ar = np.append(hpx_ar, scan.get_motor_pos('hpx'))
            cen, FWHM = scan.knife_edge_estimation('diffdio', sigma=sigma, smooth=True, plot=False)
            FWHM_ar = np.append(FWHM_ar, FWHM)

        plt.plot(hpx_ar, FWHM_ar, 'x-', label='hpy')
        plt.xlabel('hpx (um)')
        plt.ylabel('FWHM')
        plt.legend()
        plt.show()
        return


if __name__ == '__main__':
    start_alignment()
