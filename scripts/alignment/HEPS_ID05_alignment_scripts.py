# -*- coding: utf-8 -*-
import numpy as np
import os
import sys
sys.path.append(r'E:\Work place 3\testprog\pyCXIM_master')
from pyCXIM.scan_reader.HEPS.spec_reader import HEPSScanImporter
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def start_alignment():
    path = r'F:\Raw Data\20260422_ID05_HEPS_BCDI_test\raw\RenZhe'
    id05_newfile = r'Sample1_1'
    pathsave = r'F:\Work place 4\Temp'
    align = alignment(path, id05_newfile, pathsave)

    # ####align phi center of rotation: Step1####
    # scan_num = [58, 61]
    # align.align_phi_rotation_step1(scan_num)

    # ####align omega center of rotation####
    # scan_num = [93, 94, 95]
    # align.align_om_rotation(scan_num)

    ####Measurement of the beamsize####
    scan_num = [100]
    align.measure_beamsize(scan_num)

class alignment():
    def __init__(self, path, id05_newfile, pathsave):
        self.path = path
        self.id05_newfile = id05_newfile
        self.pathsave = pathsave
        self.path_save_corrrection = os.path.join(pathsave, 'correction.txt')

    def align_phi_rotation_step1(self, scan_num_range, sigma=0.005):
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
            scan = HEPSScanImporter('id05_6c', self.path, self.id05_newfile, scan_num, creat_save_folder=False)
            cen = scan.tophat_estimation('imroi1', sigma=sigma, normalize_signal=None, smooth=True, plot=True)
            cen_ar = np.append(cen_ar, cen)
        plt.legend()
        plt.show()
        print('umv tx %.4f, umvr Xtable %.4f' % ((cen_ar[1] + cen_ar[0]) / 2.0, -(cen_ar[0] - cen_ar[1]) / 2.0))
        return

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

    def align_om_rotation(self, scan_num_ar, sigma=0.005):
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
        cen_ar = np.array([])
        angle_ar = np.array([])
        scan = HEPSScanImporter('id05_6c', self.path, self.id05_newfile, scan_num_ar[0], creat_save_folder=False)

        for scan_num in scan_num_ar:
            scan = HEPSScanImporter('id05_6c', self.path, self.id05_newfile, scan_num, creat_save_folder=False)
            angle = scan.get_motor_pos('eta')
            angle_ar = np.append(angle_ar, angle)
            cen = scan.tophat_estimation('imroi1', sigma=sigma, normalize_signal=None, smooth=True, plot=True)
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
        print('umvr ty %.4f, umvr zz %.4f, umvr zt %.4f' % (-shift_x, shift_y, -shift_y))
        return

    def measure_beamsize(self, scan_num_ar, sigma=0.005):
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
        for scan_num in scan_num_ar:
            scan = HEPSScanImporter('id05_6c', self.path, self.id05_newfile, scan_num, creat_save_folder=False)
            plt.figure(figsize=(16,8))
            cen, FWHM = scan.knife_edge_estimation('imroi1', sigma=sigma, normalize_signal=None, display_range=1.0, smooth=True, plot=True)
            plt.show()
            print('FWHM in %s direction: %.4f' % (scan.get_scan_motor(), FWHM))
        return


if __name__ == '__main__':
    start_alignment()
