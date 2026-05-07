#!/usr/local/bin/python2.7.3 -tttt
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import sys
sys.path.append(r'F:\Work place 3\testprog\pyCXIM_master')
from pyCXIM.scan_reader.HEPS.tif_reader import HEPSTifImporter
from pyCXIM.scan_reader.HEPS.spec_reader import HEPSScanImporter

"""
The code to put the single crystalline particle into the center of rotation at P10.

Example:    umvr om -.3
            e4mdscan hpx -5 5 50 1 0 0.5
            umvr om .3
            e4mdscan hpx -5 5 50 1 0 0.5
            umvr om .3
            e4mdscan hpx -5 5 50 1 0 0.5
            umvr om -.3

Input:
    scan_ar: several line scans recorded at different angle around the Bragg peak
    p10_file: the p10_newfile name
    path: The raw folder for the experiment
    pathmask: the path for the mask of the detector
    pch: the estimated center for the peak position in [X, Y]
    wxy: the half width of the roi around the peak position

Output:
    print on the screen the suggestion to move the particle in the center of rotation

In case of questions, please contact me. Detailed explaination file of the code could be sent upon requist.
Author: Ren Zhe
Date: 2024/05/30
Email: zhe.ren@desy.de or renzhetu001@gmail.com
"""


def vertical_shift(x, A, B):
    return A / np.tan(x) + B


def horizontal_shift(x, A, B):
    return A * np.tan(x) + B


# Inputs: general information
sample_name = r"Sample1_1"
path = r"F:\Raw Data\20260422_ID05_HEPS_BCDI_test\raw\RenZhe"
pathmask = r'F:\Work place 3\testprog\pyCXIM_master\detector_mask\id05_e4m_mask.npy'
counter = r'e4m_roi1'

# The scan numbers for the alignment
scan_num_ar = [174, 175, 176]
# The center of the diffraction peak on the detector in X, Y
pch = [714, 805]
# The half width of the roi on the detector
wxy = [100, 100]
# Please select the geometry for the diffraction
geometry = 'out_of_plane'
# geometry = 'in_plane'

pk_ar = np.zeros_like(scan_num_ar, dtype=float)
angle_ar = np.zeros_like(scan_num_ar, dtype=float)
for i, scan_num in enumerate(scan_num_ar):
    if counter == 'e4m_roi1':
        scan = HEPSTifImporter('id05_6c', path, sample_name, scan_num, detector='e4m', pathmask=pathmask, creat_save_folder=False)
    else:
        scan = HEPSScanImporter('id05_6c', path, sample_name, scan_num, creat_save_folder=False)
    motor_name = scan.get_scan_motor()
    if geometry == 'out_of_plane':
        angle_ar[i] = np.radians(scan.get_motor_pos('eta'))
    elif geometry == 'in_plane':
        angle_ar[i] = np.radians(scan.get_motor_pos('phi'))
    if counter == 'e4m_roi1':
        scan.image_roi_sum([pch[0] - wxy[0], pch[0] + wxy[0], pch[1] - wxy[1], pch[1] + wxy[1]], roi_order='XY', save_img_sum=False)
        amp, cen, FWHM = scan.Gaussian_estimation('e4m_roi1', sigma=0.001, normalize_signal=None, normalize=True, plot=True)
    else:
        amp, cen, FWHM = scan.Gaussian_estimation(counter, sigma=0.001, normalize_signal=None, normalize=True, plot=True)
    pk_ar[i] = cen

pk_ar = pk_ar[np.argsort(angle_ar)]
angle_ar = np.sort(angle_ar)
print(np.rad2deg(angle_ar))
plt.show()
if geometry == 'out_of_plane':
    popt, pcov = curve_fit(vertical_shift, angle_ar, pk_ar, p0=[20, 0])
    print("Please move relatively zz by %.4f, %s by %.4f" % (-popt[0], scan.get_scan_motor(), popt[0] / np.tan(np.average(angle_ar))))
elif geometry == 'in_plane':
    popt, pcov = curve_fit(horizontal_shift, angle_ar, pk_ar, p0=[20, 0])
    print("Please move relatively zz by %.4f, %s by %.4f" % (-popt[0], scan.get_scan_motor(), popt[0] / np.tan(np.average(angle_ar))))
