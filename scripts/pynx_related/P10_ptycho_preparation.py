# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 10:42:06 2022

@author: renzhe
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(r'F:\Work place 3\testprog\pyCXIM_master')
from pyCXIM.scan_reader.Desy.eiger_reader import DesyEigerImporter


def draw_rectangular(box_pos, margin):
    """
    Draw the box to check the position of the beam stop.

    Parameters
    ----------
    box_pos : list
        The positiono of the inner circle of the box in [Ymin, Ymax, Xmin, Xmax] order.
    margin : int
        THe margin of the box edge.

    Returns
    -------
    None.

    """

    inner_box = np.array([box_pos[0], box_pos[1], box_pos[2], box_pos[3]], dtype=int)
    outer_box = np.array([box_pos[0] - margin, box_pos[1] + margin, box_pos[2] - margin, box_pos[3] + margin], dtype=int)
    for pos_ar in [inner_box, outer_box]:
        v_line = np.arange(pos_ar[0], pos_ar[1] + 1)
        h_line = np.arange(pos_ar[2], pos_ar[3] + 1)
        plt.plot((np.zeros_like(v_line) + pos_ar[2]), v_line, "r-", alpha=0.3)
        plt.plot((np.zeros_like(v_line) + pos_ar[3]), v_line, "r-", alpha=0.3)
        plt.plot(h_line, (np.zeros_like(h_line) + pos_ar[0]), "r-", alpha=0.3)
        plt.plot(h_line, (np.zeros_like(h_line) + pos_ar[1]), "r-", alpha=0.3)
    return


# %% load basic information

# Inputs: general information
year = "2020"
beamtimeID = "11010012"
p10_file = r"siemenstar"
scan_num = 6

# Inputs:Detector parameters
detector = 'e4m'
distance = 4990

# Direct beam position on the detector Y, X
cch = [1369, 1380]
r0 = 5
wxy = [240, 240]
bs1 = [1334, 1401, 1346, 1409]                                                               #the with of the 5mm*5mm beam stopper with respect to the direct beam position
bs2 = [1356, 1382, 1366, 1391]                                                               #the width of the 3mm*3mm beam stopper with respect to the direct beam position
margin = 2

# please check the transmission with https://henke.lbl.gov/optical_constants/
# transmission factor of 5mm*5mm (100um thick) Si wafer
factor_large = 0.26897
# transmission factor of 3mm*3mm (100um thick) Si wafer
factor_small = 0.26897

# Inputs: paths
path = r"E:\Work place 3\sample\XRD\20200330 Inhouse P10 desy\Ptychography raw"
pathsave = r"E:\Work place 3\sample\XRD\Test"
pathmask = r'E:\Work place 3\testprog\X-ray diffraction\Common functions\general_mask.npy'

# %%Mask generator
scan = DesyEigerImporter('p10', path, p10_file, scan_num, detector, pathsave, pathmask)
img = scan.eiger_img_sum(sum_img_num=200)
plt.subplot(2, 2, 1)
plt.imshow(np.log10(img + 1.0), cmap='jet')
draw_rectangular(bs1, margin)
draw_rectangular(bs2, margin)
plt.subplot(2, 2, 2)
mask, correction = scan.eiger_semi_transparent_mask(cch, r0, bs1, factor_large, bs2, factor_small, margin)
plt.imshow(np.log10((img * correction + 1.0)), cmap='Blues')
plt.imshow(np.ma.masked_where(mask == 0, mask), cmap='Reds', alpha=0.5, vmin=0, vmax=1)
plt.subplot(2, 2, 3)
wxy = scan.eiger_cut_check(cch, wxy)
plt.imshow(np.log10((img * correction + 1.0)[(cch[0] - wxy[0]):(cch[0] + wxy[0]), (cch[1] - wxy[1]):(cch[1] + wxy[1])]), cmap='Blues')
plt.imshow(np.ma.masked_where(mask == 0, mask)[(cch[0] - wxy[0]):(cch[0] + wxy[0]), (cch[1] - wxy[1]):(cch[1] + wxy[1])], cmap='Reds', alpha=0.5, vmin=0, vmax=1)
plt.subplot(2, 2, 4)
plt.plot(scan.get_scan_data('curpetra'))
plt.show()

# # %%
hpy = scan.get_scan_data('hpy')
hpz = scan.get_scan_data('hpz')
index_array = np.logical_and(hpy > 410.9, hpz < 344)
scan.eiger_ptycho_cxi(cch, wxy, detector_distance=distance, index_array=index_array)
