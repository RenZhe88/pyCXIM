# -*- coding: utf-8 -*-
"""
Description
Created on Thu May  4 15:52:44 2023

@author: renzhe
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(r'E:\Work place 3\testprog\pyCXIM_master')
from pyCXIM.p08_scan_reader.p08_eiger_reader import P08EigerScan
from pyCXIM.p08_scan_reader.p08_scan_reader import P08Scan

def draw_roi(roi, roi_name=''):
    v_line = np.arange(roi[0], roi[1])
    h_line = np.arange(roi[2], roi[3])
    plt.plot((np.zeros_like(v_line) + roi[2]), v_line, 'r-', alpha=0.3)
    plt.plot((np.zeros_like(v_line) + roi[3]), v_line, 'r-', alpha=0.3)
    plt.plot(h_line, (np.zeros_like(h_line) + roi[0]), 'r-', alpha=0.3)
    plt.plot(h_line, (np.zeros_like(h_line) + roi[1]), 'r-', alpha=0.3)
    if roi_name != '':
        plt.text((roi[2] + roi[3]) / 2 - 150, roi[0] - 30, roi_name, color='r', fontsize=16)
    return


# %% Inputs
scan_num_ar = range(623, 642)
p08_file = "PTO_STO_DSO_730"
motor1 = 'xs'
motor2 = 'ys'


# path information
path = r"U:\2023\data\11016147\raw"
path_eiger1m_mask = r''
pathsavefolder = r"E:\Work place 3\sample\XRD\20230615 PTO_insitu"

# The rois for the Eiger 4M detector
eiger1m_roi1 = [0, 1000, 0, 1000]
eiger1m_roi2 = [414, 457, 200, 900]
eiger1m_roi3 = [1630, 1660, 950, 1000]
eiger1m_roi4 = [400, 600, 300, 500]

cal_eiger1m_roi = [eiger1m_roi1, eiger1m_roi2]

# Plot selection
counter_select = ['eiger1m_roi1', 'eiger1m_roi2']
scale = 'Linear'
# scale = 'Normalized'
# scale = 'Log'

# %%Integrating the detector roi, write new fio and spec file
scan_num_ar = list(scan_num_ar)
scan_num_ar.append(scan_num_ar[-1] + 1)

pathsavefolder = os.path.join(pathsavefolder, '%s_map_%04d_%04d' % (p08_file, scan_num_ar[0], scan_num_ar[-1]))
if not os.path.exists(pathsavefolder):
    os.mkdir(pathsavefolder)

for i, scan_num in enumerate(scan_num_ar):
    p08_newfile = p08_file

    if len(cal_eiger1m_roi) != 0:
        scan = P08EigerScan(path, p08_newfile, scan_num, 'eiger1m', pathsavefolder, path_eiger1m_mask)
        scan.eiger_roi_sum(cal_eiger1m_roi, roi_order='XY', save_img_sum=True)
        scan.write_fio()

    if (len(cal_eiger1m_roi) == 0):
        scan = P08Scan(path, p08_newfile, scan_num, pathsavefolder)

    print(scan)

# %% Plot and save
assert len(counter_select) > 0, 'Please select at least one of the counters to plot!'
for counter in counter_select:
    assert (counter in scan.get_counter_names()), 'The counter %s does not exist! Please check the name of the counter again!' % counter

    scan = P08Scan(path, p08_newfile, scan_num_ar[0], pathsavefolder)
    xdim = int(scan.get_num_points())
    ydim = len(scan_num_ar)

    motor1_pos = np.zeros((ydim, xdim))
    motor2_pos = np.zeros((ydim, xdim))
    intensity = np.zeros((ydim, xdim))

    for i, scan_num in enumerate(scan_num_ar):
        scan = P08Scan(path, p08_newfile, scan_num, pathsavefolder)

        command_infor = scan.get_command_infor()
        ion_bl_ar = scan.get_scan_data('ion_bl') / np.round(np.average(scan.get_scan_data('ion_bl')))

        motor1_pos[i, :] = scan.get_scan_data(motor1)
        motor2_pos[i, :] = scan.get_motor_pos(motor2)
        intensity[i, :] = (scan.get_scan_data(counter) / ion_bl_ar)

    if counter[:7] == 'eiger1m':
        img_sum = scan.get_imgsum(det_type='eiger1m')
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(np.log10(img_sum + 1.0))
        if counter != 'eiger1m_full':
            draw_roi(scan.get_motor_pos(counter), counter)
        plt.subplot(1, 2, 2)
    else:
        plt.figure(figsize=(8, 8))
    plt.contourf(motor1_pos, motor2_pos, np.log10(intensity + 1.0), 150, cmap="jet")
    plt.axis('scaled')
    plt.xlabel("%s (mm)" % motor1)
    plt.ylabel("%s (mm)" % motor2)
    plt.savefig(os.path.join(pathsavefolder, '%s_scan%05d_%05d_%s.png' % (scan.get_p08_file(), scan_num_ar[0], scan_num_ar[-1] - 1, counter)))
    plt.show()
