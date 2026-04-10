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
scan_num_ar = [622]
p08_file = ["PTO_STO_DSO_730"]

# path information
path = r"U:\2023\data\11016147\raw"
path_eiger1m_mask = r''
pathsavefolder = r"E:\Work place 3\sample\XRD\20230615 PTO_insitu"

# The rois for the Eiger 4M detector
eiger1m_roi1 = [0, 1000, 0, 1000]
eiger1m_roi2 = [1264, 1464, 1278, 1478]
eiger1m_roi3 = [1630, 1660, 950, 1000]
eiger1m_roi4 = [400, 600, 300, 500]

cal_eiger1m_roi = []

# Plot selection
counter_select = ['eiger1m_roi1']
scale = 'Linear'
# scale = 'Normalized'
# scale = 'Log'

# %%Integrating the detector roi, write new fio and spec file
for i, scan_num in enumerate(scan_num_ar):
    if len(scan_num_ar) != len(p08_file):
        p08_newfile = p08_file[0]
    else:
        p08_newfile = p08_file[i]

    if len(cal_eiger1m_roi) != 0:
        scan = P08EigerScan(path, p08_newfile, scan_num, 'eiger1m', pathsavefolder, path_eiger1m_mask)
        scan.eiger_roi_sum(cal_eiger1m_roi, roi_order='XY', save_img_sum=True)
        scan.write_fio()

    if (len(cal_eiger1m_roi) == 0):
        scan = P08Scan(path, p08_newfile, scan_num, pathsavefolder)

    print(scan)
    print('Following counters are available, please select at least one of them to plot:')
    print(scan.get_counter_names())

# %% Plot and save
assert len(counter_select) > 0, 'Please select at least one of the counters to plot!'
for counter in counter_select:
    assert (counter in scan.get_counter_names()), 'The counter %s does not exist! Please check the name of the counter again!' % counter

plt_y = int(np.sqrt(len(counter_select)))
plt_x = len(counter_select) // plt_y
plt.figure(figsize=(8 * plt_x, 8 * plt_y))
for i, scan_num in enumerate(scan_num_ar):
    if len(scan_num_ar) != len(p08_file):
        p08_newfile = p08_file[0]
    else:
        p08_newfile = p08_file[i]

    scan = P08Scan(path, p08_newfile, scan_num, pathsavefolder)
    command_infor = scan.get_command_infor()

    ion_bl_ar = scan.get_scan_data('ion_bl') / np.round(np.average(scan.get_scan_data('ion_bl')))
    motor = command_infor['motor1_name']
    motor_pos = scan.get_scan_data(motor)

    for i, counter_name in enumerate(counter_select):
        intensity = scan.get_scan_data(counter_name) / ion_bl_ar

        plt.subplot(plt_y, plt_x, i + 1)
        if scale == 'Linear':
            plt.plot(motor_pos, intensity, label='scan%d' % (scan_num))
        elif scale == 'Normalized':
            plt.plot(motor_pos, intensity / np.amax(intensity), label='scan%d' % (scan_num))
        elif scale == 'Log':
            plt.plot(motor_pos, np.log10(intensity + 1.0), label='scan%d' % (scan_num))
        plt.title(counter_name)
        plt.xlabel("%s" % motor)
        plt.ylabel('Intensity (a.u.)')
        plt.legend()
    plt.show()

    for i, scan_num in enumerate(scan_num_ar):
        if len(scan_num_ar) != len(p08_file):
            p08_newfile = p08_file[0]
        else:
            p08_newfile = p08_file[i]
        scan = P08Scan(path, p08_newfile, scan_num, pathsavefolder)

        eiger1mcounters = [counter_name for counter_name in counter_select if 'eiger1m' in counter_name]
        if len(eiger1mcounters) > 0:
            plt.figure(figsize=(8, 8))
            img_sum = scan.get_imgsum(det_type='eiger1m')
            plt.imshow(np.log10(img_sum + 1.0), cmap='jet')
            for counter_name in eiger1mcounters:
                if counter_name != 'eiger1m_full':
                    draw_roi(scan.get_motor_pos(counter_name), counter_name)
            plt.title('scan%05d' % scan_num)
            plt.show()
