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
sys.path.append(r'F:\Work place 3\testprog\pyCXIM_master')
from pyCXIM.scan_reader.HEPS.pilatus_reader import HEPSPilatusImporter
from pyCXIM.scan_reader.HEPS.spec_reader import HEPSScanImporter
# from pyCXIM.scan_reader.p10_fluo_reader import P10FluoScan

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
scan_num_ar = [6]
sample_names = ["S2_1"]
beamline_mode = 'id05_6c'

# path information
path = r"F:\Work place 4\sample\XRD\Additional Task\20260325 ID05 RSM\raw\Zhanglinxing"
path_e4m_mask = r'F:\Work place 3\testprog\pyCXIM_master\detector_mask\p10_e4m_mask.npy'
path_pilatus_mask = r'E:\Work place 3\testprog\X-ray diffraction\Common functions\pilatus_mask.npy'
pathsavefolder = r"F:\Work place 4\Temp"

# The rois for the Eiger 4M detector
e4m_roi1 = [1050, 1750, 450, 1250]
e4m_roi2 = [100, 2000, 500, 750]
e4m_roi3 = [100, 2000, 100, 400]

cal_e4m_roi = []

# The rois for the Eiger500 detector
pilatus_roi1 = [120, 200, 100, 300]
pilatus_roi2 = [100, 300, 100, 600]
cal_pilatus_roi = [pilatus_roi1]

# Plot selection
counter_select = ['eta', 'pilatus_roi1']
# scale = 'Linear'
# scale = 'Normalized'
scale = 'Log'

# %% Sorting the scan types
if len(scan_num_ar) != len(sample_names):
    sample_names = sample_names[0] * len(scan_num_ar)
else:
    sample_names = sample_names

dscan_scan_num = []
d2scan_scan_num = []
mesh_scan_num = []
for i, scan_num in enumerate(scan_num_ar):
    scan = HEPSScanImporter(beamline_mode, path, sample_names[i], scan_num, pathsavefolder)

    if scan.get_scan_type() in ['ascan', 'dscan']:
        dscan_scan_num.append((sample_names[i], scan_num))
    elif scan.get_scan_type() in ['a2scan', 'd2scan']:
        d2scan_scan_num.append((sample_names[i], scan_num))
    elif scan.get_scan_type() in ['dmesh', 'mesh']:
        mesh_scan_num.append((sample_names[i], scan_num))

# %% Calculate the roi intensity
for i, scan_num in enumerate(scan_num_ar):
    if len(cal_e4m_roi) != 0:
        scan = DesyEigerImporter(beamline_mode, path, sample_names[i], scan_num, 'e4m', pathsavefolder, path_e4m_mask)
        scan.image_roi_sum(cal_e4m_roi, roi_order='XY', save_img_sum=True)
        
    if len(cal_pilatus_roi) != 0:
        scan = HEPSPilatusImporter(beamline_mode, path, sample_names[i], scan_num, 'pilatus', pathsavefolder, path_pilatus_mask)
        scan.image_roi_sum(cal_pilatus_roi, roi_order='XY', save_img_sum=True)

    if (len(cal_e4m_roi) == 0) and (len(cal_pilatus_roi) == 0):
        scan = HEPSScanImporter(beamline_mode, path, sample_names[i], scan_num, pathsavefolder)
    
    scan.write_scan()
    print(scan)
    print('Following counters are available, please select at least one of them to plot:')
    print(scan.get_counter_names())

# %% Plot and save
assert len(counter_select) > 0, 'Please select at least one of the counters to plot!'
for counter in counter_select:
    assert (counter in scan.get_counter_names()) or (scan.name_converter_short_to_full(counter) in scan.get_counter_names()), 'The counter %s does not exist! Please check the name of the counter again!' % counter

# Plot one motor line scans
if len(dscan_scan_num) > 0:
    plt_y = int(np.sqrt(len(counter_select)))
    plt_x = len(counter_select) // plt_y
    plt.figure(figsize=(8 * plt_x, 8 * plt_y))
    for sample_name, scan_num in dscan_scan_num:
        scan = HEPSScanImporter(beamline_mode, path, sample_name, scan_num, pathsavefolder)
        command_infor = scan.get_command_infor()
        # curpetra_ar = scan.get_scan_data('curpetra') / np.round(np.average(scan.get_scan_data('curpetra')))
        motor = command_infor['motor_name']
        motor_pos = scan.get_scan_data(motor)

        for i, counter_name in enumerate(counter_select):
            assert (counter in scan.get_counter_names()) or (scan.name_converter_short_to_full(counter) in scan.get_counter_names()), 'Counter %s does not exist, please check it again!' % counter_name
            intensity = scan.get_scan_data(counter_name)

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

    for sample_name, scan_num in dscan_scan_num:
        scan = HEPSScanImporter(beamline_mode, path, sample_name, scan_num, pathsavefolder)

        e4mcounters = [counter_name for counter_name in counter_select if 'e4m' in counter_name]
        if len(e4mcounters) > 0:
            plt.figure(figsize=(8, 8))
            img_sum = scan.get_imgsum(det_type='e4m')
            plt.imshow(np.log10(img_sum + 1.0), cmap='jet')
            for counter_name in e4mcounters:
                if counter_name != 'e4m_full':
                    draw_roi(scan.get_motor_pos(counter_name), counter_name)
            plt.title('scan%05d' % scan_num)
            plt.show()

        pilatuscounters = [counter_name for counter_name in counter_select if 'pilatus' in counter_name]
        if len(pilatuscounters) > 0:
            plt.figure(figsize=(8, 8))
            img_sum = scan.get_imgsum(det_type='pilatus')
            plt.imshow(np.log10(img_sum + 1.0), cmap='jet')
            for counter_name in e4mcounters:
                if counter_name != 'pilatus_full':
                    draw_roi(scan.get_motor_pos(counter_name), counter_name)
            plt.title('scan%05d' % scan_num)
            plt.show()

# Plot two motor line scans
if len(d2scan_scan_num) > 0:
    plt_y = 2
    plt_x = int(len(counter_select))

    plt.figure(figsize=(8 * plt_x, 8 * plt_y))
    for sample_name, scan_num in d2scan_scan_num:
        scan = HEPSScanImporter(beamline_mode, path, sample_name, scan_num, pathsavefolder)
        command_infor = scan.get_command_infor()
        # curpetra_ar = scan.get_scan_data('curpetra') / np.round(np.average(scan.get_scan_data('curpetra')))
        motor1 = command_infor['motor1_name']
        motor1_pos = scan.get_scan_data(motor1)
        motor2 = command_infor['motor2_name']
        motor2_pos = scan.get_scan_data(motor2)

        for i, counter_name in enumerate(counter_select):
            assert (counter in scan.get_counter_names()) or (scan.name_converter_short_to_full(counter) in scan.get_counter_names()), 'Counter %s does not exist, please check it again!' % counter_name
            intensity = scan.get_scan_data(counter_name)

            plt.subplot(plt_y, plt_x, i + 1)
            if scale == 'Linear':
                plt.plot(motor1_pos, intensity, label='scan%d' % (scan_num))
            elif scale == 'Normalized':
                plt.plot(motor1_pos, intensity / np.amax(intensity), label='scan%d' % (scan_num))
            elif scale == 'Log':
                plt.plot(motor1_pos, np.log10(intensity + 1.0), label='scan%d' % (scan_num))
            plt.title(counter_name)
            plt.xlabel("%s" % motor1)
            plt.ylabel('Intensity (a.u.)')
            plt.legend()

            plt.subplot(plt_y, plt_x, i + plt_x + 1)
            if scale == 'Linear':
                plt.plot(motor2_pos, intensity, label='scan%d' % (scan_num))
            elif scale == 'Normalized':
                plt.plot(motor2_pos, intensity / np.amax(intensity), label='scan%d' % (scan_num))
            elif scale == 'Log':
                plt.plot(motor2_pos, np.log10(intensity + 1.0), label='scan%d' % (scan_num))
            plt.title(counter_name)
            plt.xlabel("%s" % motor2)
            plt.ylabel('Intensity (a.u.)')
            plt.legend()
    plt.show()

    for sample_name, scan_num in d2scan_scan_num:
        scan = HEPSScanImporter(beamline_mode, path, sample_name, scan_num, pathsavefolder)

        e4mcounters = [counter_name for counter_name in counter_select if 'e4m' in counter_name]
        if len(e4mcounters) > 0:
            plt.figure(figsize=(8, 8))
            img_sum = scan.get_imgsum(det_type='e4m')
            plt.imshow(np.log10(img_sum + 1.0), cmap='jet')
            for counter_name in e4mcounters:
                if counter_name != 'e4m_full':
                    draw_roi(scan.get_motor_pos(counter_name), counter_name)
            plt.title('scan%05d' % scan_num)
            plt.show()

        pilatuscounters = [counter_name for counter_name in counter_select if 'pilatus' in counter_name]
        if len(pilatuscounters) > 0:
            plt.figure(figsize=(8, 8))
            img_sum = scan.get_imgsum(det_type='pilatus')
            plt.imshow(np.log10(img_sum + 1.0), cmap='jet')
            for counter_name in e4mcounters:
                if counter_name != 'pilatus_full':
                    draw_roi(scan.get_motor_pos(counter_name), counter_name)
            plt.title('scan%05d' % scan_num)
            plt.show()

# Plot mesh scans
if len(mesh_scan_num) > 0:
    for sample_name, scan_num in mesh_scan_num:
        scan = HEPSScanImporter(beamline_mode, path, sample_name, scan_num, pathsavefolder)

        command_infor = scan.get_command_infor()
        # curpetra_ar = scan.get_scan_data('curpetra') / np.round(np.average(scan.get_scan_data('curpetra')))

        motor1 = command_infor['motor1_name']
        motor2 = command_infor['motor2_name']
        num_x = int(command_infor['motor1_step_num']) + 1
        num_y = int(command_infor['motor2_step_num']) + 1
        motor1_pos = scan.get_scan_data(motor1).reshape((num_y, num_x))
        motor2_pos = scan.get_scan_data(motor2).reshape((num_y, num_x))

        for counter_name in counter_select:
            intensity = (scan.get_scan_data(counter_name)).reshape((num_y, num_x))
            if counter_name[:3] == 'e4m':
                img_sum = scan.get_imgsum(det_type='e4m')
                plt.figure(figsize=(16, 8))
                plt.subplot(1, 2, 1)
                plt.imshow(np.log10(img_sum + 1.0))
                if counter_name != 'e4m_full':
                    draw_roi(scan.get_motor_pos(counter_name), counter_name)
                plt.subplot(1, 2, 2)
            elif counter_name[:4] == 'pilatus':
                img_sum = scan.get_imgsum(det_type='pilatus')
                plt.figure(figsize=(16, 8))
                plt.subplot(1, 2, 1)
                plt.imshow(np.log10(img_sum + 1.0))
                if counter_name != 'pilatus_full':
                    draw_roi(scan.get_motor_pos(counter_name), counter_name)
                plt.subplot(1, 2, 2)
            else:
                plt.figure(figsize=(8, 8))
            plt.contourf(motor1_pos, motor2_pos, np.log10(intensity + 1.0), 150, cmap="jet")
            plt.axis('scaled')
            plt.xlabel("%s (um)" % motor1)
            plt.ylabel("%s (um)" % motor2)
            plt.savefig(os.path.join(scan.get_pathsave(), '%s_scan%05d_%s.png' % (scan.get_sample_names(), scan_num, counter_name)))
            plt.show()
