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
from pyCXIM.scan_reader.Desy.eiger_reader import DesyEigerImporter
from pyCXIM.scan_reader.Desy.fio_reader import DesyScanImporter
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
scan_num_ar = [1, 9]
p10_file = ["LiNiMnO2_1_1"]

# path information
path = r"F:\Raw Data\20230925_P10_BFO_Pt_LiNiMnO2_AlScN\raw"
path_e4m_mask = r'F:\Work place 3\testprog\pyCXIM_master\detector_mask\p10_e4m_mask.npy'
path_e500_mask = r'E:\Work place 3\testprog\X-ray diffraction\Common functions\e500_mask.npy'
pathsavefolder = r"F:\Work place 4\sample\XRD\20230924_BFO_Pt_P10_Desy\LiNiMnO2"

# The rois for the Eiger 4M detector
e4m_roi1 = [100, 2000, 948, 1148]
e4m_roi2 = [100, 2000, 200, 500]
e4m_roi3 = [1630, 1660, 950, 1000]
e4m_roi4 = [400, 600, 300, 500]

cal_e4m_roi = [e4m_roi1, e4m_roi2]

# The rois for the Eiger500 detector
e500_roi1 = [300, 800, 100, 600]
e500_roi2 = [100, 300, 100, 600]
cal_e500_roi = []

# Fluo channels
fluo_05 = [1520, 1574]
fluo_06 = [1020, 1374]
cal_fluo_channels = []

# Plot selection
counter_select = ['e4m_roi1']
scale = 'Linear'
# scale = 'Normalized'
# scale = 'Log'

# %%Integrating the detector roi, write new fio and spec file
for i, scan_num in enumerate(scan_num_ar):
    if len(scan_num_ar) != len(p10_file):
        p10_newfile = p10_file[0]
    else:
        p10_newfile = p10_file[i]

    if len(cal_e4m_roi) != 0:
        scan = DesyEigerImporter('p10', path, p10_newfile, scan_num, 'e4m', pathsavefolder, path_e4m_mask)
        scan.eiger_roi_sum(cal_e4m_roi, roi_order='XY', save_img_sum=True)
        scan.write_fio()

    if len(cal_e500_roi) != 0:
        scan = DesyEigerImporter('p10', path, p10_newfile, scan_num, 'e500', pathsavefolder, path_e500_mask)
        scan.eiger_roi_sum(cal_e500_roi, roi_order='XY', save_img_sum=True)
        scan.write_fio()

    # if len(cal_fluo_channels) != 0:
    #     scan = P10FluoScan(path, p10_newfile, scan_num, pathsavefolder)
    #     fluo_spectrum = scan.fluo_spec_reader(cal_fluo_channels)
    #     scan.write_fio()
    #     plt.plot(fluo_spectrum)
    #     plt.show()

    if (len(cal_e4m_roi) == 0) and (len(cal_e500_roi) == 0) and (len(cal_fluo_channels) == 0):
        scan = DesyScanImporter('p10', path, p10_newfile, scan_num, pathsavefolder)

    print(scan)
    print('Following counters are available, please select at least one of them to plot:')
    print(scan.get_counter_names())
# beamtimeID=scan.get_motor_pos('_beamtimeID')
# spec_writer(beamtimeID, path, scan.get_p10_file(), pathsavefolder)

# %% Plot and save
assert len(counter_select) > 0, 'Please select at least one of the counters to plot!'
for counter in counter_select:
    assert (counter in scan.get_counter_names()), 'The counter %s does not exist! Please check the name of the counter again!' % counter
line_scan_num = []
mesh_scan_num = []
for i, scan_num in enumerate(scan_num_ar):
    if len(scan_num_ar) != len(p10_file):
        p10_newfile = p10_file[0]
    else:
        p10_newfile = p10_file[i]

    scan = DesyScanImporter('p10', path, p10_newfile, scan_num, pathsavefolder)

    if scan.get_scan_type() in ['ascan', 'dscan', 'a2scan', 'd2scan']:
        line_scan_num.append(scan_num)
    if scan.get_scan_type() in ['dmesh', 'mesh']:
        mesh_scan_num.append(scan_num)

# Plot line scans
if len(line_scan_num) > 0:
    plt_y = int(np.sqrt(len(counter_select)))
    plt_x = len(counter_select) // plt_y
    plt.figure(figsize=(8 * plt_x, 8 * plt_y))
    for i, scan_num in enumerate(line_scan_num):
        if len(scan_num_ar) != len(p10_file):
            p10_newfile = p10_file[0]
        else:
            p10_newfile = p10_file[i]

        scan = DesyScanImporter('p10', path, p10_newfile, scan_num, pathsavefolder)
        command_infor = scan.get_command_infor()
        curpetra_ar = scan.get_scan_data('curpetra') / np.round(np.average(scan.get_scan_data('curpetra')))
        motor = command_infor['motor2_name']
        motor_pos = scan.get_scan_data(motor)

        for i, counter_name in enumerate(counter_select):
            assert counter_name in scan.get_counter_names(), 'Counter %s does not exist, please check it again!' % counter_name
            intensity = scan.get_scan_data(counter_name) / curpetra_ar

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

    for i, scan_num in enumerate(line_scan_num):
        if len(scan_num_ar) != len(p10_file):
            p10_newfile = p10_file[0]
        else:
            p10_newfile = p10_file[i]
        scan = DesyScanImporter('p10', path, p10_newfile, scan_num, pathsavefolder)

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

        e500counters = [counter_name for counter_name in counter_select if 'e500' in counter_name]
        if len(e500counters) > 0:
            plt.figure(figsize=(8, 8))
            img_sum = scan.get_imgsum(det_type='e500')
            plt.imshow(np.log10(img_sum + 1.0), cmap='jet')
            for counter_name in e4mcounters:
                if counter_name != 'e500_full':
                    draw_roi(scan.get_motor_pos(counter_name), counter_name)
            plt.title('scan%05d' % scan_num)
            plt.show()

# Plot mesh scans
if len(mesh_scan_num) > 0:
    for i, scan_num in enumerate(mesh_scan_num):
        if len(scan_num_ar) != len(p10_file):
            p10_newfile = p10_file[0]
        else:
            p10_newfile = p10_file[i]
        scan = DesyScanImporter('p10', path, p10_newfile, scan_num, pathsavefolder)

        command_infor = scan.get_command_infor()
        curpetra_ar = scan.get_scan_data('curpetra') / np.round(np.average(scan.get_scan_data('curpetra')))

        motor1 = command_infor['motor1_name']
        motor2 = command_infor['motor2_name']
        num_x = int(command_infor['motor1_step_num']) + 1
        num_y = int(command_infor['motor2_step_num']) + 1
        motor1_pos = scan.get_scan_data(motor1).reshape((num_y, num_x))
        motor2_pos = scan.get_scan_data(motor2).reshape((num_y, num_x))

        for counter_name in counter_select:
            intensity = (scan.get_scan_data(counter_name) / curpetra_ar).reshape((num_y, num_x))
            if counter_name[:3] == 'e4m':
                img_sum = scan.get_imgsum(det_type='e4m')
                plt.figure(figsize=(16, 8))
                plt.subplot(1, 2, 1)
                plt.imshow(np.log10(img_sum + 1.0))
                if counter_name != 'e4m_full':
                    draw_roi(scan.get_motor_pos(counter_name), counter_name)
                plt.subplot(1, 2, 2)
            elif counter_name[:4] == 'e500':
                img_sum = scan.get_imgsum(det_type='e500')
                plt.figure(figsize=(16, 8))
                plt.subplot(1, 2, 1)
                plt.imshow(np.log10(img_sum + 1.0))
                if counter_name != 'e500_full':
                    draw_roi(scan.get_motor_pos(counter_name), counter_name)
                plt.subplot(1, 2, 2)
            else:
                plt.figure(figsize=(8, 8))
            plt.contourf(motor1_pos, motor2_pos, np.log10(intensity + 1.0), 150, cmap="jet")
            plt.axis('scaled')
            plt.xlabel("%s (um)" % motor1)
            plt.ylabel("%s (um)" % motor2)
            plt.savefig(os.path.join(scan.get_pathsave(), '%s_scan%05d_%s.png' % (scan.get_p10_file(), scan_num, counter_name)))
            plt.show()
