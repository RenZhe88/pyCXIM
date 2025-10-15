# -*- coding: utf-8 -*-
"""
Treat and plot the theta-2theta scans.


Created on Tue Jul 30 16:26:36 2024

@author: renzh
"""

import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import sys
import os
import time
import gc
sys.path.append(r'E:\Work place 3\testprog\pyCXIM_master')
from pyCXIM.scan_reader.Desy.eiger_reader import DesyEigerImporter
from pyCXIM.scan_reader.Desy.fio_reader import DesyScanImporter
from pyCXIM.Common.Information_file_generator import InformationFileIO
from pyCXIM.RSM.TT2RSM import det2q_2D

def theta2theta_analyser(scan_num):
    # %% Input
    # scan_num = 11
    p10_newfile = "WTY_P3_ALF"
    rebinfactor = 2.0

    # ['Check mask', 'GIF', 'Check results', 'Theta2theta']
    Functions_selected = ['GIF', 'Theta2theta']

    # path information
    path = r"F:\Raw Data\20250506_P10_In_situ_battery_test_01\raw"
    path_e4m_mask = r'F:\Work place 3\testprog\pyCXIM_master\detector_mask\p10_e4m_mask.npy'
    pathsavefolder = r"F:\Work place 4\sample\XRD\20250506_in_situ_battery_P10_Desy\results\LXD\WTY_P3_ALF"
    path_calib = r'F:\Work place 4\sample\XRD\20250506_in_situ_battery_P10_Desy\results\calibration.txt'

    delta_offset = 0.0
    energy_offset = 0.0
    # %% reading scan information
    calibinfor = InformationFileIO(path_calib)
    calibinfor.infor_reader()
    cch = calibinfor.get_para_value('direct_beam_position', section='Detector calibration')
    distance = calibinfor.get_para_value('detector_distance', section='Detector calibration')
    pixelsize = calibinfor.get_para_value('pixelsize', section='Detector calibration')
    det_rot = calibinfor.get_para_value('detector_rotation', section='Detector calibration')

    scan = DesyEigerImporter('p10', path, p10_newfile, scan_num, 'e4m', pathsavefolder, path_e4m_mask)
    print(scan)
    detector_size = scan.get_detector_size()
    command_infor = scan.get_command_infor()
    delta_pos = scan.get_scan_data('del')
    gamma = scan.get_motor_pos('gam')
    energy = scan.get_motor_pos('fmbenergy')

    delta_pos += delta_offset
    energy += energy_offset

    pathsave = scan.get_pathsave()
    infor = InformationFileIO(os.path.join(pathsave, 'theta2theta_infor.txt'))
    section_ar = ['General Information', 'Detector calibration', 'Offsets_applied', 'Scan Information']
    infor.add_para('scan_num', section_ar[0], scan_num)
    infor.add_para('sample_name', section_ar[0], p10_newfile)
    infor.add_para('path', section_ar[0], path)
    infor.add_para('path_calib', section_ar[0], path_calib)
    infor.add_para('start_time', section_ar[0], scan.start_time.strftime('%a %b %d %H:%M:%S %Y'))

    infor.add_para('direct_beam_position', section_ar[1], cch)
    infor.add_para('detector_distance', section_ar[1], distance)
    infor.add_para('pixelsize', section_ar[1], pixelsize)
    infor.add_para('detector_rotation', section_ar[1], det_rot)
    infor.add_para('energy', section_ar[1], energy)

    infor.add_para('delta_offset', section_ar[2], delta_offset)
    infor.add_para('energy_offset', section_ar[2], energy_offset)

    # # %% Plot for mask determination
    image_sum = scan.image_sum()
    scan.add_mask_inverse_circle([1035, 1310], 1240)

    if 'Check mask' in Functions_selected:
        mask_for_plot = scan.get_mask_for_plot()
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(np.log10(image_sum + 1.0), cmap='jet')
        plt.subplot(1, 2, 2)
        plt.imshow(np.log10(image_sum + 1.0), cmap='Blues')
        plt.imshow(mask_for_plot, cmap='Reds', alpha=0.2, vmax=1.0, vmin=0)
        plt.show()

    # %% loadall the images into memory
    data, mask_3D, pch, roi = scan.load_rois(normalize_signal='curpetra', correction_mode='constant')
    del mask_3D, pch, roi

    # %% Generate Gif
    if 'GIF' in Functions_selected:
        starting_time = time.time()
        fig = plt.figure(figsize=(6, 6))
        plt.axis("off")
        img_frames = []
        for i in range(scan.get_num_points()):
            img = data[i, :, :]
            plt_im = plt.imshow(np.log10(img + 1.0), cmap='jet')
            img_frames.append([plt_im])
        fig.tight_layout()
        gif_img = anim.ArtistAnimation(fig, img_frames)
        pathsavegif = os.path.join(scan.get_pathsave(), "%s_%05d.gif" % (p10_newfile, scan_num))
        gif_img.save(pathsavegif, writer='pillow', fps=5)
        print('GIF image saved')
        plt.close()
        ending_time = time.time()
        print('time to generate gif images： %.2f' % (ending_time - starting_time))

    # %% Calculate the theta 2theta scan
    starting_time = time.time()
    mask = scan.get_mask()
    mask = (mask == 0)
    hc = 1.23984 * 10000.0
    wavelength = hc / energy
    unit = 2 * np.pi * pixelsize / wavelength / distance
    unit = unit * rebinfactor

    delta = delta_pos[0]
    detector2q = det2q_2D([delta, gamma, energy], [distance, pixelsize, det_rot, cch], detector_size)
    qmin = np.amin(detector2q[mask])

    delta = delta_pos[-1]
    detector2q = det2q_2D([delta, gamma, energy], [distance, pixelsize, det_rot, cch], detector_size)
    qmax = np.amax(detector2q[mask])

    q_val = np.arange(qmin, qmax, unit)
    int_sum = np.zeros(len(q_val))
    pixel_sum = np.zeros(len(q_val))

    infor.add_para('unit', section_ar[3], unit)
    infor.add_para('rebinfactor', section_ar[3], rebinfactor)
    infor.add_para('qmin', section_ar[3], qmin)
    infor.add_para('qmax', section_ar[3], qmax)
    infor.add_para('q_num', section_ar[3], len(q_val))

    for i in range(scan.get_num_points()):
        delta = delta_pos[i]
        image = data[i, :, :]
        detector2q = det2q_2D([delta, gamma, energy], [distance, pixelsize, det_rot, cch], detector_size)
        q_indice = np.digitize(detector2q, bins=q_val) - 1
        q_indice = q_indice[mask]
        image = image[mask]
        int_sum += np.bincount(q_indice, weights=image, minlength=len(q_val))
        pixel_sum += np.bincount(q_indice, minlength=len(q_val))

        sys.stdout.write('\rprogress:%d%%' % ((i + 1) * 100.0 / scan.get_num_points()))
        sys.stdout.flush()

    q_val = q_val + unit / 2.0
    real_delta = np.rad2deg(np.arcsin(q_val * wavelength / 4 / np.pi)) * 2.0
    lab_wavelength = 1.540562
    lab_delta = np.rad2deg(np.arcsin(q_val * lab_wavelength / 4 / np.pi)) * 2.0
    result_ar = np.stack((q_val, real_delta, lab_delta, int_sum, pixel_sum, int_sum / pixel_sum)).T
    ending_time = time.time()
    print('')
    print('time to calculate theta2theta scan： %.2f' % (ending_time - starting_time))

    np.savetxt(os.path.join(pathsave, 'theta2theta.txt'), result_ar, header='q_value    2theta    lab2theta    total_intensity    number_of_pixels     average_intensity')
    infor.infor_writer()
    if 'Check results' in Functions_selected:
        int_ar = scan.get_scan_data('e4m_roi1')
        plt.figure(figsize=(24, 8))
        plt.subplot(1, 3, 1)
        plt.plot(int_ar)
        plt.xlabel(r'image number')
        plt.ylabel(r'intensity')
        plt.subplot(1, 3, 2)
        plt.plot(q_val, pixel_sum)
        plt.xlabel(r'q ($1/\AA$)')
        plt.ylabel(r'number of pixels')
        plt.subplot(1, 3, 3)
        plt.plot(q_val, int_sum)
        plt.xlabel(r'q ($1/\AA$)')
        plt.ylabel(r'intensity sum')
        plt.show()
    if 'Theta2theta' in Functions_selected:
        plt.figure(figsize=(24, 8))
        plt.subplot(1, 3, 1)
        plt.plot(q_val, int_sum / pixel_sum)
        plt.title(r'q vs intensity')
        plt.xlabel(r'q ($1/\AA$)')
        plt.ylabel(r'average intensity (a.u.)')
        plt.subplot(1, 3, 2)
        plt.plot(real_delta, int_sum / pixel_sum)
        plt.title(r'theta-2theta @ %.1f eV' % energy)
        plt.xlabel(r'2theta angle (degree)')
        plt.ylabel(r'average intensity (a.u.)')
        plt.subplot(1, 3, 3)
        plt.plot(lab_delta, int_sum / pixel_sum)
        plt.title(r'theta-2theta @ 8978.9 eV')
        plt.xlabel(r'2theta angle (degree)')
        plt.ylabel(r'average intensity (a.u.)')
        plt.savefig(os.path.join(pathsave, 'theta-2theta scan results.jpg'))
        plt.close()
        # plt.show()
    return

if __name__ == "__main__":
    for i in range(6, 7):
        theta2theta_analyser(i)
        gc.collect()
