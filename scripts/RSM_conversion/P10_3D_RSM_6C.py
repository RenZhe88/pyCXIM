# -*- coding: utf-8 -*-
"""

    
In case of questions, please contact me. Detailed explaination file of the code could be sent upon requist.
Author: Ren Zhe
Date: 2020/12/04
Email: zhe.ren@desy.de or renzhetu001@gmail.com
"""

import os
import numpy as np
import sys
import time
sys.path.append(r'E:\Work place 3\testprog\pyCXIM_master')
from pyCXIM.Common.Information_file_generator import InformationFileIO
from pyCXIM.p10_scan_reader.p10_eiger_reader import P10EigerScan
from pyCXIM.RSM.RC2RSM import RC2RSM_6C
import pyCXIM.RSM.RSM_post_processing as RSM_post_processing

start_time = time.time()
# Inputs: general information
year = "2022"
beamtimeID = "11013631"
p10_file = r"PTO_STO_DSO_28"
scan_num = 33
detector = 'e4m'
geometry = 'out_of_plane'

# Roi on the detector [Ymin, Ymax, Xmin, Xmax]
roi = [584, 1384, 1245, 1845]

# distance of the detector
distance = 1829.0518805745921
pixelsize = 0.075
# Direct beam position on the detector Y, X
cch = [1046, 1341]
det_rot = 0.6579914120476287
omega_offset = -0.11212434100115953
delta_offset = -0.05317783930952746
chi_offset = -0.9056682493438566
phi_offset = 3.8444261387711585
gam_offset = 0
energy_offset = 0

# Inputs: reciprocal space box size in pixels
save_full_3D_RSM = False
generating_3D_vtk_file = False

# Inputs: paths
path = r"E:\Data2\XRD raw\20220608 P10 PTO BFO\raw"
pathsave = r"E:\Work place 3\sample\XRD\Test"
pathmask = r'E:\Work place 3\testprog\X-ray diffraction\Common functions\general_mask.npy'

print("#################")
print("Basic information")
print("#################")

# Read images and fio files
scan = P10EigerScan(path, p10_file, scan_num, detector, pathsave, pathmask)
print(scan)
# Generate the paths for saving the data
pathsave = scan.get_pathsave()
pathinfor = os.path.join(pathsave, "scan_%04d_information.txt" % scan_num)
path3dintensity = os.path.join(pathsave, "scan%04d.npz" % scan_num)
path3dmask = os.path.join(pathsave, "scan%04d_mask.npz" % scan_num)


dataset, mask3D, pch, roi = scan.eiger_load_rois(roi=roi, show_cen_image=(not os.path.exists(pathinfor)))
if geometry == 'out_of_plane':
    scan_motor_ar = scan.get_scan_data('om') + omega_offset
elif geometry == 'in_plane':
    scan_motor_ar = scan.get_scan_data('phi') + phi_offset
scan_step = (scan_motor_ar[-1] - scan_motor_ar[0]) / (len(scan_motor_ar) - 1)
omega = scan.get_motor_pos('om') + omega_offset
delta = scan.get_motor_pos('del') + delta_offset
chi = scan.get_motor_pos('chi') - 90.0 + chi_offset
phi = scan.get_motor_pos('phi') + phi_offset
gamma = scan.get_motor_pos('gam') + gam_offset
energy = scan.get_motor_pos('fmbenergy') + energy_offset
scan.write_fio()

RSM_converter = RC2RSM_6C(scan_motor_ar, geometry, omega, delta, chi, phi, gamma, det_rot, energy, distance, pixelsize)

# determining the rebin parameter
rebinfactor = RSM_converter.cal_rebinfactor()

# Finding the maximum peak position
print("peak at omega = %.2f, delta = %.2f, chi = %.2f, phi = %.2f, gamma = %.2f" % (omega, delta, chi, phi, gamma))

# writing the scan information to the aimed file
section_ar = ['General Information', 'Paths', 'Scan Information', 'Routine1: Reciprocal space map']
infor = InformationFileIO(pathinfor)
infor.add_para('command', section_ar[0], scan.get_command())
infor.add_para('year', section_ar[0], year)
infor.add_para('beamtimeID', section_ar[0], beamtimeID)
infor.add_para('p10_newfile', section_ar[0], p10_file)
infor.add_para('scan_number', section_ar[0], scan_num)

infor.add_para('path', section_ar[1], path)
infor.add_para('pathsave', section_ar[1], pathsave)

infor.add_para('pathinfor', section_ar[1], pathinfor)
infor.add_para('path3dintensity', section_ar[1], path3dintensity)
infor.add_para('pathmask', section_ar[1], pathmask)
infor.add_para('path3dmask', section_ar[1], path3dmask)

infor.add_para('roi', section_ar[2], list(roi))
infor.add_para('peak_position', section_ar[2], list(pch))
infor.add_para('omega', section_ar[2], omega)
infor.add_para('delta', section_ar[2], delta)
infor.add_para('chi', section_ar[2], chi)
infor.add_para('phi', section_ar[2], phi)
infor.add_para('gamma', section_ar[2], gamma)
infor.add_para('det_rot', section_ar[2], det_rot)
infor.add_para('omega_offset', section_ar[2], omega_offset)
infor.add_para('delta_offset', section_ar[2], delta_offset)
infor.add_para('chi_offset', section_ar[2], chi_offset)
infor.add_para('phi_offset', section_ar[2], phi_offset)
infor.add_para('gam_offset', section_ar[2], gam_offset)
infor.add_para('energy_offset', section_ar[2], energy_offset)
infor.add_para('scan_step', section_ar[2], scan_step)
infor.add_para('direct_beam_position', section_ar[2], list(cch))
infor.add_para('detector_distance', section_ar[2], distance)
infor.add_para('pixelsize', section_ar[2], pixelsize)
infor.add_para('geometry', section_ar[2], geometry)
infor.add_para('detector', section_ar[2], detector)

infor.infor_writer()

print("")
print("##################")
print("Generating the RSM")
print("##################")

# calculate the qx, qy, qz ranges of the scan
q_origin, new_shape, RSM_unit = RSM_converter.cal_q_range(roi, cch, rebinfactor=rebinfactor)

# generate the 3D reciprocal space map
print('Calculating intensity...')
RSM_int = RSM_converter.RSM_conversion(dataset, new_shape, rebinfactor=rebinfactor, cval=0, prefilter=True)
del dataset
qmax = np.array([np.argmax(np.sum(RSM_int, axis=(1, 2))), np.argmax(np.sum(RSM_int, axis=(0, 2))), np.argmax(np.sum(RSM_int, axis=(0, 1)))], dtype=int)


print('Calculating the mask...')
RSM_mask = RSM_converter.RSM_conversion(mask3D, new_shape, rebinfactor=rebinfactor, cval=1, prefilter=True)
del mask3D

if save_full_3D_RSM:
    print('saving 3D RSM and the corresponding mask')
    filename = "%s_%05d_RSM.npz" % (p10_file, scan_num)
    pathsaveRSM = os.path.join(pathsave, filename)
    np.savez_compressed(pathsaveRSM, data=RSM_int)
    infor.add_para('pathRSM', section_ar[1], pathsaveRSM)

    filename = "%s_%05d_RSM_mask.npz" % (p10_file, scan_num)
    pathsaveRSMmask = os.path.join(pathsave, filename)
    np.savez_compressed(pathsaveRSMmask, data=RSM_mask)
    infor.add_para('pathRSMmask', section_ar[1], pathsaveRSMmask)

# if generating_3D_vtk_file:
#     filename="scan%04d_fast_cubic.vtk"%scan
#     pathsavevtk=os.path.join(pathtmp, filename)
#     numpy2vtk(pathsavevtk, np.log10(RSM_int+1.0))

# Generate the images of the reciprocal space map
print('Generating the images of the RSM')
pathsavetmp = os.path.join(pathsave, 'scan%04d_integrate' % scan_num + '_%s.png')
RSM_post_processing.plot_with_units(RSM_int, q_origin, RSM_unit, pathsavetmp)
pathsavetmp = os.path.join(pathsave, 'scan%04d' % scan_num + '_%s.png')
RSM_post_processing.plot_with_units(RSM_int, q_origin, RSM_unit, pathsavetmp, qmax=qmax)

# save the information
infor.add_para('RSM_shape', section_ar[3], list(new_shape))
infor.add_para('rebinfactor', section_ar[3], rebinfactor)
infor.add_para('RSM_unit', section_ar[3], RSM_unit)
infor.add_para('q_origin', section_ar[3], q_origin)
end_time = time.time()
infor.add_para('total_time', section_ar[3], end_time - start_time)
# infor.add_para('qmax', section_ar[3], qmax)
infor.infor_writer()
