#python3.7.4 -tttt
"""
The code generate 3D reciprocal space map from a typical forward CDI study at P10.

Example command: e4mdscan sprz -5 185 380 10 0 0.2.

Input:
    
Output:
    1. 3D intensity distribution saved in the form of numpy
    2. The 3D mask
    3. qx, qy, and qz cuts of the diffraction pattern
    
In case of questions, please contact me.
Author: RZ
Date: 2022/07/22
Email: zhe.ren@desy.de or renzhetu001@gmail.com
"""

import os
import numpy as np
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
import sys
sys.path.append(r'E:\Work place 3\testprog\pyCXIM_master')
from pyCXIM.Common.Information_file_generator import Information_file_io
from pyCXIM.p10_scan_reader.p10_eiger_reader import p10_eiger_scan
import pyCXIM.RSM.RSM_post_processing as RSM_post_processing
from pyCXIM.RSM.CDI2RSM import CDI2RSM


# Inputs: general information
year = "2022"
beamtimeID = "11013629"
p10_file = r"O32_1"
scan = 34

# Inputs:Detector parameters
detector = 'e4m'
distance = 5001
pixelsize = 0.075
# Direct beam position on the detector Y, X
cch = [1337, 1456]
wxy = [400, 400]

# Inputs: reciprocal space box size in pixels
RSM_bs = [150, 150, 150]

# Inputs: paths
# The path for the scan folder
path = r"E:\Data2\XRD raw\202203232 test_data\raw"
# The path to save the results
pathsave = r"E:\Work place 3\sample\XRD\Test\Gerard CDI"
pathmask = r'E:\Work place 3\testprog\X-ray diffraction\Common functions\general_mask.npy'

# Loading scan information
print("Loading scan information...")
scan_data = p10_eiger_scan(path, p10_file, scan, detector, pathsave, pathmask)
pathsave = scan_data.get_pathsave()
pathinfor = os.path.join(pathsave, "scan_%04d_information.txt" % scan)

wxy = scan_data.eiger_cut_check(cch, wxy)
roi = [cch[0] - wxy[0], cch[0] + wxy[0], cch[1] - wxy[1], cch[1] + wxy[1]]
if not os.path.exists(pathinfor):
    img = scan_data.eiger_img_sum(sum_img_num=20)
    plt.imshow(np.log10(img + 1.0), cmap='jet')
    plt.show()
scan_data.eiger_mask_circle(cch, 7)
dataset, mask3D, pch, roi = scan_data.eiger_load_rois(roi)

# reading the omega values for each step in the rocking curve
phi_ar = scan_data.get_scan_data('abrz')
energy = scan_data.get_motor_pos('fmbenergy')
RSM_converter = CDI2RSM(phi_ar, wxy[0], wxy[1], energy, distance, pixelsize)
rebinfactor = RSM_converter.cal_rebinfactor()
rebinfactor = 2
print("The chosen rebin factor is: %f" % (rebinfactor))

zd, yd, xd = dataset.shape
# Calculate the new coordinates
Npos = RSM_converter.cartesian2polar(rebinfactor)

# load the spec files and generate the 3D reciprocal space map
print("Calculating the final intensity....")
intensityfinal = RSM_converter.grid_cdi(dataset, Npos, rebinfactor, cval=0)
del dataset

print("Calculating the final mask....")
maskfinal = RSM_converter.grid_cdi(mask3D, Npos, rebinfactor, cval=1)
del mask3D
maskfinal[maskfinal >= 0.1] = 1
maskfinal[maskfinal < 0.1] = 0


nz, ny, nx = intensityfinal.shape
# save the qx qy qz cut of the 3D intensity
print("Saving the qx qy qz cuts......")
unit = RSM_converter.get_RSM_unit(rebinfactor)
q_origin = np.array([-nz / 2, -ny / 2, -nx / 2]) * unit
pathsavetmp = os.path.join(pathsave, "scan%04d_integrate" % scan + "_%s.png")
RSM_post_processing.plot_with_units(intensityfinal, q_origin, unit, pathsavetmp)
qmax = np.array([nz / 2, ny / 2, nx / 2], dtype=int)
pathsavetmp = os.path.join(pathsave, "scan%04d_cut" % scan + "_%s.png")
RSM_post_processing.plot_with_units(intensityfinal, q_origin, unit, pathsavetmp, qmax)

RSM_bs = np.array(np.amin([qmax, RSM_bs], axis=0), dtype=int)
intensity_cut, pch, RSM_bs = RSM_post_processing.Cut_central(intensityfinal, RSM_bs, cut_mode='given', peak_pos=qmax)
del intensityfinal
mask_cut, pch, RSM_bs = RSM_post_processing.Cut_central(maskfinal, RSM_bs, cut_mode='given', peak_pos=qmax)
del maskfinal

print("saving the RSM cut for pynx...")
# Creating the aimed folder
pathtmp = os.path.join(pathsave, "pynxpre")
if not os.path.exists(pathtmp):
    os.mkdir(pathtmp)
pathsavetmp = os.path.join(pathtmp, "scan%04d" % scan + "_%s.png")
RSM_post_processing.plot_without_units(intensity_cut, mask_cut, pathsavetmp)
path3dint = os.path.join(pathtmp, "scan%04d.npz" % scan)
np.savez_compressed(path3dint, data=intensity_cut)
path3dmask = os.path.join(pathtmp, "scan%04d_mask.npz" % scan)
np.savez_compressed(path3dmask, data=mask_cut)

# writing the scan information to the aimed file
section_ar = ['General Information', 'Paths', 'Scan Information']
infor = Information_file_io(pathinfor)
infor.add_para('command', section_ar[0], scan_data.get_command())
infor.add_para('year', section_ar[0], year)
infor.add_para('beamtimeID', section_ar[0], beamtimeID)
infor.add_para('p10_newfile', section_ar[0], p10_file)
infor.add_para('scan_number', section_ar[0], scan)

infor.add_para('path', section_ar[1], path)
infor.add_para('pathsave', section_ar[1], pathsave)

infor.add_para('pathinfor', section_ar[1], pathinfor)
infor.add_para('path3dintensity', section_ar[1], path3dint)
infor.add_para('pathmask', section_ar[1], pathmask)
infor.add_para('path3dmask', section_ar[1], path3dmask)

infor.add_para('roi', section_ar[2], list(roi))
infor.add_para('direct_beam_position', section_ar[2], list(cch))
infor.add_para('detector_distance', section_ar[2], distance)
infor.add_para('pixelsize', section_ar[2], pixelsize)
infor.add_para('energy', section_ar[2], energy)
infor.add_para('RSM_unit', section_ar[2], unit)
infor.add_para('pynx_box_size', section_ar[2], list(RSM_bs))

infor.add_para('roi_width', section_ar[2], list(wxy))

infor.infor_writer()
