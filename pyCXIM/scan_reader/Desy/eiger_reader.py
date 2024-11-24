# -*- coding: utf-8 -*-
"""
Read and treat the p10 scans with eiger detectors.
Created on Thu Apr 27 15:33:21 2023

@author: Ren Zhe
@email: renzhe@ihep.ac.cn
"""

import ast
import datetime
import hdf5plugin
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys
from scipy.ndimage import center_of_mass

from .fio_reader import DesyScanImporter
from ..general_detector import DetectorMixin


class DesyEigerImporter(DesyScanImporter, DetectorMixin):
    """
    Read and treat the scans with eiger detector. It is a child class of DesyScanImporter and DetectorMixin.

    Parameters
    ----------
    beamline : str
        The name of the beamline. Please chose between 'p10' and 'p08'.
    path : str
        The path for the raw file folder.
    sample_name : str
        The name of the sample defined by the p10_newfile name in the system.
    scan : int
        The scan number.
    detector : str, optional
        The name of the detecter, it can be either 'e4m', 'eiger1m' or 'e500'. The default is 'e4m'.
    pathsave : str, optional
        The folder to save the results, if not given no results will be saved. The default is ''.
    pathmask : str, optional
        The path of the detector mask. If not given, mask will be generated according to the hot pixels in the first image of the scan. The default is ''.
    creat_save_folder : bool, optional
        Whether the save folder should be created. The default is True.

    Raises
    ------
    IOError
        If the image folder for eiger detector does not exist, raise IOError.
    KeyError
        If the detector type is not previously registered, raise KeyError.

    """

    def __init__(self, beamline, path, sample_name, scan, detector='e4m', pathsave='', pathmask='', creat_save_folder=True):
        super().__init__(beamline, path, sample_name, scan, pathsave, creat_save_folder)
        self.detector = detector

        self.path_image_folder = os.path.join(self.path, self.detector)
        self.path_img = os.path.join(self.path_image_folder, "%s_%05d_data_%06d.h5")
        self.path_imgsum = os.path.join(self.pathsave, '%s_scan%05d_%s_imgsum.npy' % (self.sample_name, self.scan, self.detector))
        if not os.path.exists(self.path_image_folder):
            raise IOError('The image folder for %s images %s does not exist, please check the path again!' % (self.detector, self.path_image_folder))

        self.add_header_infor('detector')
        self.add_header_infor('path_image_folder')

        if self.beamline == 'p08':
            if self.detector == 'eiger1m':
                self.detector_size = (1062, 1028)
            else:
                raise KeyError('Detector type not registered! Please check the detector type or contact the author! Email: renzhe@ihep.ac.cn')
            self.pixel_size = 75e-3
            self.img_per_point = 'one'

        elif self.beamline == 'p10':
            if self.detector == 'e4m':
                self.detector_size = (2167, 2070)
            elif self.detector == 'e500':
                self.detector_size = (514, 1030)
            else:
                raise KeyError('Detector type not registered! Please check the detector type or contact the author! Email: renzhe@ihep.ac.cn')
            self.pixel_size = 75e-3

            if self.get_scan_type() == 'time_series':
                self.read_batchinfo()
                self.read_master_file()
                self.command = 'time_series %d %f' % (self.npoints - 1, self.count_time)

            if "%s_%05d_data_%06d.h5" % (self.sample_name, self.scan, self.npoints) in os.listdir(self.path_image_folder):
                self.img_per_point = 'multiple'
            else:
                self.img_per_point = 'one'
        self.load_mask(pathmask)
        return

    def load_single_image(self, img_index, correction_mode='constant'):
        """
        Read a single eiger image stored in h5 format.

        Parameters
        ----------
        img_index : int
            The index of the single image in the scan.
        correction_mode : str, optional
            If correction_mode is 'constant',intensity of the masked pixels will be corrected according to the img_correction array generated before.
            Most of the time, intensity of the masked pixels will be set to zero.
            However, for the semitransparent mask the intensity will be corrected according to the transmission.
            If the correction_mode is 'medianfilter', the intensity of the masked pixels will be set to the median filter value according the surrounding pixels.
            If the correction_mode is 'off', the intensity of the masked pixels will not be corrected.
            The default is 'constant'.

        Returns
        -------
        image : ndarray
            The image of the pilatus detector.

        """
        assert img_index < self.npoints, 'The image number wanted is larger than the total image number in the scan!'
        img_index = int(img_index)

        if self.img_per_point == 'one':
            pathimg = (self.path_img) % (self.sample_name, self.scan, img_index // 2000 + 1)
            with h5py.File(pathimg, "r") as f:
                dataset = f['entry/data/data']
                image = np.array(dataset[img_index % 2000, :, :], dtype=float)
        elif self.img_per_point == 'multiple':
            pathimg = (self.path_img) % (self.sample_name, self.scan, img_index + 1)
            with h5py.File(pathimg, "r") as f:
                dataset = f['entry/data/data']
                image = np.sum(f['entry/data/data'], axis=0)
        image = self.image_mask_correction(image, correction_mode=correction_mode)
        return image

    def read_batchinfo(self):
        """
        Read the batchinformation file.

        Returns
        -------
        None.

        """
        assert self.beamline == 'p10', \
            'Batchinfor is a special file format that is used for time series scans at P10 beamline, Desy.'
        self.path_batchinfo = os.path.join(self.path_image_folder, '%s_%05d.batchinfo' % (self.sample_name, self.scan))
        pattern1 = r'(\w+): (.+)\n'
        with open(self.path_batchinfo, 'r') as batchinfofile:
            self.batchinfo = dict(re.findall(pattern1, batchinfofile.read()))
        for parameter_name in self.batchinfo:
            try:
                self.batchinfo[parameter_name] = ast.literal_eval(self.batchinfo[parameter_name])
            except (ValueError, SyntaxError):
                self.batchinfo[parameter_name] = self.batchinfo[parameter_name]
        self.start_time = self.batchinfo['start_time']
        self.start_time = datetime.datetime.strptime(self.start_time, '%a %b %d %H:%M:%S %Y')
        self.npoints = int(self.batchinfo['ndataend'][0])
        cch = [0.0, 0.0]
        cch[0] = (self.batchinfo['y0'] - 1.0)
        cch[1] = (self.batchinfo['x0'] - 1.0)
        self.add_motor_pos('cch', cch)
        return

    def read_master_file(self):
        """
        Read the master file generated by the eiger detector.

        Returns
        -------
        None.

        """
        path_master_file = os.path.join(self.path_image_folder, r'%s_%05d_master.h5' % (self.sample_name, self.scan))
        with h5py.File(path_master_file, "r") as f:
            self.count_time = f['/entry/instrument/detector/count_time'][()]
            self.add_scan_infor('count_time')
            self.detector_readout_time = f['/entry/instrument/detector/detector_readout_time'][()]
            self.add_scan_infor('detector_readout_time')
            self.frame_time = f['/entry/instrument/detector/frame_time'][()]
            self.add_scan_infor('frame_time')
        return

    def load_time_series_single_pixel(self, xpos, ypos, start_frame=None, end_frame=None, mask_correction=True):

        if xpos < self.detector_size[1]:
            xpos = int(xpos)
        else:
            xpos = int(self.detector_size[1])

        if ypos < self.detector_size[0]:
            ypos = int(ypos)
        else:
            ypos = int(self.detector_size[0])

        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.npoints
        assert end_frame > start_frame, 'The end frame should always be larger than the start frame!'

        if self.get_scan_type() == 'time_series':
            pathimg = (self.path_eiger_img) % (self.sample_name, self.scan, 1)
            f = h5py.File(pathimg, "r")
            dataset = f['entry/data/data']
            time_series_data = np.array(dataset[start_frame:end_frame, ypos, xpos], dtype=float)
        else:
            time_series_data = np.array([])
            for i in range(self.npoints // 2000 + 1):
                pathimg = (self.path_eiger_img) % (self.sample_name, self.scan, i + 1)
                f = h5py.File(pathimg, "r")
                dataset = f['entry/data/data']
                time_series_data = np.append(time_series_data, np.array(dataset[:, ypos, xpos], dtype=float))
            time_series_data = time_series_data[start_frame:end_frame]

        if mask_correction:
            time_series_data = time_series_data * self.img_correction[ypos, xpos]
        return time_series_data

    def load_time_series_multiple_pixel(self, q_range_mask, start_frame=None, end_frame=None):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = self.npoints
        assert end_frame > start_frame, 'The end frame should always be larger than the start frame!'

        time_series_data = np.zeros((np.sum(q_range_mask), end_frame - start_frame))
        if self.get_scan_type() == 'time_series':
            pathimg = (self.path_eiger_img) % (self.sample_name, self.scan, 1)
            f = h5py.File(pathimg, "r")
            dataset = f['entry/data/data']
            for i in range(end_frame - start_frame):
                time_series_data[:, i] = np.array(dataset[i + start_frame, :, :], dtype=float)[q_range_mask]
        else:
            for i in range(start_frame, end_frame):
                pathimg = (self.path_eiger_img) % (self.sample_name, self.scan, i // 2000 + 1)
                f = h5py.File(pathimg, "r")
                dataset = f['entry/data/data']
                time_series_data[:, i - start_frame] = np.array(dataset[i - i // 2000 * 2000, :, :], dtype=float)[q_range_mask]

        return time_series_data

    def eiger_mask_sum(self, q_range_mask, start_frame=None, end_frame=None):

        if start_frame is None:
            start_frame = 0
        else:
            print(start_frame)
        if end_frame is None:
            end_frame = self.npoints
        assert end_frame > start_frame, 'The end frame should always be larger than the start frame!'

        int_ar = np.zeros(end_frame - start_frame)
        if self.get_scan_type() == 'time_series':
            pathimg = (self.path_eiger_img) % (self.sample_name, self.scan, 1)
            f = h5py.File(pathimg, "r")
            dataset = f['entry/data/data']
            for i in range(start_frame, end_frame):
                int_ar[i - start_frame] = np.average(np.array(dataset[i, :, :], dtype=float)[q_range_mask])
        else:
            for i in range(start_frame, end_frame):
                pathimg = (self.path_eiger_img) % (self.sample_name, self.scan, i // 2000 + 1)
                f = h5py.File(pathimg, "r")
                dataset = f['entry/data/data']
                int_ar[i - start_frame] = np.average(np.array(dataset[i - i // 2000 * 2000, :, :], dtype=float)[q_range_mask])
        return int_ar

    def load_6C_peak_infor(self, roi=None, cut_width=[50, 50]):
        """
        Load the motor positions of the six circle diffractometer.

        Parameters
        ----------
        roi : list, optional
            The region of interest. If not given, the complete detector image will be used. The default is None.
        cut_width : list, optional
            The cut width in Y, X direction. The default is [50, 50].

        Returns
        -------
        pixel_position : list
            The pixel position on the detector in [Y, X] order.
        motor_position : list
            motor positions in the order of [omega, delta, chi, phi, gamma, energy]. The angles are in degree and the energy in eV.

        """
        pch = self.image_find_peak_position(roi=roi, cut_width=cut_width)
        scan_motor = self.get_scan_motor()
        scan_motor_ar = self.get_scan_data(scan_motor)
        if self.beamline == 'p08':
            if scan_motor == 'om':
                omega = scan_motor_ar[int(pch[0])]
                phi = self.get_motor_pos('phis')
            elif scan_motor == 'phis':
                omega = self.get_motor_pos('om')
                phi = scan_motor_ar[int(pch[0])]
            delta = self.get_motor_pos('tt')
            chi = self.get_motor_pos('chi')
            gamma = self.get_motor_pos('tth')
            mu = self.get_motor_pos('omh')
            energy = self.get_motor_pos('energyfmb')
        elif self.beamline == 'p10':
            if scan_motor == 'om':
                omega = scan_motor_ar[int(pch[0])]
                phi = self.get_motor_pos('phi')
            elif scan_motor == 'phi':
                omega = self.get_motor_pos('om')
                phi = scan_motor_ar[int(pch[0])]
            delta = self.get_motor_pos('del')
            chi = self.get_motor_pos('chi')
            gamma = self.get_motor_pos('gam')
            mu = self.get_motor_pos('mu')
            energy = self.get_motor_pos('fmbenergy')

        motor_position = np.array([omega, delta, chi, phi, gamma, mu, energy], dtype=float)
        pixel_position = np.array([pch[1], pch[2]])
        return pixel_position, motor_position

    def ptycho_cxi(self, cen, cut_width, detector_distance=5000.0, index_array=None):
        """
        Convert the 2D Scans with Eiger 4M to the CXI format to generate the input for PyNX.

        Generated according to the PyNX package (http://ftp.esrf.fr/pub/scisoft/PyNX/doc/).

        Parameters
        ----------
        cen : list
            The central position of the direct beam on the detector in the [Y, X] form.
        cut_width : list
            The half width for cutting around the direct beam position.
        detector_distance : float, optional
            The detector distance in millimeter. The default is 5000.0.
        index_array : ndarray, optional
            1D boolen array indicating which images should be used for the ptychography calculation. The default is None.

        Returns
        -------
        None.

        """
        path_save_cxi = os.path.join(self.pathsave, '%s_%05d.cxi' % (self.sample_name, self.scan))
        print('Create cxi file: %s' % path_save_cxi)
        f = h5py.File(path_save_cxi, "w")
        f.attrs['file_name'] = path_save_cxi
        f.attrs['file_time'] = self.get_start_time().strftime("%Y-%m-%dT%H:%M:%S")
        f.attrs['creator'] = 'p10'
        f.attrs['HDF5_Version'] = h5py.version.hdf5_version
        f.attrs['h5py_version'] = h5py.version.version
        f.attrs['default'] = 'entry_1'
        f.create_dataset("cxi_version", data=140)

        entry_1 = f.create_group("entry_1")
        entry_1.create_dataset('start_time', data=self.get_start_time().strftime("%Y-%m-%dT%H:%M:%S"))
        entry_1.attrs['NX_class'] = 'NXentry'
        entry_1.attrs['default'] = 'data_1'

        sample_1 = entry_1.create_group("sample_1")
        sample_1.attrs['NX_class'] = 'NXsample'

        command_infor = self.get_command_infor()
        geometry_1 = sample_1.create_group("geometry_1")
        sample_1.attrs['NX_class'] = 'NXgeometry'  # Deprecated NeXus class, move to NXtransformations

        if index_array is None:
            xyz = np.zeros((3, self.npoints), dtype=np.float32)
            xyz[0] = self.get_scan_data(command_infor['motor1_name']) * 1.0e-6
            xyz[1] = self.get_scan_data(command_infor['motor2_name']) * 1.0e-6
        else:
            xyz = np.zeros((3, np.sum(index_array)), dtype=np.float32)
            xyz[0] = (self.get_scan_data(command_infor['motor1_name']) * 1.0e-6)[index_array]
            xyz[1] = (self.get_scan_data(command_infor['motor2_name']) * 1.0e-6)[index_array]
        geometry_1.create_dataset("translation", data=xyz)

        data_1 = entry_1.create_group("data_1")
        data_1.attrs['NX_class'] = 'NXdata'
        data_1.attrs['signal'] = 'data'
        data_1.attrs['interpretation'] = 'image'
        data_1["translation"] = h5py.SoftLink('/entry_1/sample_1/geometry_1/translation')

        instrument_1 = entry_1.create_group("instrument_1")
        instrument_1.attrs['NX_class'] = 'NXinstrument'
        instrument_1.create_dataset("name", data='six_circle_diffractometer')

        source_1 = instrument_1.create_group("source_1")
        source_1.attrs['NX_class'] = 'NXsource'
        # Convert the energy to J
        nrj = self.get_motor_pos('fmbenergy') * 1.60218e-19
        source_1.create_dataset("energy", data=nrj)
        source_1["energy"].attrs['note'] = 'Incident photon energy (instead of source energy), for CXI compatibility'

        detector_1 = instrument_1.create_group("detector_1")
        detector_1.attrs['NX_class'] = 'NX_detector'
        if index_array is None:
            iobs = detector_1.create_dataset("data", (self.npoints, cut_width[0] * 2, cut_width[1] * 2), chunks=(1, cut_width[0] * 2, cut_width[1] * 2), shuffle=True, compression="gzip")
        else:
            iobs = detector_1.create_dataset("data", (int(np.sum(index_array)), cut_width[0] * 2, cut_width[1] * 2), chunks=(1, cut_width[0] * 2, cut_width[1] * 2), shuffle=True, compression="gzip")
        cut_width = self.image_cut_check(cen, cut_width)
        petra_current = self.get_scan_data('curpetra')
        petra_current = petra_current / np.round(np.average(petra_current))
        if index_array is None:
            for i in range(self.npoints):
                image = self.load_single_image(i, correction_mode='constant')
                print(i)
                iobs[i, :, :] = image[(cen[0] - cut_width[0]):(cen[0] + cut_width[0]), (cen[1] - cut_width[1]):(cen[1] + cut_width[1])] / petra_current[i]
        else:
            for i, num in enumerate(np.arange(self.npoints)[index_array]):
                image = self.load_single_image(num, correction_mode='constant')
                print(num)
                iobs[i, :, :] = image[(cen[0] - cut_width[0]):(cen[0] + cut_width[0]), (cen[1] - cut_width[1]):(cen[1] + cut_width[1])] / petra_current[num]
        detector_1.create_dataset("distance", data=detector_distance * 1.0e-3)
        detector_1["distance"].attrs['units'] = 'm'
        detector_1.create_dataset("x_pixel_size", data=self.pixel_size * 1.0e-3)
        detector_1["x_pixel_size"].attrs['units'] = 'm'
        detector_1.create_dataset("y_pixel_size", data=self.pixel_size * 1.0e-3)
        detector_1["y_pixel_size"].attrs['units'] = 'm'
        mask_cut = self.mask[(cen[0] - cut_width[0]):(cen[0] + cut_width[0]), (cen[1] - cut_width[1]):(cen[1] + cut_width[1])]
        detector_1.create_dataset("mask", data=mask_cut, chunks=True, shuffle=True, compression="gzip")
        detector_1["mask"].attrs['note'] = "Mask of invalid pixels, applying to each frame"
        basis_vectors = np.zeros((2, 3), dtype=np.float32)
        basis_vectors[0, 1] = -self.pixel_size * 1.0e-3
        basis_vectors[1, 0] = -self.pixel_size * 1.0e-3
        detector_1.create_dataset("basis_vectors", data=basis_vectors)

        detector_1["translation"] = h5py.SoftLink('/entry_1/sample_1/geometry_1/translation')
        data_1["data"] = h5py.SoftLink('/entry_1/instrument_1/detector_1/data')

        process_1 = data_1.create_group("process_1")
        process_1.attrs['NX_class'] = 'NXprocess'
        process_1.create_dataset("program", data='PyNX')  # NeXus spec
        # process_1.create_dataset("version", data="%s" % __version__)  # NeXus spec
        # process_1.create_dataset("command", data=command)  # CXI spec
        config = process_1.create_group("configuration")
        config.attrs['NX_class'] = 'NXcollection'

        f.close()
        return
