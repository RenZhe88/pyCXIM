# -*- coding: utf-8 -*-
"""
Description
Created on Tue Apr 25 14:51:06 2023

@author: renzhe
"""
import numpy as np
from scipy.ndimage import affine_transform
import sys


class RC2RSM_2C(object):
    """
    Calculate the reciprocal space map for rocking curve with just theta and 2theta motor considered.

    z direction is surface direction of the sample space.
    x direction is the along the beam.
    y direction is perpendicular to the diffraction plane.

    Parameters
    ----------
    scan_motor_ar : list
        The motor values in the rocking curve, which should linearly change during the scan.
    two_theta : float
        The two theta angle for the rocking curve.
    energy : float, optional
        The energy value in eV. The default is 8000.
    distance : float, optional
        The detector distance value in mm. The default is 1830.
    pixelsize : float, optional
        The pixelsize of the detector in mm. The default is 0.075.

    Returns
    -------
    None.

    """

    def __init__(self, scan_motor_ar, two_theta, energy=8000, distance=1830, pixelsize=0.075):
        self.scan_motor_ar = np.deg2rad(scan_motor_ar)
        self.scan_step = np.deg2rad((scan_motor_ar[-1] - scan_motor_ar[0]) / (len(scan_motor_ar) - 1))
        self.two_theta = np.deg2rad(two_theta)
        self.distance = distance
        self.pixelsize = pixelsize
        self.energy = energy

        hc = 1.23984 * 10000.0
        wavelength = hc / self.energy
        self.units = 2 * np.pi * self.pixelsize / wavelength / self.distance
        return

    def get_RSM_unit(self, rebinfactor=1):
        """
        Get the unit of the RSM.

        The unit would be inverse angstrom.

        Parameters
        ----------
        rebinfactor : float optional
            The rebin factor of the RSM. The default is 1.

        Returns
        -------
        float
            The units of the reciprocal space map in inverse angstrom.

        """
        return (self.units * rebinfactor)

    def cal_rel_q_pos(self, pch, cch):
        """
        Calculate the relative q values of a detector pixel.

        The unit corresponds to the detector pixelsize in reciprocal space.

        Parameters
        ----------
        pch : list
            The position of the detector pixels.
            pch[0] corresponds to the frame number in the rocking curve
            pch[1] corresponds to the peak position Y on the detector
            pch[2] corresponds to the peak position X on the detector
        cch : list
            The position of the direct beam position on the detector in [Y, X] form.

        Returns
        -------
        q_vector : list
            The calculated q_vector as [qz, qy, qx] in inverse angstrom

        """
        self.peak_angle = self.scan_motor_ar[int(pch[0])]
        q_vector = np.array([(cch[0] - pch[1]), (cch[1] - pch[2]), self.distance / self.pixelsize])
        delta_transform = np.array([[np.cos(self.two_theta), 0, np.sin(self.two_theta)], [0, 1, 0], [-np.sin(self.two_theta), 0, np.cos(self.two_theta)]])
        q_vector = np.dot(delta_transform, q_vector)
        q_vector = q_vector - np.array([0, 0, self.distance / self.pixelsize])
        omega_transform = np.array([[np.cos(self.peak_angle), 0, -np.sin(self.peak_angle)], [0, 1, 0], [np.sin(self.peak_angle), 0, np.cos(self.peak_angle)]])
        q_vector = np.dot(omega_transform, q_vector)
        return q_vector

    def cal_rebinfactor(self):
        """
        Calculate the recommended rebin factors for the RSM.

        Returns
        -------
        rebinfactor : int
            The rebin factor recommended for the reciprocal space map.

        """
        rebinfactor = np.abs(2.0 * np.sin(self.two_theta / 2.0) * self.distance / self.pixelsize * self.scan_step)

        print("rebinfactor calculated:%f" % rebinfactor)
        if rebinfactor > 1.5:
            rebinfactor = int(round(rebinfactor))
            print('Maybe consider increasing the step numbers in the scan.')
        else:
            rebinfactor = 1
            print('The number of steps for the scan shoulde be fine.')
        return rebinfactor

    def cal_transformation_matrix(self, rebinfactor=1):
        """
        Calculte the corresponding transformation matrix between the detector coordinates and the sample coordinates.

        The unit corresponds to the detector pixelsize in reciprocal space.

        Parameters
        ----------
        rebinfactor : flaot, optional
            The rebin factor of the RSM. The default is 1.

        Returns
        -------
        transformation_matrix : ndarray
            The transofmration matrix for the RSM conversion.

        """
        step_C = self.distance * self.scan_step / self.pixelsize
        transformation_matrix = np.array([[(np.cos(self.peak_angle) - np.cos(self.two_theta - self.peak_angle)) * step_C, -np.cos(self.two_theta - self.peak_angle)],
                                          [(np.sin(self.two_theta - self.peak_angle) + np.sin(self.peak_angle)) * step_C, np.sin(self.two_theta - self.peak_angle)]])
        transformation_matrix = transformation_matrix / rebinfactor
        return transformation_matrix

    def cal_q_range(self, roi, cch, om_range=None, rebinfactor=1):
        """
        Calculate the q range of the converted RSM.

        Generate the origin of the q space.
        The shape of the RSM after conversion.
        The unit of the RSM.

        Parameters
        ----------
        roi : list
            The region of interest in [Ymin, Ymax, Xmin, Xmax] form.
        cch : list
            The direct beam position on the detector in [Y, X] form.
        scan_range : list, optional
            The index of the scan motor defining the range of the scan motor values.
            If not given, the complete range of the scanning motor will be used.
            The default is None.
        rebinfactor : float, optional
            The rebin factor value of the reciprocal space. The default is 1.

        Returns
        -------
        q_origin : list
            The minimum value of the q space.
        new_shape : list
            The shape of the new reciprocal space map.
        RSM_unit : float
            The unit of the result reciprocal space.

        """
        corners_q = np.zeros((8, 3))
        if om_range is None:
            om_range = [0, -1]
        corner_num = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    corners_q[corner_num, :] = self.cal_rel_q_pos([om_range[i], roi[j], roi[k + 2]], cch)
                    corner_num += 1
        q_origin = np.amin(corners_q, axis=0) * self.units
        new_shape = (np.ptp(corners_q, axis=0) / rebinfactor).astype(int)
        RSM_unit = self.units * rebinfactor

        print("number of points for the reciprocal space:")
        print(" qz  qy  qx")
        print(new_shape)
        return q_origin, new_shape, RSM_unit

    def RSM_conversion(self, dataset, new_shape, rebinfactor=1, cval=0, prefilter=False):
        """
        Convert the dataset to three dimensional reciprocal space map.

        Parameters
        ----------
        dataset : ndarray
            The stacked detector images.
        new_shape : list
            The shape of the aimed reciprocal space map..
        rebinfactor : float, optional
            The rebin factor value of the reciprocal space. The default is 1.
        cval : float, optional
            The constant value for the interpolation if the correponding point is missing. The default is 0.
        prefilter : bool, optional
            Whether prefilter should be used before interpolation. The default is False.

        Returns
        -------
        intensityfinal : ndarray
            Final 3D diffraction intensity.

        """
        # Calculate the transformation matrix for the RMS
        self.peak_angle = self.scan_motor_ar[int(dataset.shape[0] / 2)]
        zd, yd, xd = dataset.shape
        nz, ny, nx = new_shape

        Coords_transform = self.cal_transformation_matrix(rebinfactor)
        inv_Coords_transform = np.linalg.inv(Coords_transform)
        intensityfinal = np.zeros((nz, ny, nx))

        offset = np.array([zd / 2.0, yd / 2.0]) - np.dot(inv_Coords_transform, np.array([nz / 2.0, nx / 2.0]))
        for X in np.arange(ny):
            intensity2d = dataset[:, :, int((ny - 1 - X) * rebinfactor + (xd / 2) % rebinfactor)]
            intensity2dinterpolation = affine_transform(intensity2d, inv_Coords_transform, offset=offset, output_shape=(nz, nx), order=3, mode='constant', cval=cval, output=float, prefilter=prefilter)
            intensityfinal[:, X, :] = intensity2dinterpolation
            sys.stdout.write('\rprogress:%d%%' % ((X + 1) * 100.0 / ny))
            sys.stdout.flush()

        print('')
        intensityfinal = np.clip(intensityfinal, 0, 1.0e7)
        return intensityfinal
