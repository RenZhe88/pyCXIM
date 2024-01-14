# -*- coding: utf-8 -*-
"""
Description
Created on Tue Apr 25 14:51:06 2023

@author: renzhe
"""
import numpy as np
from scipy.ndimage import affine_transform
import sys


class RC2RSM_6C(object):
    """
    Convert the detector images to the three dimensional reciprocal space map at the 6 circle diffractometer.

    The final coordinates would be in sample coordinates.
    z direction is vertically upwards.
    x direction is along the beam.
    y direction is perpendicular to the diffraction plane.

    Parameters
    ----------
    scan_motor_ar : list
        The motor values in the rocking curve, which should linearly change during the scan.
    geometry : str, optional
        The geometry can be either 'out_of_plane' or 'in_plane'.
        If geometry is 'out_of_plane', the scan motor should be om.
        If geometry is 'in_plane', the scan motor should be phi.
        The default is 'out_of_plane'.
    omega : float, optional
        The value of the omega motor in degree. The default is 0.
    delta : float, optional
        The value of the delta motor in degree. The default is 0.
    chi : float, optional
        The value of the chi motor in degree. The default is 0.
    phi : float, optional
        The value of the phi motor in degree. The default is 0.
    gam : float, optional
        The value of the gamma motor in degree. The default is 0.
    det_rot : float, optional
        The in-plane rotation of the detector obtained from the detector calibration scan. The default is 0.
    energy : float, optional
        The X-ray beam energy in eV. The default is 8000.
    distance : float, optional
        The sample detector distance in mm. The default is 1830.
    pixelsize : float, optional
        The detector pixelsize in mm. The default is 0.075.

    Returns
    -------
    None.

    """

    def __init__(self, scan_motor_ar, geometry='out_of_plane', omega=0, delta=0, chi=0, phi=0, gam=0, energy=8000, det_rot=0, distance=1830, pixelsize=0.075):

        self.scan_motor_ar = np.deg2rad(scan_motor_ar)
        self.scan_step = np.deg2rad((scan_motor_ar[-1] - scan_motor_ar[0]) / (len(scan_motor_ar) - 1.0))
        self.geometry = geometry
        self.omega = np.deg2rad(omega)
        self.delta = np.deg2rad(delta)
        self.chi = np.deg2rad(chi)
        self.phi = np.deg2rad(phi)
        self.gamma = np.deg2rad(gam)
        self.det_rot = np.deg2rad(det_rot)
        self.distance = distance
        self.pixelsize = pixelsize
        self.energy = energy

        hc = 1.23984 * 10000.0
        wavelength = hc / self.energy
        self.units = 2 * np.pi * self.pixelsize / wavelength / self.distance
        return

    def cal_abs_q_pos(self, pch, cch):
        """
        Calculate the absolute q values of a detector pixel in inverse angstrom.

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
            The calculated q_vector as [qz, qy, qx] in inverse angstrom.

        """
        if self.geometry == 'out_of_plane':
            self.omega = self.scan_motor_ar[int(pch[0])]
        elif self.geometry == 'in_plane':
            self.phi = self.scan_motor_ar[int(pch[0])]
        pixel_distance = np.linalg.norm([self.distance, (pch[1] - cch[0]) * self.pixelsize, (pch[2] - cch[1]) * self.pixelsize])
        q_vector = np.array([(cch[0] - pch[1]), (cch[1] - pch[2]), self.distance / self.pixelsize])
        det_rot_transform = np.array([[np.cos(self.det_rot), -np.sin(self.det_rot), 0],
                                      [np.sin(self.det_rot), np.cos(self.det_rot), 0],
                                      [0, 0, 1.0]])
        q_vector = np.dot(det_rot_transform, q_vector)
        delta_transform = np.array([[np.cos(self.delta), 0, np.sin(self.delta)],
                                    [0, 1, 0],
                                    [-np.sin(self.delta), 0, np.cos(self.delta)]])
        q_vector = np.dot(delta_transform, q_vector)
        gamma_transform = np.array([[1, 0, 0],
                                    [0, np.cos(self.gamma), np.sin(self.gamma)],
                                    [0, -np.sin(self.gamma), np.cos(self.gamma)]])
        q_vector = np.dot(gamma_transform, q_vector)
        q_vector = q_vector - np.array([0, 0, pixel_distance / self.pixelsize])
        omega_transform = np.array([[np.cos(self.omega), 0, -np.sin(self.omega)],
                                    [0, 1, 0],
                                    [np.sin(self.omega), 0, np.cos(self.omega)]])
        q_vector = np.dot(omega_transform, q_vector)
        chi_transform = np.array([[np.cos(self.chi), -np.sin(self.chi), 0],
                                  [np.sin(self.chi), np.cos(self.chi), 0],
                                  [0, 0, 1]])
        q_vector = np.dot(chi_transform, q_vector)
        phi_transform = np.array([[1, 0, 0],
                                  [0, np.cos(self.phi), np.sin(self.phi)],
                                  [0, -np.sin(self.phi), np.cos(self.phi)]])
        q_vector = np.dot(phi_transform, q_vector)
        hc = 1.23984 * 10000.0
        wavelength = hc / self.energy
        units = 2 * np.pi * self.pixelsize / wavelength / pixel_distance
        q_vector = q_vector * units
        return q_vector

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
        if self.geometry == 'out_of_plane':
            self.omega = self.scan_motor_ar[int(pch[0])]
        elif self.geometry == 'in_plane':
            self.phi = self.scan_motor_ar[int(pch[0])]
        q_vector = np.array([(cch[0] - pch[1]), (cch[1] - pch[2]), self.distance / self.pixelsize])
        det_rot_transform = np.array([[np.cos(self.det_rot), -np.sin(self.det_rot), 0], [np.sin(self.det_rot), np.cos(self.det_rot), 0], [0, 0, 1.0]])
        q_vector = np.dot(det_rot_transform, q_vector)
        delta_transform = np.array([[np.cos(self.delta), 0, np.sin(self.delta)], [0, 1, 0], [-np.sin(self.delta), 0, np.cos(self.delta)]])
        q_vector = np.dot(delta_transform, q_vector)
        gamma_transform = np.array([[1, 0, 0], [0, np.cos(self.gamma), np.sin(self.gamma)], [0, -np.sin(self.gamma), np.cos(self.gamma)]])
        q_vector = np.dot(gamma_transform, q_vector)
        q_vector = q_vector - np.array([0, 0, self.distance / self.pixelsize])
        omega_transform = np.array([[np.cos(self.omega), 0, -np.sin(self.omega)], [0, 1, 0], [np.sin(self.omega), 0, np.cos(self.omega)]])
        q_vector = np.dot(omega_transform, q_vector)
        chi_transform = np.array([[np.cos(self.chi), -np.sin(self.chi), 0], [np.sin(self.chi), np.cos(self.chi), 0], [0, 0, 1]])
        q_vector = np.dot(chi_transform, q_vector)
        phi_transform = np.array([[1, 0, 0], [0, np.cos(self.phi), np.sin(self.phi)], [0, -np.sin(self.phi), np.cos(self.phi)]])
        q_vector = np.dot(phi_transform, q_vector)
        return q_vector

    def cal_rebinfactor(self):
        """
        Calculate the recommended rebin factors for the RSM.

        Returns
        -------
        rebinfactor : int
            The rebin factor recommended for the reciprocal space map.

        """
        if self.geometry == 'out_of_plane':
            rebinfactor = np.abs(2.0 * np.sin(self.delta / 2.0) * self.distance / self.pixelsize * self.scan_step)
        elif self.geometry == 'in_plane':
            rebinfactor = np.abs(2.0 * np.sin(self.gamma / 2.0) * self.distance / self.pixelsize * self.scan_step)

        print("rebinfactor calculated:%f" % rebinfactor)
        if rebinfactor > 1.5:
            rebinfactor = int(round(rebinfactor))
            print('Maybe consider increasing the step numbers in the scan.')
        else:
            rebinfactor = 1
            print('The number of steps for the scan shoulde be fine.')
        return rebinfactor

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
            The units of the reciprocal space map.

        """
        return (self.units * rebinfactor)

    def cal_transformation_matrix(self, rebinfactor=1):
        """
        Calculte the corresponding transformation matrix between the detector coordinates and the sample coordinates.

        If the geometry is out_of_plane, the rotation of gamma motor is ignored.
        If the geometry is in_plane, the rotation of the chi and omega is ignored.
        Please try to avoid these ignored motors in the scan.
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

        if self.geometry == 'out_of_plane':
            flip_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            det_rot_matrix = np.array([[1, 0, 0], [0, np.cos(self.det_rot), -np.sin(self.det_rot)], [0, np.sin(self.det_rot), np.cos(self.det_rot)]])
            rocking_matrix = np.array([[(np.cos(self.omega) - np.cos(self.delta - self.omega)) * step_C, np.cos(self.delta - self.omega), 0],
                                       [0, 0, 1],
                                       [(np.sin(self.delta - self.omega) + np.sin(self.omega)) * step_C, -np.sin(self.delta - self.omega), 0]])
            chi_transform = np.array([[np.cos(self.chi), -np.sin(self.chi), 0], [np.sin(self.chi), np.cos(self.chi), 0], [0, 0, 1]])
            phi_transform = np.array([[1, 0, 0], [0, np.cos(self.phi), np.sin(self.phi)], [0, -np.sin(self.phi), np.cos(self.phi)]])

            transformation_matrix = np.dot(phi_transform, np.dot(chi_transform, np.dot(rocking_matrix, np.dot(det_rot_matrix, flip_matrix))))
        elif self.geometry == 'in_plane':
            flip_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            det_rot_matrix = np.array([[1, 0, 0], [0, np.cos(self.det_rot), -np.sin(self.det_rot)], [0, np.sin(self.det_rot), np.cos(self.det_rot)]])
            delta_transform = np.array([[np.cos(self.delta), 0, np.sin(self.delta)], [0, 1, 0], [-np.sin(self.delta), 0, np.cos(self.delta)]])
            rocking_matrix = np.array([[0, 1, 0],
                                       [(np.cos(self.phi + self.gamma) - np.cos(self.phi)) * step_C, 0, np.cos(self.gamma + self.phi)],
                                       [(np.sin(self.phi) - np.sin(self.phi + self.gamma)) * step_C, 0, -np.sin(self.phi + self.gamma)]])
            transformation_matrix = np.dot(rocking_matrix, np.dot(delta_transform, np.dot(det_rot_matrix, flip_matrix)))
        transformation_matrix = transformation_matrix / rebinfactor
        return transformation_matrix

    def cal_q_range(self, roi, cch, scan_range=None, rebinfactor=1):
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
        if scan_range is None:
            scan_range = [0, -1]
        corner_num = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    corners_q[corner_num, :] = self.cal_rel_q_pos([scan_range[i], roi[j], roi[k + 2]], cch)
                    corner_num += 1
        q_origin = np.amin(corners_q, axis=0) * self.units
        new_shape = (np.ptp(corners_q, axis=0) / rebinfactor).astype(int)
        RSM_unit = self.units * rebinfactor

        if self.geometry == 'in_plane':
            q_origin[[0, 1]] = q_origin[[1, 0]]
            new_shape[[0, 1]] = new_shape[[1, 0]]

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
        if self.geometry == 'out_of_plane':
            self.omega = self.scan_motor_ar[int(dataset.shape[0] / 2)]
        elif self.geometry == 'in_plane':
            self.phi = self.scan_motor_ar[int(dataset.shape[0] / 2)]
        Coords_transform = self.cal_transformation_matrix(rebinfactor)
        inv_Coords_transform = np.linalg.inv(Coords_transform)
        offset = np.array(dataset.shape) / 2.0 - np.dot(inv_Coords_transform, new_shape.astype(float) / 2.0)

        # Interpolation
        intensityfinal = affine_transform(dataset, inv_Coords_transform, offset=offset, output_shape=tuple(new_shape.astype(int)), order=3, mode='constant', cval=cval, output=float, prefilter=prefilter)
        intensityfinal = np.clip(intensityfinal, 0, 1.0e7)
        return intensityfinal

def cal_abs_q_pos(pixel_position, motor_position, detector_para):
    """
    Calculate the absolute q position.

    Parameters
    ----------
    pixel_position : list
        The peak position in [Y, X] order.
    motor_position : list
        list of scan parameters in [omega, delta, chi, phi, nu, energy] order. Angle in degree, and energy in eV.
    detector_para : list
        The detector parameters in [distance, pixelsize, det_rot, cch] order.

    Returns
    -------
    q_vector : list
        q_vector = [qz, qy, qx] in inverse angstrom.

    """
    omega, delta, chi, phi, nu = np.deg2rad(motor_position[:5])
    energy = motor_position[5]
    distance, pixelsize, det_rot, cch = detector_para
    det_rot = np.deg2rad(det_rot)

    pixel_distance = np.linalg.norm([distance, (cch[0] - pixel_position[0]) * pixelsize, (cch[1] - pixel_position[1]) * pixelsize])
    q_vector = np.array([cch[0] - pixel_position[0], cch[1] - pixel_position[1], distance / pixelsize])
    det_rot_transform = np.array([[np.cos(det_rot), -np.sin(det_rot), 0],
                                  [np.sin(det_rot), np.cos(det_rot), 0],
                                  [0, 0, 1.0]])
    q_vector = np.dot(det_rot_transform, q_vector)
    delta_transform = np.array([[np.cos(delta), 0, np.sin(delta)],
                                [0, 1, 0],
                                [-np.sin(delta), 0, np.cos(delta)]])
    q_vector = np.dot(delta_transform, q_vector)
    nu_transform = np.array([[1, 0, 0],
                             [0, np.cos(nu), np.sin(nu)],
                             [0, -np.sin(nu), np.cos(nu)]])
    q_vector = np.dot(nu_transform, q_vector)
    q_vector = q_vector - np.array([0, 0, pixel_distance / pixelsize])
    omega_transform = np.array([[np.cos(omega), 0, -np.sin(omega)],
                                [0, 1, 0],
                                [np.sin(omega), 0, np.cos(omega)]])
    q_vector = np.dot(omega_transform, q_vector)
    chi_transform = np.array([[np.cos(chi), -np.sin(chi), 0],
                              [np.sin(chi), np.cos(chi), 0],
                              [0, 0, 1]])
    q_vector = np.dot(chi_transform, q_vector)
    phi_transform = np.array([[1, 0, 0],
                              [0, np.cos(phi), np.sin(phi)],
                              [0, -np.sin(phi), np.cos(phi)]])
    q_vector = np.dot(phi_transform, q_vector)
    hc = 1.23984 * 10000.0
    wavelength = hc / energy
    units = 2 * np.pi * pixelsize / wavelength / pixel_distance
    q_vector = q_vector * units
    return q_vector
