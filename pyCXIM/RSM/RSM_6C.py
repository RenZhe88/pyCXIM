# -*- coding: utf-8 -*-
"""
Functions for converting 2D detector images recorded in the Rocking curve to the reciprocal space maps.

Created on Tue Apr 25 14:51:06 2023

@author: Ren Zhe
@email: renzhe@ihep.ac.cn
"""
import numpy as np
from scipy.ndimage import affine_transform


class RSM_6C(object):
    """
    Convert the detector images to the three dimensional reciprocal space map at the 6 circle diffractometer.
    
    Basic assumption:
        1. scan angle is small,e.g. 1-2 degrees
        2. sample detector distance is much large than the detecor size.

    The final coordinates would be in sample coordinates.
    z direction is vertically upwards.
    x direction is along the beam.
    y direction is perpendicular to the diffraction plane.

    Parameters
    ----------
    scan_type : str
        The geometry can be either 'RC' (Rocking curve) or 'TT' (theta-2theta scan).
    omega : float or ndarray, optional
        The value of the omega motor in degree. The default is 0.
    delta : float or ndarray, optional
        The value of the delta motor in degree. The default is 0.
    chi : float or ndarray, optional
        The value of the chi motor in degree. The default is 0.
    phi : float or ndarray, optional
        The value of the phi motor in degree. The default is 0.
    gamma : float or ndarray, optional
        The value of the gamma motor in degree. The default is 0.
    mu : float, optional
        The value of the mu motor in degree. The default is 0.
    energy : float, optional
        The X-ray beam energy in eV. The default is 8000.
    distance : float, optional
        The sample detector distance in mm. The default is 1830.
    pixelsize : float, optional
        The detector pixelsize in mm. The default is 0.075.
    det_rot : float, optional
        The in-plane rotation of the detector obtained from the detector calibration scan. The default is 0.
    cch : list, optional
        The central channel of the detector, which is the direct beam position on the detector in [Y, X] order.
    additional_rotation_matrix: list optional
        The rotation matrix, which describes the difference from the measured U matrix and the expected U matrix.

    Returns
    -------
    None.

    """

    def __init__(self, scan_type,
                 omega=0, delta=0, chi=0, phi=0, gamma=0, mu=0, energy=8000,
                 distance=1830, pixelsize=0.075, det_rot=0, cch=[0, 0],
                 additional_rotation_matrix=None):

        self.scan_type = scan_type
        self.omega = np.deg2rad(omega)
        self.delta = np.deg2rad(delta)
        self.chi = np.deg2rad(chi)
        self.phi = np.deg2rad(phi)
        self.gamma = np.deg2rad(gamma)
        self.mu = np.deg2rad(mu)
        self.det_rot = np.deg2rad(det_rot)
        self.distance = distance
        self.pixelsize = pixelsize
        self.energy = energy
        self.cch = cch

        # cheche the scan type and geometry
        motor_check = []
        for motor in [self.omega, self.delta, self.chi, self.phi, self.gamma, self.mu, self.energy]:
            if isinstance(motor, np.ndarray):
                motor_check.append(True)
            else:
                motor_check.append(False)
        if self.scan_type == 'RC':
            if motor_check == [True, False, False, False, False, False, False]:
                self.geometry = 'out_of_plane'
                self.scan_step = (self.omega[-1] - self.omega[0]) / (len(self.omega) - 1.0)
                self.npoint = len(self.omega)
            elif motor_check == [False, False, False, True, False, False, False]:
                self.geometry = 'in_plane'
                self.scan_step = (self.phi[-1] - self.phi[0]) / (len(self.phi) - 1.0)
                self.npoint = len(self.phi)
            else:
                raise TypeError("Scan type and data does not match!")
        elif self.scan_type == 'TT':
            if motor_check == [True, True, False, False, False, False, False]:
                self.geometry = 'out_of_plane'
                self.scan_step = (self.omega[-1] - self.omega[0]) / (len(self.omega) - 1.0)
                self.npoint = len(self.omega)
                if (self.mu != 0) or (self.gamma !=0):
                    print('Warning: For the theta 2theta scan in this geometry, mu and gamma should be around zeros!')
            elif motor_check == [False, False, False, True, True, False, False]:
                self.geometry = 'in_plane'
                self.scan_step = (self.phi[-1] - self.phi[0]) / (len(self.phi) - 1.0)
                self.npoint = len(self.phi)
                if (self.chi != 0) or (self.omega !=0):
                    print('Warning: For the theta 2theta scan in this geometry, mu and gamma should be around zeros!')
            else:
                raise TypeError("Scan type and data does not match!")
        else:
            raise TypeError("Scan type could only be RC (rocking curve) or TT (theta-2theta scan)!")

        if additional_rotation_matrix is None:
            self.additional_rotation_matrix = np.eye(3)
            print('calibration rotation matrix does not exist?')
        else:
            self.additional_rotation_matrix = np.array(additional_rotation_matrix, dtype=float)

        assert self.additional_rotation_matrix.shape == (3, 3), 'The shape of additional rotation matrix must be (3, 3)'

        hc = 1.23984 * 10000.0
        wavelength = hc / self.energy
        self.units = 2 * np.pi * self.pixelsize / wavelength / self.distance
        return

    def get_motor_pos(self, img_index):
        """
        Get the motor positions for a certain point in the scan.

        Parameters
        ----------
        img_index : int
            The index of the scan point.

        Returns
        -------
        motor_position : list
            List of motor position values in omega, delta, chi, phi, gamma, mu, energy order.

        """
        motor_position = []
        for motor in [self.omega, self.delta, self.chi, self.phi, self.gamma, self.mu, self.energy]:
            if isinstance(motor, np.ndarray):
                motor_position.append(motor[img_index])
            else:
                motor_position.append(motor)
        return np.array(motor_position)

    def cal_abs_q_pos(self, pch):
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
        pixel_position = pch[1:]
        motor_position = self.get_motor_pos(int(pch[0]))
        detector_para = [self.distance, self.pixelsize, self.det_rot, self.cch]
        q_vector = cal_q_pos(pixel_position, motor_position, detector_para, DEG=False)
        q_vector = np.dot(self.additional_rotation_matrix, q_vector)
        return q_vector

    def cal_rebinfactor(self):
        """
        Calculate the recommended rebin factors for the RSM.

        Returns
        -------
        rebinfactor : int
            The rebin factor recommended for the reciprocal space map.

        """
        step_C = self.distance * self.scan_step / self.pixelsize
        
        if (self.scan_type == 'RC') and (self.geometry == 'out_of_plane'):
            rebinfactor = np.abs(2.0 * np.sin(self.delta / 2.0) * step_C)
        elif (self.scan_type == 'RC') and (self.geometry == 'in_plane'):
            rebinfactor = np.abs(2.0 * np.sin(self.gamma / 2.0) * step_C)
        elif (self.scan_type == 'TT') and (self.geometry == 'out_of_plane'):
            rebinfactor = np.abs(2.0 * (1 + np.cos((self.detla[0] + self.detla[-1]) / 2.0)) * step_C)
        elif (self.scan_type == 'TT') and (self.geometry == 'in_plane'):
            rebinfactor = np.abs(2.0 * (1 + np.cos((self.gamma[0] + self.gamma[-1]) / 2.0)) * step_C)

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
        omega, delta, chi, phi, gamma, mu, energy = self.get_motor_pos(int(self.npoint / 2))
        
        step_C = self.distance * self.scan_step / self.pixelsize

        flip_matrix = np.array([[1, 0, 0],
                                [0, -1, 0],
                                [0, 0, -1]])
        det_rot_transform = np.array([[np.cos(self.det_rot), -np.sin(self.det_rot), 0],
                                      [np.sin(self.det_rot), np.cos(self.det_rot), 0],
                                      [0, 0, 1.0]])
        delta_transform = np.array([[np.cos(delta), 0, np.sin(delta)],
                                    [0, 1, 0],
                                    [-np.sin(delta), 0, np.cos(delta)]])
        gamma_transform = np.array([[1, 0, 0],
                                    [0, np.cos(gamma), np.sin(gamma)],
                                    [0, -np.sin(gamma), np.cos(gamma)]])
        mu_transform = np.array([[1.0, 0, 0],
                                 [0, np.cos(mu), -np.sin(mu)],
                                 [0, np.sin(mu), np.cos(mu)]])
        omega_transform = np.array([[np.cos(omega), 0, -np.sin(omega)],
                                    [0, 1, 0],
                                    [np.sin(omega), 0, np.cos(omega)]])
        chi_transform = np.array([[np.cos(chi - np.pi / 2.0), -np.sin(chi - np.pi / 2.0), 0],
                                  [np.sin(chi - np.pi / 2.0), np.cos(chi - np.pi / 2.0), 0],
                                  [0, 0, 1]])
        phi_transform = np.array([[1, 0, 0],
                                  [0, np.cos(phi), np.sin(phi)],
                                  [0, -np.sin(phi), np.cos(phi)]])

        if (self.scan_type == 'RC') and (self.geometry == 'out_of_plane'):
            matrix1 = np.dot(mu_transform,
                             np.dot(gamma_transform,
                                    np.dot(delta_transform,
                                           det_rot_transform)))
            matrix2 = np.dot(phi_transform, np.dot(chi_transform, omega_transform))
            matrix3 = np.array([[step_C * (np.cos(self.mu) - matrix1[2, 2]), matrix1[0, 0], matrix1[0, 1]],
                                [0, matrix1[1, 0], matrix1[1, 1]],
                                [step_C * matrix1[0, 2], matrix1[2, 0], matrix1[2, 1]]])
            transformation_matrix = np.dot(self.additional_rotation_matrix,
                                           np.dot(matrix2,
                                                  np.dot(matrix3, flip_matrix)))
        elif (self.scan_type == 'RC') and (self.geometry == 'in_plane'):
            matrix1 = np.dot(phi_transform,
                             np.dot(chi_transform,
                                    np.dot(omega_transform,
                                           np.dot(mu_transform,
                                                  np.dot(gamma_transform,
                                                         np.dot(delta_transform,
                                                                det_rot_transform))))))
            matrix2 = np.dot(phi_transform,
                             np.dot(chi_transform,
                                    np.dot(omega_transform,
                                           mu_transform)))
            matrix3 = np.array([[0, matrix1[0, 0], matrix1[0, 1]],
                                [step_C * (matrix1[2, 2] - matrix2[2, 2]), matrix1[1, 0], matrix1[1, 1]],
                                [step_C * (matrix2[1, 2] - matrix1[1, 2]), matrix1[2, 0], matrix1[2, 1]]])
            transformation_matrix = np.dot(self.additional_rotation_matrix,
                                           np.dot(matrix3, flip_matrix))
        elif (self.scan_type == 'TT') and (self.geometry == 'out_of_plane'):
            matrix1 = np.dot(delta_transform, det_rot_transform)
            matrix2 = np.dot(phi_transform, 
                             np.dot(chi_transform, 
                                    omega_transform))
            matrix3 = np.array([[step_C * (1 + matrix1[2, 2]), matrix1[0, 0], matrix1[0, 1]],
                                [0, matrix1[1, 0], matrix1[1, 1]],
                                [-step_C * matrix1[0, 2], matrix1[2, 0], matrix1[2, 1]]])
            transformation_matrix = np.dot(self.additional_rotation_matrix,
                                           np.dot(matrix2,
                                                  np.dot(matrix3, flip_matrix)))
        elif (self.scan_type == 'TT') and (self.geometry == 'in_plane'):
            matrix1 = np.dot(gamma_transform,
                             np.dot(delta_transform,
                                    det_rot_transform))
            matrix2 = np.dot(phi_transform, mu_transform)
            matrix3 = np.array([[0, matrix1[0, 0], matrix1[0, 1]],
                                [-step_C * (1 + matrix1[2, 2]), matrix1[1, 0], matrix1[1, 1]],
                                [step_C * matrix1[1, 2], matrix1[2, 0], matrix1[2, 1]]])
            transformation_matrix = np.dot(self.additional_rotation_matrix,
                                           np.dot(matrix2,
                                                  np.dot(matrix3, flip_matrix)))
        transformation_matrix = transformation_matrix / rebinfactor
        return transformation_matrix

    def cal_q_range(self, roi, rebinfactor=1):
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

        scan_range = [0, -1]
        corner_num = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    corners_q[corner_num, :] = self.cal_abs_q_pos([scan_range[i], roi[j], roi[k + 2]])
                    corner_num += 1

        new_shape = (np.ptp(corners_q, axis=0) / self.units / rebinfactor).astype(int)
        RSM_unit = self.units * rebinfactor

        pch = [int(self.npoint / 2), (roi[0] + roi[1]) / 2.0, (roi[2] + roi[3]) / 2.0]
        q_center = self.cal_abs_q_pos(pch)
        q_origin = q_center - np.ptp(corners_q, axis=0) / 2.0

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
        Coords_transform = self.cal_transformation_matrix(rebinfactor)
        inv_Coords_transform = np.linalg.inv(Coords_transform)
        offset = np.array(dataset.shape) / 2.0 - np.dot(inv_Coords_transform, new_shape.astype(float) / 2.0)

        # Interpolation
        intensityfinal = affine_transform(dataset, inv_Coords_transform, offset=offset, output_shape=tuple(new_shape.astype(int)), order=3, mode='constant', cval=cval, output=float, prefilter=prefilter)
        intensityfinal = np.clip(intensityfinal, 0, 1.0e7)
        return intensityfinal


def cal_q_pos(pixel_position, motor_position, detector_para, DEG=True):
    """
    Calculate the absolute q position.

    Parameters
    ----------
    pixel_position : list
        The peak position in [Y, X] order.
    motor_position : list
        list of scan parameters in [omega, delta, chi, phi, gamma, energy] order. Energy in eV.
    detector_para : list
        The detector parameters in [distance, pixelsize, det_rot, cch] order.
        distance is the detector-sample distance in mm.
        pixelsize is the pixelsize of the detector in mm.
        det_rot is the inplane rotation of the detector.
        cch is the direct beam position in [Y, X] order.
    DEG : bool
        If the DEG is true, the angles should be given in degree, else the angles should be given in radians.

    Returns
    -------
    q_vector : list
        q_vector = [qz, qy, qx] in inverse angstrom.

    """
    omega, delta, chi, phi, gamma, mu, energy = motor_position
    distance, pixelsize, det_rot, cch = detector_para

    if DEG:
        omega, delta, chi, phi, gamma, mu, det_rot = np.deg2rad([omega, delta, chi, phi, gamma, mu, det_rot])

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
    gamma_transform = np.array([[1, 0, 0],
                                [0, np.cos(gamma), np.sin(gamma)],
                                [0, -np.sin(gamma), np.cos(gamma)]])
    q_vector = np.dot(gamma_transform, q_vector)
    q_vector = q_vector - np.array([0, 0, pixel_distance / pixelsize])
    mu_transform = np.array([[1.0, 0, 0],
                             [0, np.cos(mu), -np.sin(mu)],
                             [0, np.sin(mu), np.cos(mu)]])
    q_vector = np.dot(mu_transform, q_vector)
    omega_transform = np.array([[np.cos(omega), 0, -np.sin(omega)],
                                [0, 1, 0],
                                [np.sin(omega), 0, np.cos(omega)]])
    q_vector = np.dot(omega_transform, q_vector)
    chi_transform = np.array([[np.cos(chi - np.pi / 2.0), -np.sin(chi - np.pi / 2.0), 0],
                              [np.sin(chi - np.pi / 2.0), np.cos(chi - np.pi / 2.0), 0],
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

def det2q_2D(motor_position, detector_para, det_size):
    """
    Calculate the corresponding q_values of each pixel of a 2D detector mounted on the six circle diffractometer.

    Used to analysis the theta2theta scans performed with the 2D area detector and the 6 circle diffractometer.

    Parameters
    ----------
    motor_position : list
        list of scan parameters in [omega, delta, chi, phi, gamma, energy] order. Angles in degree, Energy in eV.
    detector_para : list
        The detector parameters in [distance, pixelsize, det_rot, cch] order.
        distance is the detector-sample distance in mm.
        pixelsize is the pixelsize of the detector in mm.
        det_rot is the inplane rotation of the detector in degree.
        cch is the direct beam position in [Y, X] order.
    det_size : tuple
        The size of the 2D detector in [Y, X] order.

    Returns
    -------
    q_2D : ndarray
        The corresponding q_values of each pixels on the detector.

    """
    delta, gamma, energy = motor_position
    distance, pixelsize, det_rot, cch = detector_para

    detY, detX = np.mgrid[0:det_size[0], 0:det_size[1]]
    pixel_distance = np.linalg.norm([np.zeros(det_size) + distance, (cch[0] - detY) * pixelsize, (cch[1] - detX) * pixelsize], axis=0)
    q_vector = np.stack(((cch[0] - detY) * pixelsize / pixel_distance, (cch[1] - detX) * pixelsize / pixel_distance, distance / pixel_distance))

    delta, gamma, det_rot = np.deg2rad([delta, gamma, det_rot])
    det_rot_transform = np.array([[np.cos(det_rot), -np.sin(det_rot), 0],
                                  [np.sin(det_rot), np.cos(det_rot), 0],
                                  [0, 0, 1.0]])
    delta_transform = np.array([[np.cos(delta), 0, np.sin(delta)],
                                [0, 1, 0],
                                [-np.sin(delta), 0, np.cos(delta)]])
    gamma_transform = np.array([[1, 0, 0],
                                [0, np.cos(gamma), np.sin(gamma)],
                                [0, -np.sin(gamma), np.cos(gamma)]])
    transform_2theta_matrix = np.dot(gamma_transform, np.dot(delta_transform, det_rot_transform))

    q_vector = np.tensordot(transform_2theta_matrix, q_vector, (1, 0))
    q_vector = q_vector - np.array([0, 0, 1])[:, np.newaxis, np.newaxis]
    q_2D = np.linalg.norm(q_vector, axis=0)

    hc = 1.23984 * 10000.0
    wavelength = hc / energy
    units = 2 * np.pi / wavelength
    q_2D = q_2D * units
    return q_2D