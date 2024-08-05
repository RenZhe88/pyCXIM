# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 15:41:03 2024

@author: renzh
"""
import numpy as np


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
