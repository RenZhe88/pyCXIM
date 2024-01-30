# -*- coding: utf-8 -*-
"""
Transform the detector images in a CDI scan to intensities in reciprocal space.

Created on Mon May 22 14:54:20 2023

@author: Ren Zhe
@email: renzhe@ihep.ac.cn, renzhetu001@gmail.com
"""
import os
import numpy as np
from scipy.ndimage import map_coordinates
import sys


class CDI2RSM():
    """
    Code to grid the CDI data into the orthogonal space.

    Parameters
    ----------
    phi_ar : ndarray
        The phi motor values. The typical value can be -5 to 185 with stepsize of 0.01 degree.
    half_roi_width_Y : int
        The half pixel width along the Y direction on the detector.
    half_roi_width_X : int
        The half pixel width along the X direction on the detector.
    energy : float,
        The energy of the X-ray beam in eV, e.g. 9000.
    distance : float,
        The sample detector distance in mm, e.g. 4950.
    pixelsize : float,
        The detector pixel size in mm, e.g. 0.075 for eiger detector.

    Returns
    -------
    None.

    """

    def __init__(self, phi_ar, half_roi_width_Y, half_roi_width_X, energy, distance, pixelsize):
        self.phi_ar = np.deg2rad(phi_ar)
        self.npoints = len(phi_ar)
        self.phistep = np.deg2rad(phi_ar[-1] - phi_ar[0]) / (self.npoints - 1)
        self.radius = half_roi_width_X
        self.half_height = half_roi_width_Y
        hc = 1.23984 * 10000.0
        wavelength = hc / energy
        self.unit = 2.0 * np.pi * pixelsize / wavelength / distance

    def cal_rebinfactor(self):
        """
        Suggest the possible rebin factors according to the step size of the scan.

        Returns
        -------
        float
            The suggested rebin factor.

        """
        rebinfactor = self.radius * self.phistep
        print("the pixel size of the outer circle / the pixel size on the detector: %f" % rebinfactor)
        return np.round(rebinfactor)

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
        return (self.unit * rebinfactor)

    def cartesian2polar(self, rebinfactor=1):
        """
        Calculate the 2D Qx Qy slice of reciprocal space coordinates to the corresponding detector space coordinates.

        Parameters
        ----------
        rebinfactor : int, optional
            The rebinfactor used. The default is 1.

        Returns
        -------
        ndarray
            The position of the corresponding detector pixels.

        """
        Y_ar, X_ar = np.mgrid[-(self.radius // rebinfactor):(self.radius // rebinfactor), -(self.radius // rebinfactor):(self.radius // rebinfactor)]
        Y_ar = np.ravel(Y_ar) * rebinfactor
        X_ar = np.ravel(X_ar) * rebinfactor
        Ny = np.zeros_like(Y_ar)
        Nx = np.zeros_like(X_ar)
        for i in range(len(Y_ar)):
            if Y_ar[i] >= 0:
                Ny[i] = (np.arctan2(Y_ar[i], X_ar[i]) - self.phi_ar[0]) / self.phistep
                Nx[i] = self.radius - np.sqrt(X_ar[i] ** 2 + Y_ar[i] ** 2)
            elif Y_ar[i] < 0:
                Ny[i] = (np.arctan2(Y_ar[i], X_ar[i]) + np.pi - self.phi_ar[0]) / self.phistep
                Nx[i] = self.radius + np.sqrt(X_ar[i] ** 2 + Y_ar[i] ** 2)
        return np.array([Ny, Nx])

    def grid_cdi(self, dataset, Npos, rebinfactor=1, cval=0, prefilter=False):
        """
        Transform the detector images into RSM.

        Parameters
        ----------
        dataset : ndarray
            The detector images.
        Npos : ndarray
            The ZX pixel position of the corrsponding Qx Qy slice.
        rebinfactor : int, optional
            The rebin factors. The default is 1.
        cval : float, optional
            The constant value filled for the missing boundaries. The default is 0.
        prefilter : bool, optional
            Whether prefilter is enabled during the interpolation. The default is False.

        Returns
        -------
        intensityfinal : ndarray
            The grided intensity in RSM.

        """
        height = 2 * (self.half_height // rebinfactor)
        width = 2 * (self.radius // rebinfactor)
        intensityfinal = np.zeros((height, width, width))
        for Y in np.arange(height):
            intensity2d = dataset[:, int(Y * rebinfactor) + self.half_height % rebinfactor, :]
            intensity2dinterpolation = map_coordinates(intensity2d, Npos, mode="constant", cval=cval, output=float, prefilter=prefilter).reshape((width, width))
            intensityfinal[Y, :, :] = np.clip(intensity2dinterpolation, 0, 1.0e8)
            sys.stdout.write('\rprogress:%d%%' % ((Y + 1) * 100.0 / height))
            sys.stdout.flush()
        print()
        return intensityfinal
