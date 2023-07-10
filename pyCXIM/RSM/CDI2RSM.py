# -*- coding: utf-8 -*-
"""
Created on Mon May 22 14:54:20 2023

@author: renzh
"""
import os
import numpy as np
from scipy.ndimage import map_coordinates
import sys


class CDI2RSM():
    def __init__(self, phi_ar, half_roi_width_Y, half_roi_width_X, energy=8000, distance=4950, pixelsize=0.075):
        self.phi_ar = np.deg2rad(phi_ar)
        self.npoints = len(phi_ar)
        self.phistep = np.deg2rad(phi_ar[-1] - phi_ar[0]) / (self.npoints - 1)
        self.radius = half_roi_width_X
        self.half_height = half_roi_width_Y
        hc = 1.23984 * 10000.0
        wavelength = hc / energy
        self.unit = 2.0 * np.pi * pixelsize / wavelength / distance

    def cal_rebinfactor(self):
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
