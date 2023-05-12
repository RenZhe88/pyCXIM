# -*- coding: utf-8 -*-
"""
Description
Created on Tue Apr 25 14:51:06 2023

@author: renzhe
"""
import numpy as np
from scipy.ndimage import affine_transform
import sys

class RC2RSM_6C():
    def __init__(self, scan_motor_ar, geometry='out_of_plane', omega=0, delta=0, chi=0, phi=0, gam=0, det_rot=0, energy=8000, distance=1830, pixelsize=0.075):
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
        if self.geometry == 'out_of_plane':
            self.omega = self.scan_motor_ar[int(pch[0])]
        elif self.geometry == 'in_plane':
            self.phi = self.scan_motor_ar[int(pch[0])]
        # print('om=%f, delta=%.2f, chi=%.2f, phi=%.2f, gamma=%.2f' % (np.rad2deg(self.omega), np.rad2deg(self.delta), np.rad2deg(self.chi), np.rad2deg(self.phi), np.rad2deg(self.gamma)))
        pixel_distance = np.linalg.norm([self.distance, (pch[1] - cch[0]) * self.pixelsize, (pch[2] - cch[1]) * self.pixelsize])
        q_vector = np.array([(cch[0] - pch[1]), (cch[1] - pch[2]), self.distance / self.pixelsize])
        det_rot_transform = np.array([[np.cos(self.det_rot), -np.sin(self.det_rot), 0], [np.sin(self.det_rot), np.cos(self.det_rot), 0], [0, 0, 1.0]])
        q_vector = np.dot(det_rot_transform, q_vector)
        delta_transform = np.array([[np.cos(self.delta), 0, np.sin(self.delta)], [0, 1, 0], [-np.sin(self.delta), 0, np.cos(self.delta)]])
        q_vector = np.dot(delta_transform, q_vector)
        gamma_transform = np.array([[1, 0, 0], [0, np.cos(self.gamma), np.sin(self.gamma)], [0, -np.sin(self.gamma), np.cos(self.gamma)]])
        q_vector = np.dot(gamma_transform, q_vector)
        q_vector = q_vector - np.array([0, 0, pixel_distance / self.pixelsize])
        omega_transform = np.array([[np.cos(self.omega), 0, -np.sin(self.omega)], [0, 1, 0], [np.sin(self.omega), 0, np.cos(self.omega)]])
        q_vector = np.dot(omega_transform, q_vector)
        chi_transform = np.array([[np.cos(self.chi), -np.sin(self.chi), 0], [np.sin(self.chi), np.cos(self.chi), 0], [0, 0, 1]])
        q_vector = np.dot(chi_transform, q_vector)
        phi_transform = np.array([[1, 0, 0], [0, np.cos(self.phi), np.sin(self.phi)], [0, -np.sin(self.phi), np.cos(self.phi)]])
        q_vector = np.dot(phi_transform, q_vector)
        hc = 1.23984 * 10000.0
        wavelength = hc / self.energy
        units = 2 * np.pi * self.pixelsize / wavelength / pixel_distance
        q_vector = q_vector * units
        # print(q_vector)
        return q_vector

    def cal_rel_q_pos(self, pch, cch):
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
        return (self.units * rebinfactor)

    def cal_transformation_matrix(self, rebinfactor=1):
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

    def cal_q_range(self, roi, cch, om_range=None, rebinfactor=1):
        corners_q = np.zeros((8, 3))
        if om_range is None:
            om_range = [0, -1]
        corner_num = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    corners_q[corner_num, :] = self.cal_rel_q_pos([om_range[i], roi[j], roi[k]], cch)
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

    def RSM_conversion(self, dataset, new_shape, rebinfactor=1, cval=0, prefilter=True):
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

class RC2RSM_2C():
    def __init__(self, scan_motor_ar, two_theta, energy=8000, distance=1830, pixelsize=0.075, geometry='out_of_plane'):

        self.scan_motor_ar = np.deg2rad(scan_motor_ar)
        self.scan_step = np.deg2rad((scan_motor_ar[-1] - scan_motor_ar[0]) / (len(scan_motor_ar) - 1))
        self.two_theta = np.deg2rad(two_theta)
        self.geometry = geometry
        self.distance = distance
        self.pixelsize = pixelsize
        self.energy = energy

        hc = 1.23984 * 10000.0
        wavelength = hc / self.energy
        self.units = 2 * np.pi * self.pixelsize / wavelength / self.distance
        return

    def get_RSM_unit(self, rebinfactor=1):
        return (self.units * rebinfactor)

    def cal_rel_q_pos(self, pch, cch):
        self.peak_angle = self.scan_motor_ar[int(pch[0])]
        if self.geometry == 'out_of_plane':
            q_vector = np.array([(cch[0] - pch[1]), (cch[1] - pch[2]), self.distance / self.pixelsize])
            delta_transform = np.array([[np.cos(self.two_theta), 0, np.sin(self.two_theta)], [0, 1, 0], [-np.sin(self.two_theta), 0, np.cos(self.two_theta)]])
            q_vector = np.dot(delta_transform, q_vector)
            q_vector = q_vector - np.array([0, 0, self.distance / self.pixelsize])
            omega_transform = np.array([[np.cos(self.peak_angle), 0, -np.sin(self.peak_angle)], [0, 1, 0], [np.sin(self.peak_angle), 0, np.cos(self.peak_angle)]])
            q_vector = np.dot(omega_transform, q_vector)
        elif self.geometry == 'in_plane':
            q_vector = np.array([(cch[0] - pch[1]), (cch[1] - pch[2]), self.distance / self.pixelsize])
            gamma_transform = np.array([[1, 0, 0], [0, np.cos(self.two_theta), np.sin(self.two_theta)], [0, -np.sin(self.two_theta), np.cos(self.two_theta)]])
            q_vector = np.dot(gamma_transform, q_vector)
            q_vector = q_vector - np.array([0, 0, self.distance / self.pixelsize])
            phi_transform = np.array([[1, 0, 0], [0, np.cos(self.peak_angle), np.sin(self.peak_angle)], [0, -np.sin(self.peak_angle), np.cos(self.peak_angle)]])
            q_vector = np.dot(phi_transform, q_vector)
        return q_vector

    def cal_rebinfactor(self):
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
        step_C = self.distance * self.scan_step / self.pixelsize
        if self.geometry == 'out_of_plane':
            transformation_matrix = np.array([[(np.cos(self.peak_angle) - np.cos(self.two_theta - self.peak_angle)) * step_C, -np.cos(self.two_theta - self.peak_angle)],
                                              [(np.sin(self.two_theta - self.peak_angle) + np.sin(self.peak_angle)) * step_C, np.sin(self.two_theta - self.peak_angle)]])
        elif self.geometry == 'in_plane':
            transformation_matrix = np.array([[(np.cos(self.peak_angle + self.two_theta) - np.cos(self.peak_angle)) * step_C, -np.cos(self.two_theta + self.peak_angle)],
                                              [(np.sin(self.peak_angle) - np.sin(self.peak_angle + self.two_theta)) * step_C, np.sin(self.peak_angle + self.two_theta)]])
        transformation_matrix = transformation_matrix / rebinfactor
        return transformation_matrix

    def cal_q_range(self, roi, cch, om_range=None, rebinfactor=1):
        corners_q = np.zeros((8, 3))
        if om_range is None:
            om_range = [0, -1]
        corner_num = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    corners_q[corner_num, :] = self.cal_rel_q_pos([om_range[i], roi[j], roi[k]], cch)
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

    def RSM_conversion(self, dataset, new_shape, rebinfactor=1, cval=0, prefilter=True):
        # Calculate the transformation matrix for the RMS
        self.peak_angle = self.scan_motor_ar[int(dataset.shape[0] / 2)]
        zd, yd, xd = dataset.shape
        nz, ny, nx = new_shape

        Coords_transform = self.cal_transformation_matrix(rebinfactor)
        inv_Coords_transform = np.linalg.inv(Coords_transform)
        intensityfinal = np.zeros((nz, ny, nx))
        if self.geometry == 'out_of_plane':
            offset = np.array([zd / 2.0, yd / 2.0]) - np.dot(inv_Coords_transform, np.array([nz / 2.0, nx / 2.0]))
            for X in np.arange(ny):
                intensity2d = dataset[:, :, int((ny / 2 - X) * rebinfactor + xd / 2 - 0.5)]
                intensity2dinterpolation = affine_transform(intensity2d, inv_Coords_transform, offset=offset, output_shape=(nz, nx), order=3, mode='constant', cval=cval, output=float, prefilter=prefilter)
                intensityfinal[:, X, :] = intensity2dinterpolation
                sys.stdout.write('\rprogress:%d%%' % ((X + 1) * 100.0 / ny))
                sys.stdout.flush()
        elif self.geometry == 'in_plane':
            offset = np.array([zd / 2.0, xd / 2.0]) - np.dot(inv_Coords_transform, np.array([nz / 2.0, nx / 2.0]))
            for Y in np.arange(ny):
                intensity2d = dataset[:, int((ny / 2 - Y) * rebinfactor + yd / 2 - 0.5), :]
                intensity2dinterpolation = affine_transform(intensity2d, inv_Coords_transform, offset=offset, output_shape=(nz, nx), order=3, mode='constant', cval=cval, output=float, prefilter=prefilter)
                intensityfinal[:, Y, :] = intensity2dinterpolation
                sys.stdout.write('\rprogress:%d%%' % ((Y + 1) * 100.0 / ny))
                sys.stdout.flush()
        print('')
        intensityfinal = np.clip(intensityfinal, 0, 1.0e7)
        return intensityfinal
