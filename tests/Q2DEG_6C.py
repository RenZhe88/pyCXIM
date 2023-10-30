# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 15:41:26 2023

@author: renzh
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import sys


def cal_abs_q_pos(omega, delta, chi, phi, gamma, energy, distance, pixelsize):

    omega = np.deg2rad(omega)
    delta = np.deg2rad(delta)
    chi = np.deg2rad(chi)
    phi = np.deg2rad(phi)
    gamma = np.deg2rad(gamma)

    # print('om=%f, delta=%.2f, chi=%.2f, phi=%.2f, gamma=%.2f' % (np.rad2deg(self.omega), np.rad2deg(self.delta), np.rad2deg(self.chi), np.rad2deg(self.phi), np.rad2deg(self.gamma)))
    pixel_distance = np.linalg.norm([distance, 0, 0])
    q_vector = np.array([0, 0, distance / pixelsize])
    delta_transform = np.array([[np.cos(delta), 0, np.sin(delta)], [0, 1, 0], [-np.sin(delta), 0, np.cos(delta)]])
    q_vector = np.dot(delta_transform, q_vector)
    gamma_transform = np.array([[1, 0, 0], [0, np.cos(gamma), np.sin(gamma)], [0, -np.sin(gamma), np.cos(gamma)]])
    q_vector = np.dot(gamma_transform, q_vector)
    q_vector = q_vector - np.array([0, 0, pixel_distance / pixelsize])
    omega_transform = np.array([[np.cos(omega), 0, -np.sin(omega)], [0, 1, 0], [np.sin(omega), 0, np.cos(omega)]])
    q_vector = np.dot(omega_transform, q_vector)
    chi_transform = np.array([[np.cos(chi), -np.sin(chi), 0], [np.sin(chi), np.cos(chi), 0], [0, 0, 1]])
    q_vector = np.dot(chi_transform, q_vector)
    phi_transform = np.array([[1, 0, 0], [0, np.cos(phi), np.sin(phi)], [0, -np.sin(phi), np.cos(phi)]])
    q_vector = np.dot(phi_transform, q_vector)
    hc = 1.23984 * 10000.0
    wavelength = hc / energy
    units = 2 * np.pi * pixelsize / wavelength / pixel_distance
    q_vector = q_vector * units
    return q_vector


def cal_q_single_peak(angles_expected, parameters, para_selected, expected_q):

    parameters = np.array(parameters)
    parameters[para_selected] = np.array(angles_expected)
    parameters = list(parameters)
    q_vector = cal_abs_q_pos(*parameters)
    error_ar = q_vector - expected_q
    return error_ar


def cal():
    expected_q = [2 * np.pi * 4 / 5.1860, 0, 0]
    energy = 12000
    pixelsize = 75e-3
    distance = 1828.843760
    rotation_needed = ['omega', 'delta', 'phi']
    parameters = [0, 0, 0, 0, 0, energy, distance, pixelsize]
    paranames = ['omega', 'delta', 'chi', 'phi', 'gamma', 'energy', 'distance', 'pixelsize']

    para_selected = []
    for element in paranames:
        if element in rotation_needed:
            para_selected.append(1)
        else:
            para_selected.append(0)

    print(para_selected)
    para_selected = np.array(para_selected, dtype='bool')
    print(para_selected)

    angles_expected = fsolve(cal_q_single_peak, [5, 5, 5], args=(parameters, para_selected, expected_q))
    print(angles_expected)

if __name__ == '__main__':
    cal()