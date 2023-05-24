# -*- coding: utf-8 -*-
"""
The common phase retrieval functions for CPU based phase retrieval process.

Due to the speed limitations this package should be only used for 2D phase retieval process.

In case of questions, please contact us.

Author: Zhe Ren, Han Xu
Date: %(date)s
Email: zhe.ren@desy.de, han.xu@desy.de or renzhetu001@gmail.com
"""

import numpy as np
from scipy.ndimage import measurements
from scipy.ndimage import affine_transform


def cal_PRTF(intensity, Img_sum, MaskFFT=None):
    """
    Calculate the phase retrieval transfer function from the retrieved results.

    Parameters
    ----------
    intensity : ndarray
        Measured diffraction intensity.
    Img_sum : ndarray
        Summed results of the phase retrieval.
    MaskFFT : ndarray, optional
        The mask defining the bad pixels on the detector. The default is None.

    Returns
    -------
    PRTF_1D : ndarray
        1D phase retrieval transfer function.

    """
    if MaskFFT is None:
        MaskFFT = np.zeros_like(intensity)
    PRTF = np.ma.masked_array((np.abs(np.fft.fftshift(np.fft.fftn(Img_sum))) + 0.1) / (np.sqrt(intensity) + 0.1), mask=MaskFFT)
    if intensity.ndim == 2:
        r_2D = np.linalg.norm(np.indices(intensity.shape) - np.array(intensity.shape)[:, np.newaxis, np.newaxis] / 2 + 0.5, axis=0)
        PRTF_1D = np.zeros(int(np.amax(r_2D)))
        for i in range(int(np.amax(r_2D))):
            con = np.logical_and(r_2D >= i, r_2D < i + 1)
            PRTF_1D[i] = np.mean(PRTF[con])
    elif intensity.ndim == 3:
        r_3D = np.linalg.norm(np.indices(intensity.shape) - np.array(intensity.shape)[:, np.newaxis, np.newaxis, np.newaxis] / 2 + 0.5, axis=0)
        PRTF_1D = np.zeros(int(np.amax(r_3D)))
        for i in range(int(np.amax(r_3D))):
            con = np.logical_and(r_3D >= i, r_3D < i + 1)
            PRTF_1D[i] = np.mean(PRTF[con])
    return PRTF_1D


def get_Fourier_space_error(img, support, intensity, MaskFFT):
    """
    Calculate the fourier space error of the result reconstruction.

    Parameters
    ----------
    img : ndarray
        the result for the phase retrieval.
    support : ndarray
        the support.
    intensity : ndarray
        the orignal diffraction pattern.
    MaskFFT : ndarray
        the mask for the bad pixels.

    Returns
    -------
    error : float
        Fourier space error of the result reconstruction.


    """
    ModulusFFT = np.fft.fftshift(np.sqrt(intensity))
    error = np.sum(np.square(np.abs(np.fft.fftn(img * support)) * (1.0 - MaskFFT) - ModulusFFT * (1.0 - MaskFFT))) / np.sum(np.square(ModulusFFT * (1.0 - MaskFFT)))
    return error


def unwrap_phase(phase, number_of_ffts=6):
    """
    Unwrap the phase in the retrieved image.

    Phase unwrap performed according to the paper:
        Computationally effective 2D and 3D fast phase unwrapping algorithms and their applications to Doppler optical coherence tomography
        E. Pijewska, et al, Opt. Express 10, 1365-1382 (2019)

    Parameters
    ----------
    phase : ndarray
        float type array .
    number_of_ffts : int
        the number of FFT loops which defines algorithms to be used.

    Returns
    -------
    phase : ndarray.
        The unwraped phase.

    """
    dim = len(phase.shape)
    if dim == 2:
        yd, xd = phase.shape
        fy, fx = np.mgrid[0:yd, 0:xd]
        fy = (fy - yd / 2 + 0.5) / yd
        fx = (fx - xd / 2 + 0.5) / xd
        K = np.square(fy) + np.square(fx) + np.finfo(float).eps
    elif dim == 3:
        zd, yd, xd = phase.shape
        fz, fy, fx = np.mgrid[0:zd, 0:yd, 0:xd]
        fz = (fz - zd / 2 + 0.5) / zd
        fy = (fy - yd / 2 + 0.5) / yd
        fx = (fx - xd / 2 + 0.5) / xd
        K = np.square(fz) + np.square(fy) + np.square(fx) + np.finfo(float).eps

    K = np.fft.fftshift(K)

    if number_of_ffts == 4:
        estimated_psi = (np.fft.ifftn(np.fft.fftn(np.imag(np.fft.ifftn(K * np.fft.fftn(np.exp(1j * phase))) / np.exp(1j * phase))) / K))
    elif number_of_ffts == 6:
        estimated_psi = (np.fft.ifftn(np.fft.fftn(((np.cos(phase) * np.fft.ifftn(K * np.fft.fftn(np.sin(phase))) - np.sin(phase) * np.fft.ifftn(K * np.fft.fftn(np.cos(phase)))))) / K))
    elif number_of_ffts == 8:
        estimated_psi = (np.fft.ifftn(np.fft.fftn(np.cos(phase) * (np.fft.ifftn(K * np.fft.fftn(np.sin(phase))))) / K) - np.fft.ifftn(np.fft.fftn(np.sin(phase) * (np.fft.ifftn(K * np.fft.fftn(np.cos(phase))))) / K))

    Q = np.around((np.real(estimated_psi) - phase) / (2 * np.pi))
    phase = phase + 2 * np.pi * Q
    return phase


def phase_corrector(phase, support, cval=0.0):
    """
    Unwrap the phase and remove the phase offset.

    Parameters
    ----------
    phase : ndarray
        phase of the retrieved image.
    support : ndarray
        The support of the retrieved image.
    cval : float, optional
        The phase offset, which the phase is aligned to. The default is 0.0.

    Returns
    -------
    phase : ndarray
        The corrected phase.

    """
    phase = phase * support
    phase = unwrap_phase(phase)
    phase = phase * support
    const = np.sum(phase) / np.sum(support)
    i = 0
    while np.abs(const - cval) > 0.1 and i < 10:
        phase = phase * support
        const = np.sum(phase) / np.sum(support)
        phase = (phase - const + cval) * support
        phase = unwrap_phase(phase)
        i = i + 1
    phase = phase * support
    return phase


def CenterSup(support, img):
    """
    Center the retrieved image according to the support.

    Parameters
    ----------
    support : ndarray
        support of the phase retrieval results.
    img : ndarray
        phase retrieval results (complex density in real space).

    Returns
    -------
    support : ndarray
        centered support.
    img : ndarray
        centered image.

    """
    support_shift = np.around(np.array(support.shape, dtype=float) / 2.0 - 0.5 - measurements.center_of_mass(support))
    for i, shift in enumerate(support_shift):
        support = np.roll(support, round(shift), axis=i)
        img = np.roll(img, round(shift), axis=i)
    return support, img


def Orth3D(PRimage, om, om_step, delta, distance, pixelsize, energy):
    """
    Transfrom the 3D Phase retrieval results from detector space to the orthogonal space.

    Parameters
    ----------
    PRimage : ndarray
        Phase retrieval results.
    om : float
        The omega vaule of the diffraction peak in degree.
    om_step : float
        The step size of the omega scan in degree.
    delta : float
        The 2theta value of the detector in degree.
    distance : float
        Sample detector distance in mm.
    pixelsize : float
        Detector pixel size in mm.
    energy : float
        The energy of the X-ray beam in eV.

    Returns
    -------
    Ortho_Img : ndarray
        The orthonormalizaed reconstruction.
    Ortho_unit : ndarray
        The unit of the reconstruction.

    """
    print("Transforming into the orthoganol coordinates...")
    zd, yd, xd = PRimage.shape
    hc = 1.23984 * 10000.0
    wavelength = hc / energy
    unit = 2.0 * np.pi * pixelsize / wavelength / distance
    om = np.radians(om)
    om_step = np.radians(om_step)
    delta = np.radians(delta)

    # Calculate the new coordinates
    om_C = distance * om_step / pixelsize
    Coords_transform = np.array([[(-np.cos(delta - om) + np.cos(om)) * om_C, -np.cos(delta - om)], [(np.sin(delta - om) + np.sin(om)) * om_C, np.sin(delta - om)]])
    N_matrix = np.array([[zd, 0], [0, yd]]) / float(xd)
    Coords_transform = np.dot(N_matrix, np.transpose(Coords_transform))
    inv_Coords = np.linalg.inv(Coords_transform)
    nz, nx = np.dot(inv_Coords, np.array([zd / 2, 0])) - np.dot(inv_Coords, np.array([0, yd / 2]))
    offset = np.array([zd / 2, yd / 2]) - np.dot(Coords_transform, np.array([nz, nx]))
    Ortho_unit = 2.0 * np.pi / (unit * float(xd)) / 10.0
    Ortho_Img = np.zeros((int(2 * nz), xd, int(2 * nx)), dtype=PRimage.dtype)
    # Interpolate the 3d array
    for i in range(xd):
        Ortho_Img[:, i, :] = affine_transform(PRimage[:, :, i], Coords_transform, offset=offset, output_shape=(int(2 * nz), int(2 * nx)), order=3, cval=0, output=PRimage.dtype)
    return Ortho_Img, Ortho_unit


def Orth2D(PRimage, yd, om, om_step, delta, distance, pixelsize, energy):
    """
    Transfrom the 2D Phase retrieval results in detector space to the orthogonal space.

    Parameters
    ----------
    PRimage : ndarray
        Phase retrieval results.
    yd : int
        The image size in y direction.
    om : float
        the omega vaule of the diffraction peak in degree.
    om_step : float
        the step size of the omega scan in degree.
    delta : float
        the 2theta value of the detector in degree.
    distance : float
        sample detector distance in mm.
    pixelsize : float
        detector pixel size in mm.
    energy : float
        the energy of the X-ray beam in eV.

    Returns
    -------
    Ortho_Img : ndarray
        The orthonormalizaed reconstruction.
    Ortho_unit : ndarray
        The unit of the reconstruction.

    """
    print("Transforming into the orthoganol coordinates...")
    zd, xd = PRimage.shape
    hc = 1.23984 * 10000.0
    wavelength = hc / energy
    unit = 2.0 * np.pi * pixelsize / wavelength / distance
    om = np.radians(om)
    om_step = np.radians(om_step)
    delta = np.radians(delta)

    # Calculate the new coordinates
    om_C = distance * om_step / pixelsize
    Coords_transform = np.array([[(-np.cos(delta - om) + np.cos(om)) * om_C, -np.cos(delta - om)], [(np.sin(delta - om) + np.sin(om)) * om_C, np.sin(delta - om)]])
    N_matrix = np.array([[zd, 0], [0, xd]]) / float(yd)
    Coords_transform = np.dot(N_matrix, np.transpose(Coords_transform))
    inv_Coords = np.linalg.inv(Coords_transform)
    nz, nx = np.dot(inv_Coords, np.array([zd / 2, 0])) - np.dot(inv_Coords, np.array([0, xd / 2]))
    offset = np.array([zd / 2, xd / 2]) - np.dot(Coords_transform, np.array([nz, nx]))
    Ortho_unit = 2.0 * np.pi / (unit * float(yd)) / 10.0
    Ortho_Img = affine_transform(PRimage, Coords_transform, offset=offset, output_shape=(int(2 * nz), int(2 * nx)), order=3, cval=0, output=PRimage.dtype)
    return Ortho_Img, Ortho_unit


def save_to_vti(pathsave, save_arrays, array_names, voxel_size=(1, 1, 1), origin=(0, 0, 0)):
    """
    Save the three dimensional array to the vti file for the visilization with paraview.

    The library of vtk is needed. (https://pypi.org/project/vtk/)

    Parameters
    ----------
    pathsave : str
        path to save the result vti file.
    save_arrays : tuple
        tuple of arrays to be saved.
    array_names : tuple
        Names of the arrays.
    voxel_size : tuple, optional
        The voxel size of the saved array. The default is (1,1,1).
    origin : tuple, optional
        The origin position of the saved arrray. The default is (0,0,0).

    Returns
    -------
    None.

    """
    import vtk
    from vtk.util import numpy_support

    for i in range(len(save_arrays)):
        if i == 0:
            array_to_vtk = save_arrays[i]
            nbz, nby, nbx = array_to_vtk.shape
            image_data = vtk.vtkImageData()
            image_data.SetOrigin(origin[0], origin[1], origin[2])
            image_data.SetSpacing(voxel_size[0], voxel_size[1], voxel_size[2])
            image_data.SetExtent(0, nbz - 1, 0, nby - 1, 0, nbx - 1)

            array_to_vtk = numpy_support.numpy_to_vtk(np.ravel(np.transpose(np.flip(array_to_vtk, 2))))
            pd = image_data.GetPointData()
            pd.SetScalars(array_to_vtk)
            pd.GetArray(i).SetName(array_names[i])
        else:
            array_to_vtk = save_arrays[i]
            array_to_vtk = numpy_support.numpy_to_vtk(np.ravel(np.transpose(np.flip(array_to_vtk, 2))))
            pd.AddArray(array_to_vtk)
            pd.GetArray(i).SetName(array_names[i])
            pd.Update()

    # export data to file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(pathsave)
    writer.SetInputData(image_data)
    writer.Write()
    return
