# -*- coding: utf-8 -*-
"""
Description
Created on Thu Apr 27 13:50:07 2023

@author: renzhe
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def Cut_central(dataset, bs, cut_central_pos='maximum integration', peak_pos=None):
    #Cutting the three dimensional data with the center of mass in the center of the intensity distribution
    if cut_central_pos=='maximum integration':
        peak_pos=np.array([np.argmax(np.sum(dataset, axis=(1,2))), np.argmax(np.sum(dataset, axis=(0,2))), np.argmax(np.sum(dataset, axis=(0,1)))], dtype=int)
        print('finding the centeral position for the cutting')
        bs[0]=int(np.amin([bs[0], peak_pos[0]*0.95, 0.95*(dataset.shape[0]-peak_pos[0])]))
        bs[1]=int(np.amin([bs[1], peak_pos[1]*0.95, 0.95*(dataset.shape[1]-peak_pos[1])]))
        bs[2]=int(np.amin([bs[2], peak_pos[2]*0.95, 0.95*(dataset.shape[2]-peak_pos[2])]))
        intcut=np.array(dataset[(peak_pos[0]-bs[0]):(peak_pos[0]+bs[0]),(peak_pos[1]-bs[1]):(peak_pos[1]+bs[1]), (peak_pos[2]-bs[2]):(peak_pos[2]+bs[2])])
    elif cut_central_pos=='maximum intensity':
        peak_pos=np.unravel_index(np.argmax(dataset), dataset.shape)
        bs[0]=int(np.amin([bs[0], peak_pos[0]*0.95, 0.95*(dataset.shape[0]-peak_pos[0])]))
        bs[1]=int(np.amin([bs[1], peak_pos[1]*0.95, 0.95*(dataset.shape[1]-peak_pos[1])]))
        bs[2]=int(np.amin([bs[2], peak_pos[2]*0.95, 0.95*(dataset.shape[2]-peak_pos[2])]))
        intcut=np.array(dataset[(peak_pos[0]-bs[0]):(peak_pos[0]+bs[0]),(peak_pos[1]-bs[1]):(peak_pos[1]+bs[1]), (peak_pos[2]-bs[2]):(peak_pos[2]+bs[2])])       
    elif cut_central_pos=='weight center':
        peak_pos=np.array(np.around(measurements.center_of_mass(dataset), dtype=int))
        bs[0]=int(np.amin([bs[0], peak_pos[0]*0.95, 0.95*(dataset.shape[0]-peak_pos[0])]))
        bs[1]=int(np.amin([bs[1], peak_pos[1]*0.95, 0.95*(dataset.shape[1]-peak_pos[1])]))
        bs[2]=int(np.amin([bs[2], peak_pos[2]*0.95, 0.95*(dataset.shape[2]-peak_pos[2])]))
        intcut=np.array(dataset[(peak_pos[0]-bs[0]):(peak_pos[0]+bs[0]),(peak_pos[1]-bs[1]):(peak_pos[1]+bs[1]), (peak_pos[2]-bs[2]):(peak_pos[2]+bs[2])])
        print('cut according to the weight center')
        i=0
        torlerence=0.5
        while not np.allclose(measurements.center_of_mass(intcut), np.array(bs, dtype=float)-0.5, atol=torlerence):
            peak_pos=np.array(peak_pos+np.around(measurements.center_of_mass(intcut)-np.array(bs, dtype=float)+0.5), dtype=int)
            bs[0]=int(np.amin([bs[0], peak_pos[0]*0.95, 0.95*(dataset.shape[0]-peak_pos[0])]))
            bs[1]=int(np.amin([bs[1], peak_pos[1]*0.95, 0.95*(dataset.shape[1]-peak_pos[1])]))
            bs[2]=int(np.amin([bs[2], peak_pos[2]*0.95, 0.95*(dataset.shape[2]-peak_pos[2])]))
            intcut=np.array(dataset[(peak_pos[0]-bs[0]):(peak_pos[0]+bs[0]),(peak_pos[1]-bs[1]):(peak_pos[1]+bs[1]), (peak_pos[2]-bs[2]):(peak_pos[2]+bs[2])])
            i+=1
            if i==5:
                print("Loosen the constrain for the weight center cutting")
                torlerence=1
            elif i>8:
                print("could not find the weight center for the cutting")
                break
    elif cut_central_pos=='given':
        if peak_pos is None:
            print('Could not find the given position for the cutting, please check it again!')
            peak_pos=np.array((dataset.shape)/2, dtype=int)
        else:
            peak_pos=np.array(peak_pos, dtype=int)
        bs[0]=int(np.amin([bs[0], peak_pos[0]*0.95, 0.95*(dataset.shape[0]-peak_pos[0])]))
        bs[1]=int(np.amin([bs[1], peak_pos[1]*0.95, 0.95*(dataset.shape[1]-peak_pos[1])]))
        bs[2]=int(np.amin([bs[2], peak_pos[2]*0.95, 0.95*(dataset.shape[2]-peak_pos[2])]))
        intcut=np.array(dataset[(peak_pos[0]-bs[0]):(peak_pos[0]+bs[0]),(peak_pos[1]-bs[1]):(peak_pos[1]+bs[1]), (peak_pos[2]-bs[2]):(peak_pos[2]+bs[2])])
    return intcut, peak_pos, bs

def plotandsave(RSM_int, q_origin, unit, pathsavetmp, qmax=np.array([])):
    dz, dy, dx = RSM_int.shape
    qz = np.arange(dz) * unit + q_origin[0]
    qy = np.arange(dy) * unit + q_origin[1]
    qx = np.arange(dx) * unit + q_origin[2]
    # save the qx qy qz cut of the 3D intensity
    print('Saving the qx qy qz cuts......')
    plt.figure(figsize=(12, 12))
    pathsaveimg = pathsavetmp % ('cutqz')
    if len(qmax) == 0:
        plt.contourf(qx, qy, np.log10(np.sum(RSM_int, axis=0) + 1.0), 150, cmap='jet')
    else:
        plt.contourf(qx, qy, np.log10(RSM_int[qmax[0], :, :] + 1.0), 150, cmap='jet')
    plt.xlabel(r'Q$_x$ ($1/\AA$)', fontsize=14)
    plt.ylabel(r'Q$_y$ ($1/\AA$)', fontsize=14)
    plt.axis('scaled')
    plt.savefig(pathsaveimg)
    plt.show()
    # plt.close()

    plt.figure(figsize=(12, 12))
    pathsaveimg = pathsavetmp % ('cutqy')
    if len(qmax) == 0:
        plt.contourf(qx, qz, np.log10(np.sum(RSM_int, axis=1) + 1.0), 150, cmap='jet')
    else:
        plt.contourf(qx, qz, np.log10(RSM_int[:, qmax[1], :] + 1.0), 150, cmap='jet')
    plt.xlabel(r'Q$_x$ ($1/\AA$)', fontsize=14)
    plt.ylabel(r'Q$_z$ ($1/\AA$)', fontsize=14)
    plt.axis('scaled')
    plt.savefig(pathsaveimg)
    plt.show()
    # plt.close()

    plt.figure(figsize=(12, 12))
    pathsaveimg = pathsavetmp % ('cutqx')
    if len(qmax) == 0:
        plt.contourf(qy, qz, np.log10(np.sum(RSM_int, axis=2) + 1.0), 150, cmap='jet')
    else:
        plt.contourf(qy, qz, np.log10(RSM_int[:, :, qmax[2]] + 1.0), 150, cmap='jet')
    plt.xlabel(r'Q$_y$ ($1/\AA$)', fontsize=14)
    plt.ylabel(r'Q$_z$ ($1/\AA$)', fontsize=14)
    plt.axis('scaled')
    plt.savefig(pathsaveimg)
    plt.show()
    # plt.close()
    return

def plotandsave2(RSM_int, pathsavetmp, mask=np.array([])):
    mask=np.ma.masked_where(mask==0, mask)
    dz, dy, dx=RSM_int.shape
    #save the qx qy qz cut of the 3D intensity
    print('Saving the qx qy qz cuts......')
    plt.figure(figsize=(12,12))
    pathsaveimg=pathsavetmp%('cutqz')
    plt.imshow(np.log10(RSM_int[int(dz/2), :,:]+1.0), cmap='Blues')
    if mask.ndim!=1:
        plt.imshow(mask[int(dz/2), :, :], cmap='Reds', alpha=0.5, vmin=0, vmax=1)
    plt.xlabel(r'Q$_x$', fontsize= 14)
    plt.ylabel(r'Q$_y$', fontsize= 14)
    plt.axis('scaled')
    plt.savefig(pathsaveimg)
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,12))
    pathsaveimg=pathsavetmp%('cutqy')
    plt.imshow(np.log10(RSM_int[:, int(dy/2), :]+1.0), cmap='Blues')
    if mask.ndim!=1:
        plt.imshow(mask[:, int(dy/2), :], cmap='Reds', alpha=0.5, vmin=0, vmax=1)
    plt.xlabel(r'Q$_x$', fontsize= 14)
    plt.ylabel(r'Q$_z$', fontsize= 14)
    plt.axis('scaled')
    plt.savefig(pathsaveimg)
    plt.show()
    plt.close()
    
    plt.figure(figsize=(12,12))
    pathsaveimg=pathsavetmp%('cutqx')
    plt.imshow(np.log10(RSM_int[:,:,int(dx/2)]+1.0), cmap='Blues')
    if mask.ndim!=1:
        plt.imshow(mask[:, :, int(dx/2)], cmap='Reds', alpha=0.5, vmin=0, vmax=1)
    plt.xlabel(r'Q$_y$', fontsize= 14)
    plt.ylabel(r'Q$_z$', fontsize= 14)
    plt.axis('scaled')
    plt.savefig(pathsaveimg)
    plt.show()
    plt.close()
    return

def RSM2vti(pathsave, RSM_dataset, filename, RSM_unit, origin=(0, 0, 0)):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    imdata = vtk.vtkImageData()
    imdata.SetOrigin(origin[0], origin[1], origin[2])
    imdata.SetSpacing(RSM_unit, RSM_unit, RSM_unit)
    imdata.SetDimensions(RSM_dataset.shape)

    RSM_vtk = numpy_to_vtk(np.ravel(np.transpose(RSM_dataset)), deep=True, array_type=vtk.VTK_DOUBLE)

    imdata.GetPointData().SetScalars(RSM_vtk)
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(pathsave)
    writer.SetInputData(imdata)

    writer.Write()
    return
