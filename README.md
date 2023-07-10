# pyCXIM : python scripts for coherent X-ray imaging methods
## Introduction
 pyCXIM stands for python scripts for coherent X-ray imaging methods. 
 It contains modules for:
	- Reading scans at different beamlines, e.g. p10 (desy), p08 (desy)
	- Converting detector images in rocking curves to three dimensional reciprocal space maps for the preparation of phase retrieval process.
	- Performing phase retrieval and simple post-processing.
	- Creating and modifing information file in a text format.

## pyCXIM as a python toolkit
 pyCXIM can be used as a python library with the following main modules:
	1. :mod: 'p08_scan_reader': Read the scan information from p08 beamline, Desy.
	1. :mod: 'p08_scan_reader': Read the scan information from p10 beamline, Desy.
	1. :mod: 'RSM': Convert the two dimensional detector images in the rocking curve to the three dimensional reciprocal space maps.
	1. :mod: 'phase_retrieval': Perform the phase retrieval and simple post processing of the solutions.
	1. :mod: 'Common': Generate the information file.

## Acknowledgment and third party packages
 We would like to acknowledge the BCDI package from Jerome Carnis. Some of the functions were adapted from this package.
 
 
