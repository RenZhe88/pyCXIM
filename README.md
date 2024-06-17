# pyCXIM : python scripts for coherent X-ray imaging methods
## For any question, please contact the authors Zhe Ren, Han Xu.
 * email01: renzhe@ihep.ac.cn  
 * email02: renzhetu001@gmail.com  
 * email03: xuhan@ihep.ac.cn  
## Introduction
 pyCXIM stands for python scripts for coherent X-ray imaging methods. 
 It contains modules for:  
	+  Reading scans at different beamlines, e.g. p10 (desy), p08 (desy)£¬ nanoMAX(MAX IV), 1w1a(BSRF).  
	+ Converting detector images in rocking curves to three dimensional reciprocal space maps for the preparation of phase retrieval process.  
	+ Performing phase retrieval and simple post-processing.  
	+ Creating and modifing information file in a text format.  

## pyCXIM as a python toolkit
 pyCXIM can be used as a python library with the following main modules:  
	- 1. :mod: 'scan_reader': Read the scand from different beamlines.  
	- 2. :mod: 'RSM': Convert the two dimensional detector images to the three dimensional reciprocal space maps, and perform calibration for the six cirlce diffractometers.  
	- 3. :mod: 'phase_retrieval': Perform the phase retrieval and simple post processing of the solutions.  
	- 4. :mod: 'Common': Generate the information file.  

In the scripts folder, you can find scripts for different data treatment processes.  

## Acknowledgment and third party packages
 We would like to acknowledge the BCDI package from Jerome Carnis. Some of the functions were adapted from this package.
 
 
