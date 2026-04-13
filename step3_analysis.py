#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:45:47 2024

@author: beatricecaccherano
"""

#AU Mic Analysis STEP 3

"""
data (or udata) means:
    
    x = detector x pixel integer coordinate
    y = detector y pixel integer coordinate
    r = offset from centre of slit in pixel units (float)
    f = flux of pixel (float)
    e = std error of flux (float)
    w = wavelength in nm (float)
    n = factor used to normalise integrated flux per wavelength (float)  
"""
import pickle
import os
import numpy as np
import scipy as sp 
import time
import matplotlib.pyplot as plt 

import spectroscopy_functions as spec_fun
import plot_functions as plot_func


st = time.time()
root = '/home/beatricecaccherano/Master_Thesis_Analysis/data/'
root1 = '/home/beatricecaccherano/Master_Thesis_Analysis/dataS1/'
root2 = '/home/beatricecaccherano/Master_Thesis_Analysis/dataS2/'
root3 = '/home/beatricecaccherano/Master_Thesis_Analysis/dataS3/'

"""
FLAG TO activate if you are working with files A11, A12,  B11 in order to 
cut the strange structure which compare insiede the image.
"""
file_name ="B34"
delta_w = 0.05
"""
Creation of the list of files whcih you want analyse: 
    Write the name of the first file of the group, the group number (=group_num)
    and the group (=group).
"""
file = 'AUMic000_' + file_name +'.npy'
file1 = 'S1_AUMic000_' + file_name +'.npy'
file2 = 'S2_AUMic_' + file_name +'.npy'

group = '00'
group_num = int(group)

files = [file]
files1 =[file1]
files2 =[file2]
for i in range(0,39):
    group_num += 1
    if(group_num <10):
        group = group[0] + str(group_num)
        file = file[:6] + group + file[8:]
        files.append(file)
        file1 = file1[:9] + group + file1[11:]
        files1.append(file1)
    else:
        group = str(group_num)
        file = file[:6] + group + file[8:]
        files.append(file)
        file1 = file1[:9] + group + file1[11:]
        files1.append(file1)
print(files)
print(files1)
print(files2)
j=0

for file1 in files1:
    file_path = os.path.join(root, files[j])
    udata = np.load(file_path)
    file1_path = os.path.join(root1, file1)
    data_file1 = open(file1_path, 'rb')
    psf_spl, spec_spl, spec, err_spec, weight, wavelength  = pickle.load(data_file1)
    file2_path = os.path.join(root2, file2)
    data_file2 = open(file2_path, 'rb')
    std_flux, group_spec, one_frame_spec, w_shift, w_for_cut, w_telluric, master_wavelength, master_spec, master_spec_spl = pickle.load(data_file2)
    

    w_ind = np.argsort(udata['w'])
    wdata = udata[w_ind]
    r_ind = np.argsort(udata['r'])
    rdata = udata[r_ind]
    
    print("AU Mic code is running with '", files[j], "':" )
    print("and with '", file1, "':" )
    print("and with '", files2, "':" )

    
    """Final image of the residuals:"""
    final_res = (rdata['f']-(spec_spl(rdata['w'])*psf_spl(rdata['r'])))/rdata['e']
    res_data = rdata.copy()
    res_data['f'] = final_res 
    img_res = spec_fun.make_image(res_data)
    #plot_func.plot_image(img_res, res_data, title="Model of Final Residues")
    
    """Correction of the strange structures inside A11, A12, B11"""
    udata = spec_fun.correction_structures(file_name, udata)
            
          
    """Definition of IMAGE of the clumps in term of Radial Velocity and Separtion"""
    spec_fun.image_separation_velocity(res_data, master_wavelength, spec_spl, psf_spl, master_spec_spl, w_telluric,delta_w, j)
    
    """FILES of Spectrum and PSF of the star """
    step3_data = [ res_data, spec_spl, psf_spl ]
    name_file_step3_data = root3+'S3_'+files[j]
    print(name_file_step3_data)
    with open(name_file_step3_data, 'wb') as fp:
        pickle.dump(step3_data, fp)
    j+=1




data_file1.close()
data_file2.close()




















