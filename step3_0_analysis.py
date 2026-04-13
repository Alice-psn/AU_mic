#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 11:55:41 2024

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
from astropy import constants as const

import spectroscopy_functions as spec_fun
import plot_functions as plot_func


st = time.time()
root = '/home/beatricecaccherano/Master_Thesis_Analysis/data/'
root1 = '/home/beatricecaccherano/Master_Thesis_Analysis/dataS1/'
root2 = '/home/beatricecaccherano/Master_Thesis_Analysis/dataS2_old/'
root3 = '/home/beatricecaccherano/Master_Thesis_Analysis/dataS3_old/'
root4 = '/home/beatricecaccherano/Master_Thesis_Analysis/dataS4_old/'

"""
FLAG TO activate if you are working with files A11, A12,  B11 in order to 
cut the strange structure which compare insiede the image.
"""
file_name ="A23"

"""
Useful variable to produce the final image: velocity- separation
"""
vmax = 100.0
vmin = -100.0
lenv = int(vmax-vmin+1)
vel_array = np.linspace(vmin,vmax,lenv)
aumic_name = root + 'AUMic000_' + file_name + '.npy'
aumic = np.load(aumic_name, allow_pickle=True)
rmin, rmax = -42.0, 42.0 #np.min(aumic['r']) , np.max(aumic['r'])
r_position = np.linspace(rmin, rmax, 90)
lenr = 90
wmin, wmax = np.min(aumic['w']) , np.max(aumic['w'])
vel_c = const.c.to('km/s').value
print(vel_c)
delta_w = 0.15
print(delta_w)
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

img_ccf_ERF_n1 = np.empty((9, lenr, lenv))
img_ccf_ERF_n2 = np.empty((11, lenr, lenv))
img_ccf_ERF_n3 = np.empty((10, lenr, lenv))
img_ccf_ERF_n4 = np.empty((10, lenr, lenv))
n1=0
n2=0
n3=0
n4=0
j=0

for file1 in files1:
    file_path = os.path.join(root, files[j])
    udata = np.load(file_path, allow_pickle=True)
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
    plot_func.plot_image(img_res, res_data, fmin=-10.0 , fmax=10.0, title="Model of Final Residues")
    
    """Correction of the strange structures inside A11, A12, B11"""
    res_data = spec_fun.correction_structures(file_name, res_data)
            
    print('MIN r:', np.min(res_data['r'])) 
    print('MAX r:', np.max(res_data['r'])) 
    """Definition of IMAGE of the clumps in term of Radial Velocity and Separtion"""
    img_ccf_ERF  = spec_fun.image_separation_velocity(res_data, master_wavelength, spec_spl, psf_spl, master_spec_spl, w_telluric, delta_w, j)
    
    
    if(j==1 or j==2 or j==3 or j==4 or j==5 or j==6 or j==7 or j==8 or j==9):#grade A
        img_ccf_ERF_n1[n1] =  img_ccf_ERF
        n1+=1
    if(j==10 or j==11 or j==12 or j==13 or j==14 or j==15 or j==16 or j==17 or
       j==18 or j==19 or j==27): #grade A
        img_ccf_ERF_n2[n2] = img_ccf_ERF
        n2+=1
    if(j==0 or j==20 or j==21 or j==22 or j==23 or j==24 or j==25 or j==26 or 
       j==28 or j==29): #grade C
        img_ccf_ERF_n3[n3] = img_ccf_ERF
        n3+=1
    if(j==30 or j==31 or j==32 or j==33 or j==34 or j==35 or j==36 or j==37 or
       j==38 or j==39): #grade B
        img_ccf_ERF_n4[n4] = img_ccf_ERF
        n4+=1
        
        
    """FILES of Spectrum and PSF of the star """
    step3_data = [ res_data, spec_spl, psf_spl ]
    name_file_step3_data = root3+'S3_'+files[j]
    print(name_file_step3_data)
    with open(name_file_step3_data, 'wb') as fp:
        pickle.dump(step3_data, fp)
    j+=1
    

step4_data =  [ img_ccf_ERF_n1, img_ccf_ERF_n2, img_ccf_ERF_n3, img_ccf_ERF_n4 ]   
name_file_step4_data = root4+'S4_'+file_name+ '.npy'     
print(name_file_step4_data)
with open(name_file_step4_data, 'wb') as fp:
    pickle.dump(step4_data, fp)     
    


data_file1.close()
data_file2.close()












