#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 21:48:53 2024

@author: beatricecaccherano
"""

#AU Mic ANALYSIS STEP 1.
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

root1_1 = 'C:/Users/alice/Documents/Stage Suède/data1_1/'
root1 = 'C:/Users/alice/Documents/Stage Suède/data1/'
#files = ['AUMic000_A11.npy','AUMic001_A11.npy','AUMic002_A11.npy','AUMic003_A11.npy']


"""
FLAG TO activate if you are working with files A11, A12, B11 in order to 
cut the strange structure which compare insiede the image.
"""
file_name ="B11"

#SESSION 1: group of files analysis
#SESSION 2: single file analysis
SESSION = 1

"""
Creation of the list of files whcih you want analyse: 
    Write the name of the first file of the group, the group number (=group_num)
    and the group (=group).
"""
file = 'S1_1_AUMic000_' + file_name +'.npy'
file1 = 'S1_AUMic000_' + file_name +'.npy'
group = '00'
group_num = int(group)

files = [file]
files1 =[file1]
FWHM = np.empty(40)
if(SESSION==1):
    for i in range(0,3):
        group_num += 1
        if(group_num <10):
            group = group[0] + str(group_num)
            file = file[:11] + group + file[13:]
            files.append(file)
            file1 = file1[:9] + group + file1[11:]
            files1.append(file1)
        else:
            group = str(group_num)
            file = file[:11] + group + file[13:]
            files.append(file)
            file1 = file1[:9] + group + file1[11:]
            files1.append(file1)
    print(files)    
    len_spectrum = len(files) 
    for file, file1 in zip(files, files1):
        file_path = os.path.join(root1_1, file)
        udata = np.load(file_path, allow_pickle=True)#data with corrected r and sorted by w
        file1_path = os.path.join(root1, file1)
        with open(file1_path, 'rb') as data_file1:
            psf_spl, spec_spl0, spec, err_spec, weight, wavelength = pickle.load(data_file1)
        
        print("AU Mic code is running with '", file, "':" )
        img = spec_fun.make_image(udata)
        plot_func.plot_image(img, udata, fmin=np.min(udata['f']), fmax=np.max(udata['f']), title="Data")
        
        
        """
        PSF define by RSS: this PSF is useful to have the first rough PSF function 
        to start the analysis.
        """
        min_slit = int(np.min(udata['r']))
        max_slit = int(np.max(udata['r']))
        knots=np.arange(min_slit,max_slit,1.0)
        psf_spl, psf, err_psf= spec_fun.fit_psf_rough(udata,spec_spl0,knots)
        title = "Rough PSF"
        plot_func.plot_psf(udata, psf, psf_spl, spec_spl0, title)
        #print(len(udata['r']))
                      
        ind = np.argsort(udata['r'])
        rdata = udata[ind] 
        ind_max = np.argmax(psf_spl(udata['r']))
        r_star = udata['r'][ind_max] #position of the star 
        #print('r_star',r_star)
        
        
        """Stellar Spectrum:"""
        M = 3
        min_slit = int(np.min(udata['r']))
        max_slit = int(np.max(udata['r']))
        for i in range(0,M):
            spec_spl, spec, err_spec, weight, wavelength =  spec_fun.fit_spec(udata,psf_spl,r_star,dr=5.0)
            psf_spl, psf, err_psf =  spec_fun.fit_psf(udata,spec_spl,r_star)
           
        """Final PSF and Spectrum plots:""" 
        plot_func.plot_spectrum(udata, spec, spec_spl,flag=0)
        title1 = "Final PSF"
        plot_func.plot_psf(udata, psf, psf_spl, spec_spl, title1)
        
        
        """FILES of Spectrum and PSF of the star """
        step1_funs = [psf_spl, spec_spl, spec, err_spec, weight, wavelength]
        name_file_step1_data = os.path.join(root1, file.replace('S1_1_', 'S1_', 1))
        with open(name_file_step1_data, 'wb') as fp:
            pickle.dump(step1_funs, fp)
        
            
    


    
    
    
    
    
    