#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 15:31:15 2024

@author: beatricecaccherano
"""
#AU Mic ANALYSIS STEP 1
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
root = 'C:/Users/alice/Documents/Stage Suède/data/'
root1 = 'C:/Users/alice/Documents/Stage Suède/data1/'
#files = ['AUMic000_A11.npy','AUMic001_A11.npy','AUMic002_A11.npy','AUMic003_A11.npy']


"""
FLAG TO activate if you are working with files A11, A12,  B11 in order to 
cut the strange structure which compare insiede the image.
"""
file_name ="B12"

#SESSION 1: group of files analysis
#SESSION 2: single file analysis
SESSION = 1

"""
Creation of the list of files whcih you want analyse: 
    Write the name of the first file of the group, the group number (=group_num)
    and the group (=group).
"""
file = 'AUMic000_' + file_name +'.npy'
group = '00'
group_num = int(group)

files = [file]
FWHM = np.empty(40)
j=0
if(SESSION==1):
    for i in range(0,3):
        group_num += 1
        if(group_num <10):
            group = group[0] + str(group_num)
            file = file[:6] + group + file[8:]
            files.append(file)
        else:
            group = str(group_num)
            file = file[:6] + group + file[8:]
            files.append(file)
    print(files)    
    len_spectrum = len(files) 
    for file in files:
        file_path = os.path.join(root, file)
        udata = np.load(file_path)
        
        print("AU Mic code is running with '", file, "':" )
        img = spec_fun.make_image(udata)
        plot_func.plot_image(img, udata, fmin=np.min(udata['f']), fmax=np.max(udata['f']), title="Data")
            
        """Cleaning data from OUTLIERS:"""
        clean_data = spec_fun.clean_outliers(udata)
        img = spec_fun.make_image(clean_data)
        #plot_func.plot_image(img, clean_data, title="Data")
        
        """Rough Stellar Spectrum - RSS"""
        interp_spectrum, err_interp_spectrum = spec_fun.rough_spectrum(clean_data,0)
        
        """
        PSF define by RSS: this PSF is useful to have the first rough PSF function 
        to start the analysis.
        """
        min_slit = int(np.min(clean_data['r']))
        max_slit = int(np.max(clean_data['r']))
        knots=np.arange(min_slit,max_slit,1.0)
        psf_spl, psf, err_psf= spec_fun.fit_psf_rough(clean_data,interp_spectrum,knots)
        plot_func.plot_psf(clean_data, psf, psf_spl, interp_spectrum, file)
                      
        
        """SIGMA CLIPPING METHOD:"""
        N = 2
                        
        """Model of Data:"""
        model_data = udata.copy()#copy of data
        model_data['f'] = (interp_spectrum(udata['w'])*psf_spl(udata['r']))# replace the flux with the MODEL
        img = spec_fun.make_image(model_data)#image of the model data
        #plot_func.plot_image(img, model_data, title="Model of Data")
        
        """Model of Residues:"""
        res = (udata['f']-(interp_spectrum(udata['w'])*psf_spl(udata['r'])))/udata['e']
        
        res_data = udata.copy() #copy of data
        res_data['f'] = res # replace the flux with the residuals
        img_res = spec_fun.make_image(res_data)#create an image of the residuals
        plot_func.plot_image(img_res, res_data, title="Model of Residues")
            
        sel = spec_fun.sigma_clipping(res, niter=N)
        
        """Data after the clipping method:"""
        data = udata[sel]    
        ind = np.argsort(data['r'])
        rdata = data[ind] #modif
        img_res = spec_fun.make_image(data)#image of the model data
        plot_func.plot_image(img_res, data, fmin=np.min(data['f']), fmax=np.max(data['f']), title="Model of Data after the clipping")
        ind_max = np.argmax(psf_spl(data['r']))
        r_star = data['r'][ind_max] #position of the star 
        print('r_star',r_star)

        """Stellar Spectrum:"""
        M = 3
        min_slit = int(np.min(data['r']))
        max_slit = int(np.max(data['r']))
        for i in range(0,M):
            spec_spl, spec, err_spec, weight, wavelength =  spec_fun.fit_spec(data,psf_spl,r_star,dr=5.0)
            psf_spl, psf, err_psf =  spec_fun.fit_psf(rdata,spec_spl,r_star)
         
           
        """Final PSF and Spectrum plots:""" 
        plot_func.plot_spectrum(data, spec, spec_spl,flag=0)
        plot_func.plot_psf(rdata, psf, psf_spl, spec_spl, file)
        
        
        j+=1
        
        """FILES of Spectrum and PSF of the star """
        step1_funs = [psf_spl, spec_spl, spec, err_spec, weight, wavelength ]
        name_file_step1_data = root1+'S1_'+file
        with open(name_file_step1_data, 'wb') as fp:
            pickle.dump(step1_funs, fp)
            
    


    
    
    
    
    
    