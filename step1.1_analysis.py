#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 15:34:04 2024

@author: beatricecaccherano
"""

import pickle
import os
import numpy as np
import scipy as sp 
import time
import matplotlib.pyplot as plt 
from scipy.optimize import minimize_scalar, minimize

import spectroscopy_functions as spec_fun
import plot_functions as plot_func

st = time.time()
root = 'C:/Users/alice/Documents/Stage Suède/data/'
root1 = 'C:/Users/alice/Documents/Stage Suède/data1/'
root1_1 = 'C:/Users/alice/Documents/Stage Suède/data1_1/'

file_name ="B11"

"""
Creation of the list of files whcih you want analyse: 
    Write the name of the first file of the group, the group number (=group_num)
    and the group (=group).
"""
file = 'AUMic000_' + file_name +'.npy'
file1 = 'S1_AUMic000_' + file_name +'.npy'
file1_1 = 'S1_1_AUMic_' + file_name +'.npy'
group = '00'
group_num = int(group)

files = [file]
files1_1 =[file1_1]
for i in range(0,3):
    group_num += 1
    if(group_num <10):
        group = group[0] + str(group_num)
        file = file[:6] + group + file[8:]
        files.append(file)
        file1_1 = file1_1[:9] + group + file1_1[11:]
        files1_1.append(file1_1)
    else:
        group = str(group_num)
        file = file[:6] + group + file[8:]
        files.append(file)
        file1_1 = file1_1[:9] + group + file1_1[11:]
        files1_1.append(file1_1)
print(files)
print(files1_1)

for file in files:
    file_path = os.path.join(root, file)
    udata = np.load(file_path)
    file1 = 'S1_' + file
    file1_path = os.path.join(root1, file1)
    data_file1 = open(file1_path, 'rb')
    psf_spl, spec_spl, spec, err_spec, weight, wavelength = pickle.load(data_file1)
    
    """ Flux Normalization"""
    ind = np.argsort(udata['w'])
    wdata = udata[ind] 
    data = wdata.copy()
    data['f'] = data['f']/spec_spl(data['w'])
    
    
    """Definition of the wavelength range"""
    num_delta_w =2*int(np.max(data['w'])- np.min(data['w']))+1
    w0_array = np.zeros(num_delta_w)
    dr_array =  np.zeros(num_delta_w)
    w0 = np.min(data['w'])+0.25
    dw = 0.5
    
    """Selection of the inner region of the PSF"""
    ind_max = np.argmax(psf_spl(data['r']))
    r0 = data['r'][ind_max]
    print(r0)
    delta_r = 15.0
    r_data = spec_fun.dw_selection(data, r0, delta_r, type_data = 'r')
    img = spec_fun.make_image(r_data)
    plot_func.plot_image(img, r_data, fmin = np.min(r_data['f']), fmax = np.max(r_data['f']), title="Data")
    
    save_chi2 = np.zeros(num_delta_w)
    """Estimation of the best dr in each wavelength range: """
    for i in range(0,num_delta_w):
        sel_data = spec_fun.dw_selection(r_data, w0,dw, type_data = 'w')
        #img = spec_fun.make_image(sel_data)
        #plot_func.plot_image(img, sel_data, title="Data")
        
        w =sel_data['w']
        print(len(w))
        r =sel_data['r']
        flux =sel_data['f']
        err_flux =sel_data['e']
        
        """fig, ax = plt.subplots()
        plt.title('PSF in dw ')
        plt.xlabel('flux')
        plt.ylabel('r [pixel unit]')
        ax.plot(r, flux,'.',color = 'green', ms=2)"""
        
        chi2 = lambda dr: np.sum((flux-psf_spl(r-dr))**2.0/err_flux)
        
        rr = np.linspace(-3.0,3.0,100)
        chi22 = np.zeros_like(rr)
        """for k in range(len(rr)):
            chi22[k] = chi2(rr[k])
        fig, ax = plt.subplots()
        ax.plot(rr, chi22,'.',color = 'green', ms=2)"""
        
        
        res = minimize_scalar(chi2, bounds=(-2.0, 2.0), method='bounded')
        print(res.message)
        print('Minumum: ', res.fun)
        print('Minimizer: ',res.x)
        w0_array[i] = w0
        dr_array[i] = res.x
        delta_r = res.x
        
        w0+=0.5   
        print('*****************')
        
    print('*****************')
    print(w0_array)
    knots = np.linspace(w0_array[3], w0_array[-3],7)
    dr_spl =sp.interpolate.LSQUnivariateSpline(w0_array, dr_array, knots)
    plot_func.plot_dr_corr(w0_array, dr_array, wdata, dr_spl)
    

    """r CORRECTION:"""
    print(wdata['r'])
    wdata['r'] -= dr_spl(wdata['w'])
    print(dr_spl(wdata['w']))
    print(wdata['r'])
    
    
    
    
    data_file1.close()

    """DATA with r-correction"""
    name_file_step1_1_data = root1_1+'S1_1_'+file
    with open(name_file_step1_1_data, 'wb') as fp:
        pickle.dump(wdata, fp)
    
        