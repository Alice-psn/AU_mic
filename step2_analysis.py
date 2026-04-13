#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:14:02 2024

@author: beatricecaccherano
"""
#AU Mic Analysis STEP 2

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
from astropy.time import Time
from astropy import time, coordinates as coord, units as u


import spectroscopy_functions as spec_fun
import plot_functions as plot_func
#from Master_Thesis_Analysis import step1_analysis as step1

#st = time.time()
root = '/home/beatricecaccherano/Master_Thesis_Analysis/data/'
root1 = '/home/beatricecaccherano/Master_Thesis_Analysis/dataS1/'
root2 = '/home/beatricecaccherano/Master_Thesis_Analysis/dataS2/'

"""
FLAG TO activate if you are working with files A11, A12,  B11 in order to 
cut the strange structure which compare insiede the image.
"""
file_name ="B14"

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



j=0

len_spectrum = len(files1) 
vel_shift = np.empty(40)
vel = np.empty(40)
FWHM = np.empty(40)
FWHM_for= np.empty(40)
barycorr = np.zeros(40, dtype=object) 
group_spec = np.empty([40,3000])
group_psf =  np.empty([40,3000])
one_frame_spec = np.empty([40,3000])

master_wavelength = [] #np.empty([40,3000])
master_spec = []
master_weight = []
"""AUMic coordinates:"""
aumic = coord.SkyCoord("20:45:09.5324974119", "-31:20:27.237889841", 
                        unit=(u.hourangle, u.deg), frame='icrs')
eso = coord.EarthLocation.from_geodetic(lat = 70.416666667*u.deg , 
                                        lon=-24.666667*u.deg, height=2635*u.m) 



for file1 in files1:
    file_path = os.path.join(root, files[j])
    udata = np.load(file_path)
    file1_path = os.path.join(root1, file1)
    data_file1 = open(file1_path, 'rb')
    psf_spl, spec_spl, spec, err_spec, weight, wavelength = pickle.load(data_file1)
    
    w_ind = np.argsort(udata['w'])
    wdata = udata[w_ind]
    r_ind = np.argsort(udata['r'])
    rdata = udata[r_ind]
    
    print("AU Mic code is running with '", files[j], "':" )
    print("and with '", file1, "':" )
    
    """Studing of the DATA NOISY through the FWHM of the psf"""  
    FWHM[j], FWHM_for[j], star_position = spec_fun.FWHM_estimation(udata, psf_spl)
    
    
    print("w of udata", wdata['w'])
         
    """Studing of OBSERVATION EPOCHS and TELLURIC LINES """    
    w = np.linspace(wdata['w'][2000],wdata['w'][-2000], 3000)
    print("w of file1: ",w)
    r = np.linspace(np.min(udata['r']),np.max(udata['r']), 3000)
    w_shift = np.empty([40,3000])
    vel_array = np.linspace(-10.0,10.0,2000)
    group_spec[j,:]=spec_spl(w)#array to produce image of all the spectrums
    group_psf[j,:]=psf_spl(r)
    if(j==0):
        reference_spec = spec_spl(w)
        #mean_reference_spec = np.mean(reference_spec)
        #reference_spec/= mean_reference_spec
        ccf_array=spec_fun.cross_correlation_velocity_clump(w, reference_spec, 
                                                        spec_spl, vel_array)
        #plot_func.plot_spectrum(udata, reference_spec, spec_spl)
        mean_ccf = np.mean(ccf_array)
        #plot_func.plot_ccf(vel_array, ccf_array/mean_ccf)
        ind_max = ccf_array.argmax()
        vel_shift[j] = vel_array[ind_max]
        barycorr= spec_fun.Barycentric_Correction(aumic, eso, j)

        vel[j] = -barycorr.value  
        print(vel[j])
        wavelength_shift = spec_fun.doppler_shift(wavelength, -vel[j])
        w_shift[j,:] = spec_fun.doppler_shift(w, vel[j])
        one_frame_spec[j,:]=spec_spl(w_shift[j,:])
    else:
        
        ccf_array=spec_fun.cross_correlation_velocity_clump(w, reference_spec, 
                                                        spec_spl, vel_array)
        #plot_func.plot_spectrum(udata, reference_spec, spec_spl)
        #plot_func.plot_ccf(vel_array, ccf_array)
        ind_max = ccf_array.argmax()
        vel_shift[j] = vel_array[ind_max]
        barycorr= spec_fun.Barycentric_Correction(aumic, eso, j)
        print(barycorr)
        vel[j] = -barycorr.value 
        print(vel[j])
        wavelength_shift = spec_fun.doppler_shift(wavelength, -vel[j])
        w_shift[j,:] = spec_fun.doppler_shift(w, vel[j])
        one_frame_spec[j,:]=spec_spl(w_shift[j,:])#array to produce the image of all the spectrum in an unique frame
    
    print(len(spec))
    master_spec.extend(spec/np.median(spec))
    master_wavelength.extend(wavelength_shift)    
    master_weight.extend(weight)
    
    
    j+=1



print(np.asarray(master_spec).shape)
print(len(master_wavelength))

"""PLOT of the Velocity shift:"""
#plot_func.plot_velocity_shift(vel_shift)   
"""PLOT of the Velocity shift corrected:"""
#plot_func.plot_velocity_shift(vel) 

"""PLOT of all the spectrums of the group"""
group_spec/=np.median(group_spec, axis = 1)[:,None]#normalization
plot_func.plot_image_group_spec(group_spec,wdata)
plot_func.plot_group_spec(group_spec,wdata,len_spectrum)
plot_func.plot_group_psf(group_psf,udata,len_spectrum)


"""PLOT of all the spectrums in an UNIQUE REFERENCE FRAME"""
one_frame_spec/=np.median(one_frame_spec, axis = 1)[:,None]#normalization
plot_func.plot_image_group_spec(one_frame_spec,wdata)
plot_func.plot_group_spec(one_frame_spec,wdata,len_spectrum)


"""TELLURIC LINES detection:"""
std_flux = np.std(one_frame_spec, axis=0)
mean_std_flux = np.mean(std_flux)
print(len(std_flux))
w_for_cut = np.linspace(wdata['w'][2000],wdata['w'][-2000], 3000)

plot_func.plot_std(wdata,std_flux, mean_std_flux)


limit =0.1# mean_std_flux 
dw = 0.1
w_no_tell, std_flux, w_telluric = spec_fun.telluric_lines_cut(w_for_cut, std_flux, limit , dw)

    
data_file1.close()
#PLOT of the standard deviation after removal:
plot_func.plot_std_no_tell(w_no_tell, w_telluric ,std_flux, limit)  
    
w_copy = np.linspace(wdata['w'][2000],wdata['w'][-2000], 3000)
cut_w = w_copy.copy()
null_w = np.zeros(3000)
for k in range(len(w_telluric)):
    cut_w = np.where(np.abs(cut_w-w_telluric[k])>dw, cut_w ,null_w) 
    sel = np.abs(w_copy-w_telluric[k])>dw
    w_copy  = w_copy[sel]
    
sel_cut_w = cut_w!=0


"""PLOT of the Spectrums after the telluric lines removal:"""
cut_spec =[]
cut_wavelength =[]
#Mask
for j in range(0,40):
    cut_spec.append(one_frame_spec[j,:][sel_cut_w])
    cut_wavelength.append(w_for_cut[sel_cut_w])


plot_func.plot_spectrum_after_cut(cut_spec, cut_wavelength, len_spectrum)  
    
"""Definition of the MASTER spectrum: """

master_wavelength, master_spec, master_spec_spl = spec_fun.master_spectrum_fit(master_wavelength, master_spec, master_weight)
plot_func.plot_master_spectrum(master_wavelength, master_spec, master_spec_spl)



data_file1.close()
"""FILES of EPOCHS and TELLURIC LINES"""
step2_data = [std_flux, group_spec, one_frame_spec, w_shift, w_for_cut, w_telluric, master_wavelength, master_spec, master_spec_spl]
name_file_step2_data = root2+file2
with open(name_file_step2_data, 'wb') as fp:
    pickle.dump(step2_data, fp)
    





