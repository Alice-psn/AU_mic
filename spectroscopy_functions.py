#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 14:54:40 2024

@author: beatricecaccherano
#Set of Functions to Analyse Spectroscopy data:
"""
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
import numpy as np
import matplotlib.pyplot as plt 
import scipy as sp 
import warnings
from random import uniform, choice
from astropy.constants import c
from astropy.time import Time
from astropy import time, coordinates as coord, units as u

import plot_functions as plot_func

#aumic = np.load('/home/beatricecaccherano/Master_Thesis_Analysis/blob.npy')  
aumic = np.load('C:/Users/alice/Documents/Stage Suède/data/AUMic000_B14.npy')

#Set some useful variable:
    
max_slit = int(np.max(aumic['r']))
min_slit = int(np.min(aumic['r']))
vmin = -75.0
vmax = 75.0 
    

#RESOLUTION SPECTRUM:

spec_resolution = 0.015

#print("Spectrum Resolution: ",spec_resolution)

"""AUMic coordinates:"""
aumic = coord.SkyCoord("20:45:09.5324974119", "-31:20:27.237889841", 
                        unit=(u.hourangle, u.deg), frame='icrs')
eso = coord.EarthLocation.from_geodetic(lat = 70.416666667*u.deg , 
                                        lon=-24.666667*u.deg, height=2635*u.m) 



#The ROUGH Spectrum funtion define spectrum from original data and plot it
def rough_spectrum(data, plot_flag=0):
    sum_err = 0.0
    sum_spec = 0.0
    err_spectrum = []
    spectrum = []
    wavelength = []
    for i in range (np.min(data['x']),np.max(data['x'])):
        sel_col = data['x'] == i
        n_ele = np.sum(sel_col)
        w_med = np.median(data['w'][sel_col])
        for j in range(0, n_ele):
            sum_spec += data['f'][sel_col][j]
            sum_err += data['e'][sel_col][j]**2
        spectrum.append(sum_spec)
        err_spectrum.append(np.sqrt(sum_err))
        wavelength.append(w_med)
        sum_spec = 0.0
        sum_err = 0.0
    if(plot_flag==0):
        fig, ax = plt.subplots()
        plt.title('Rough Stellar Spectrum - RSS')
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Flux')
        ax.plot(wavelength, spectrum,'.',color = 'green', markersize=1 )
        ax.tick_params(color='blue', axis='x', labelsize=10)
        fig.set_figheight(5)
        fig.set_figwidth(10)
    
        plt.show()
        
    #interpolation:
    interp_spectrum = lambda w : np.interp(w, wavelength, spectrum)
    #interp_spectrum /= np.max(interp_spectrum)
    err_interp_spectrum = lambda w : np.interp(w, wavelength, err_spectrum)
    return interp_spectrum, err_interp_spectrum

"""
make_image is the function which takes data and the label of flux column and
creates a matrix full of NaN with the shape of the CCD (x-y hitted pixels). 
Then it replaces every element of the matrix(=img) with the flux.
The function returns the original image of the CCD 
"""
def make_image(data, col='f'):
    """Make image of data table column col
    """
    xmin, xmax = np.min(data['x']), np.max(data['x'])
    ymin, ymax = np.min(data['y']), np.max(data['y'])
    lenx = xmax - xmin + 1
    leny = ymax - ymin + 1
    
    # Make image full of NaN
    img = np.empty((leny, lenx))#matrix with shape leny*lenx
    img.fill(np.NaN)#matrix of NaN

    # Replace NaNs in image with values from table
    #matrix of flux
    img[data['y'] - ymin, data['x'] - xmin] = data[col]
    return img

def clean_image(org_img, niter=2):
    """Clean image from NaN by replace NaN by 
    median values (along axis=1)
    """
    img = org_img.copy()
    
    #create a image of median value
    for n in range(niter):
        med_img = moving_median(img)

        # Replace NaN values with median
        img[np.isnan(org_img)] = med_img[np.isnan(org_img)]

    return img
""""
moving_median is the function which takes in input the data and the number
of iterations (=niter=2) and the factor of clipping (default=100). It returns
the clean data.
"""
def moving_median(img):
    """Compute moving median filter along axis=1
    (+/-1 pixel)
    """
    # Make cube with 3 image planes
    # 3 is the shape
    cube = np.empty((3, img.shape[0], img.shape[1]))
    #original image
    cube[0] = img
    #roll shifts the element of the image of one position to the left along the column
    cube[1] = np.roll(img, -1, axis=1)
    #roll shifts the element of the image of one position to the right along the column
    cube[2] = np.roll(img, 1, axis=1)#roll shifts the element of the image of one position to the right 

    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # Ignore all-nan warnings

        # Take median over shifted images ignoring NaN values
        return np.nanmedian(cube, axis=0)

""""
clean_outliers is the function which takes in input the data and the number
of iterations (=niter=2) and the factor of clipping (default=100). It returns
the clean data.
"""
def clean_outliers(data, niter=2, clip=100):
    """Identify outliers in data table and replace
    them with median of surrounding pixels.
    """
    # Identify outliers and replace them with NaN
    img = make_image(data)#image of flux
    err = make_image(data, col='e')#image of errors
    #image of elements which are the median between original flux and the flux of column before and after 
    filt = moving_median(img)
    
    #Criteria to select outliers 
    ind = np.abs(img-filt) > clip*err
    #where ind is true replace with NaN
    img[ind] = np.NaN

    # Interpolate over identified outliers in image
    cimg = clean_image(img, niter=niter)
    
    #the clean data obtained are put inside the original data
    # Replace flux data in table with clean version from clean image
    cdata = data.copy()
    xmin, ymin = np.min(data['x']), np.min(data['y'])
    cdata['f'] = cimg[data['y']-ymin, data['x']-xmin]
    
    return cdata
      

""""
fit_psf is the function which takes in input the data and the spectrum
function (=spec_fun) and returns the spline funcion of the psf (=psf_spl), 
the normalized psf (=spec) and the normalized psf errors.
"""
def fit_psf(udata,spec_fun,r_star): 
    ind = np.argsort(udata['r'])
    data = udata[ind] 
    #if(r_star !=0):
    #sel =   np.abs(rdata['r'] - r_star) < 5.0
    #data = rdata[sel]
    psf = data['f']/spec_fun(data['w'])
    err_psf = data['e']/spec_fun(data['w'])
    weight = (1.0/err_psf)**2
    knots=np.arange(data['r'][1],data['r'][-1],0.5)
    psf_spl = sp.interpolate.LSQUnivariateSpline(data['r'], psf, knots, weight)
    return psf_spl, psf, err_psf


def fit_psf_rough(udata,spec_fun,knots=np.arange(min_slit,max_slit,1.0)): 
    ind = np.argsort(udata['r'])
    data = udata[ind] 
    psf = data['f']/spec_fun(data['w'])
    err_psf = data['e']/spec_fun(data['w'])
    weight = (1.0/err_psf)**2
    psf_spl = sp.interpolate.LSQUnivariateSpline(data['r'], psf, knots, weight)
    return psf_spl, psf, err_psf


""""
fit_spec is the function which takes in input the data and the point spread 
function (psf) and returns the spline funcion of the spectrum (=spec_spl), 
the normalized flux (=spec) and the normalized flux errors.
"""
def fit_spec(udata,psf_fun, r_star,dr=float('inf')): 
    ind = np.argsort(udata['w'])
    wdata = udata[ind] 
    ind_dr = np.abs(wdata['r']-r_star)< dr
    data = wdata[ind_dr]
    spec = data['f']/psf_fun(data['r'])
    err_spec = data['e']/psf_fun(data['r'])
    weight = (psf_fun(data['r'])/err_spec)**2
    knots = np.arange(data['w'][10], data['w'][-10], spec_resolution)
    wavelength = data['w']
    spec_spl = sp.interpolate.LSQUnivariateSpline(data['w'], spec, knots, weight)
    return spec_spl, spec, err_spec, weight, wavelength

def fit_spec_test(udata,psf_fun): 
    ind = np.argsort(udata['w'])
    data = udata[ind] 
    spec = data['f']/psf_fun(data['r'])
    err_spec = data['e']/psf_fun(data['r'])
    weight = (psf_fun(data['r'])/err_spec)**2
    knots = np.arange(data['w'][10], data['w'][-10], spec_resolution)
    wavelength = data['w']
    spec_spl = sp.interpolate.LSQUnivariateSpline(data['w'], spec, knots, weight)
    return spec_spl, spec, err_spec

def fit_spec_clump(wavelength, spectrum, err_spectrum): 
    s_wavelength = np.sort(wavelength)
    
    weight = (1/err_spectrum)**2
    knots = np.arange(s_wavelength[10], s_wavelength[-10], spec_resolution)
    spec_clump_spl = sp.interpolate.LSQUnivariateSpline(s_wavelength, spectrum,
                                                        knots, weight)
    return spec_clump_spl


def spectrum(data, psf_spl, r, dr):
    #sort by index data in term of wavelength
    ind = np.argsort(data['w'])
    wdata = data[ind]
    #selection of data at a r-distance from the center of the slit
    sel_r = np.abs(wdata['r'] - r) < dr
    r_data = wdata[sel_r]
    #n_ele = int(np.sum(sel_r)/(dr*4))#pixel resolution dr*8--> 2pixels
    spec = r_data['f']/psf_spl(r_data['r'])
    err_spec = r_data['e']/psf_spl(r_data['r'])
    weight = (psf_spl(r_data['r'])/err_spec)**2
    knots = np.arange(r_data['w'][10], r_data['w'][-10], spec_resolution)
    
    spec_spl =sp.interpolate.LSQUnivariateSpline(r_data['w'], spec, knots, weight)
    
    return spec_spl
        
        
        
""""
sigma_clipping is the function which applies the SIGMA CLIPPING METHOD to 
residuals which are the difference between the outlier (or noise) data and 
the original data. It returns the index of the values which exceed the criteria
of the method (default = 10sigma)
"""
def sigma_clipping(residual, niter=100, klip=10):
    sel = np.ones(len(residual), dtype='?')
    for n in range(niter):
        sigma = np.std(residual[sel])
        sel = np.absolute(residual) < klip*sigma
    return sel


def dw_selection(data, center, delta, type_data='w'):
    """Select rows inside a symmetric window around `center`.

    Parameters
    ----------
    data : structured ndarray
        Input table with spectroscopy columns.
    center : float
        Center value for the selection window.
    delta : float
        Half-width of the window.
    type_data : str, optional
        Selection axis: 'w' (wavelength) or 'r' (slit offset).
    """
    if type_data not in ('w', 'r'):
        raise ValueError("type_data must be 'w' or 'r'")

    sel = np.abs(data[type_data] - center) < delta
    return data[sel]

""""
add_noise is the function which adds NOISE to the original set of data, giving 
in input the original set of data and the factor of noise (default value = 10)
"""
def add_noise(indata, noise_factor=10):
    data = indata.copy()
    datalen = len(data)    
    data['f'] += noise_factor*data['e']*np.random.randn(datalen)
    data['e'] *= (1+noise_factor**2)**0.5
    return data

""""
add_outlier is the function which adds OUTLIER to the original set of data, 
giving in input the original set of data and the factor of outlier (default 
value = 5)
"""
def add_outlier(indata, p=1e-4, amplitude=5):
    data = indata.copy()
    datalen = len(data)
    max_amplitude = amplitude*np.max(data['f'])
    outliers = max_amplitude*np.random.rand(datalen)*(np.random.rand(datalen)<p)
    data['f'] +=outliers
    return data

""""
clump_spec_simu is the function which generates a random speed that is the speed 
of the clump, esteems the Doppler Shift and returns these two value
"""
def clump_spec_simu(clump_data, udata):
    w_med_star = np.median(udata['w'])
    sign_speed = choice([True, False])
    rand_speed = uniform(1, 100) # Outputs a random float between 1 and 100.
    if(sign_speed == False):
        rand_speed *= -1.0
    
    doppler = 1 + rand_speed/299792.0
    clump_data['w'] *= doppler
        
    shift_RV =(clump_data['w'] - udata['w']) * sp.constants.c/w_med_star
    mean_RV = np.mean(shift_RV)*0.001
    
    return mean_RV, doppler

"""
doppler_shift function computes a shifted wavelength array from a given 
velocity which define the shift
"""
def doppler_shift(wavelength, velocity):
    #array of wavelength of the template source
    doppler = 1+(velocity/(sp.constants.c*0.001))
    wavelength_shift = wavelength*doppler
        
    return wavelength_shift
   
"""    
cross correlation function returns the CCF from the shifted spectrum and
the template spectrum (i.e. shifted spectrum of a clump) 
"""   
def cross_correlation(shift_spec, tamplate_spec):
        
    ccf = np.sum(shift_spec*tamplate_spec)
    return ccf 
     
"""
cross_correletion_velocity function takes in input two arrays of wavelength 
(the first one is the template array and the second one is the shifted array),
the spline function of the template source (i.e. star) and an array of velocity.
It returns an array of ccf values.
"""   
def cross_correlation_velocity(wavelength, wavelength_shift, template_spec,
                               vel_array):
    ccf_array = np.zeros(len(vel_array))
    for i in range(0,len(vel_array)): #len(vel_array)        
        #w_shift is the array of the original wavelength shifted by vel_array[i]
        w_shift= doppler_shift(wavelength, vel_array[i])
        #cff_array is the array of the CROSS CORRELATION FUNCTION value
        ccf_array[i] = cross_correlation(template_spec(wavelength_shift),
                                         template_spec(w_shift))
    
    mean_ccf = np.mean(ccf_array)
    
    return ccf_array/mean_ccf

def cross_correlation_velocity_clump(wavelength, spec_clump, template_spec,
                               vel_array):
    ccf_array = np.zeros(len(vel_array))
    for i in range(0,len(vel_array)):        
        #w_shift is the array of the original wavelength shifted by vel_array[i]
        w_shift= doppler_shift(wavelength, vel_array[i])
        #cff_array is the array of the CROSS CORRELATION FUNCTION value
        median_t_spec = np.median(template_spec(w_shift))
        ccf_array[i] = cross_correlation(spec_clump, template_spec(w_shift))
            
    return ccf_array

def cross_correlation_velocity_clump_cut(w_with_cut_BC, w_with_cut, spec_clump, template_spec,
                               vel_array):
    ccf_array = np.zeros(len(vel_array))
    for i in range(0,len(vel_array)):        
        #w_shift is the array of the original wavelength shifted by vel_array[i]
        w_shift= doppler_shift(w_with_cut_BC, vel_array[i])
        #cff_array is the array of the CROSS CORRELATION FUNCTION value
        ccf_array[i] = cross_correlation(spec_clump(w_with_cut), template_spec(w_shift))
            
    
    return ccf_array

def clump_spectrum(res_data, psf_spl, r, dr):
    ind = np.argsort(res_data['w'])
    wdata = res_data[ind] 
    sel_dr = np.abs(wdata['r']-r)< dr
    data = wdata[sel_dr]
    spec = data['f']
    weight = (1.0/data['e'])**2
    knots = np.arange(data['w'][10], data['w'][-10], spec_resolution)
    spec_spl = sp.interpolate.LSQUnivariateSpline(data['w'], spec, knots, weight)
    return spec_spl
    


"""
image_separation_velocity function takes in input data and the spline fuction
of data and computes them in order to obtain the image of residual data define 
by velocity and separtion.
"""   
def  image_separation_velocity(data, master_wavelength, spec_spl, psf_spl, master_spec_spl, w_telluric,delta_w, n_group):
    null_w = np.zeros(3000)
    #Range of Velocity in Km/s
    dw = vmax/(sp.constants.c*0.001)
    lenv = int(vmax- vmin +1)
    vel_array=np.linspace(vmin,vmax,lenv)
    rmax , rmin = np.max(data['r']), np.min(data['r'])
    r_position = np.arange(rmin, rmax, 1)
    lenr = int(rmax - rmin +1)
    dr = 2.0
    img_ccf_ERF= np.empty((lenr, lenv))
    img_ccf_SRF= np.empty((lenr, lenv))
    
    print('No shift: ',data['w'])
    shift_data = data.copy()
    barycorr= Barycentric_Correction(aumic, eso, n_group)
    shift_data['w'] = doppler_shift(data['w'], -barycorr.value )
    print('si shift: ',shift_data['w'])
    
    for i in range(0,len(r_position)):
        #definition of the clump spectrum in ERF
        spec_clump_spl = clump_spectrum(data, psf_spl, r_position[i], dr)
        """if(i==10):
            fig, ax = plt.subplots()
            ax.plot(data['w'],spec_clump_spl(data['w']), '.', c = 'blue', ms = 0.5 )
        """    
        #definistion of the telluric lines mask in the SRF
        w = np.linspace(np.min(shift_data['w'])*(1.0+dw), np.max(shift_data['w'])*(1.0-dw), 3000)
        w_with_cut = w.copy()
        for j in range(len(w_telluric)):
            sel_w = np.abs(w_with_cut-w_telluric[j])>delta_w
            w_with_cut  = w_with_cut[sel_w]
        w_master_with_cut = w_with_cut.copy() 
        
        #if(i==10):plot_func.plot_spectrum_clump(w, w_with_cut, spec_clump_spl, master_spec_spl)
            
        img_ccf_SRF[i,:] = cross_correlation_velocity_clump_cut(w_master_with_cut, w_with_cut,
                       spec_clump_spl, master_spec_spl, vel_array)
        #if(i==10):plot_func.plot_ccf(vel_array, img_ccf_SRF[i,:]/(np.median(img_ccf_SRF[i,:])))   
        
        
        #shift the mask to ERF || BACK BC
        barycorr= Barycentric_Correction(aumic, eso, n_group)
        w_with_cut = doppler_shift(w_with_cut, barycorr.value )
        w = doppler_shift(w, barycorr.value )
        w_master_with_cut = w_with_cut.copy()
        
        #if(i==10):plot_func.plot_spectrum_clump(w, w_with_cut, spec_clump_spl, master_spec_spl)
        
        img_ccf_ERF[i,:] = cross_correlation_velocity_clump_cut(w_master_with_cut, w_with_cut,
                       spec_clump_spl, master_spec_spl, vel_array)
        #if(i==10):plot_func.plot_ccf(vel_array, img_ccf_ERF[i,:]/(np.median(img_ccf_ERF[i,:])))
        
       
        
        
    ind_max = np.argmax(psf_spl(data['r']))
    r_star = data['r'][ind_max] #position of the star 
    print("Position of star: ",r_star)
     
    r_position -=r_star
    #Normalization of the image
    title = "Radial Velocity - Separation Image || Group: " + str(n_group)
    img_ccf_ERF -= np.min(img_ccf_ERF[:,10:-10], axis = 1)[:,None]
    img_ccf_ERF /= np.max(img_ccf_ERF[:,10:-10], axis = 1)[:,None]
    plot_func.make_image_v_r(img_ccf_ERF, r_position, vel_array,title)
    
    """img_ccf_SRF -= np.min(img_ccf_SRF[:,10:-10], axis = 1)[:,None]
    img_ccf_SRF /= np.max(img_ccf_SRF[:,10:-10], axis = 1)[:,None]
    plot_func.make_image_v_r(img_ccf_SRF, r_position, vel_array)"""

    return img_ccf_ERF

    
def spectrum_test(udata, r, dr):
    #sort by index data in term of wavelength
    ind = np.argsort(udata['w'])
    data = udata[ind] 
    #selection of data at a r-distance from the center of the slit
    sel_r = np.abs(data['r'] - r) < dr
    r_data = data[sel_r]
    n_ele = int(np.sum(sel_r)/(dr*4))#pixel resolution dr*8--> 2pixels
    weight = (1/r_data['e'])**2
    knots = np.arange(r_data['w'][10], r_data['w'][-10], 
                      (r_data['w'][-10]-r_data['w'][10])/n_ele)
    spec_spl =sp.interpolate.LSQUnivariateSpline(r_data['w'], r_data['f'], 
                                                 knots, weight)
    return spec_spl
    
def image_separation_velocity_test(data, spec_spl, psf_spl):
    #Range of Velocity in Km/s

    dw = vmax/(sp.constants.c*0.001)
    lenv = int(vmax- vmin +1)
    vel_array=np.linspace(vmin,vmax,lenv)
    rmax , rmin = np.max(data['r']), np.min(data['r'])
    r_position = np.arange(rmin, rmax, 1)
    lenr = int(rmax - rmin +1)
    
    dr = 2.0
    img_ccf= np.empty((lenr, lenv))
    
    for i in range(0,len(r_position)):
        spec_clump_spl = spectrum_test(data, r_position[i], dr)
        wavelength = np.linspace(np.min(data['w'])*(1.0+dw), np.max(data['w'])*(1.0-dw), 3000)
        plot_func.plot_spectrum_clump_test(wavelength, spec_clump_spl, spec_spl)
        
        img_ccf[i,:] = cross_correlation_velocity_clump(wavelength,
                       spec_clump_spl(wavelength), spec_spl, vel_array)
        plot_func.plot_ccf(vel_array, img_ccf[i,:]/(np.mean(img_ccf[i,:])))
        
    psf_max = np.max(psf_spl(data['r']))
    ind_max = psf_spl(data['r']) == psf_max
    r_star = data['r'][ind_max] #position of the star 
    #print("Position of star: ",r_star)
     
    r_position -=r_star
    #Normalization of the image
    img_ccf -= np.min(img_ccf, axis = 1)[:,None]
    img_ccf /= np.max(img_ccf, axis = 1)[:,None]
    plot_func.make_image_v_r(img_ccf, r_position, vel_array)
    
    
"""
FWHM estimation: This function esteems the value of the FWHM of the psf, taking
in input the rdata, data sorted by r and the psf_spl, the spline function of 
the psf
    
"""
def FWHM_estimation(rdata, psf_spl):
    half_max_psf = np.max(psf_spl(rdata['r']))*0.5
    fwhm_signs= np.sign(psf_spl(rdata['r']) - half_max_psf)
    psf_signs = psf_spl(rdata['r']) - half_max_psf
    zero_crossing = (fwhm_signs[0:-2] != fwhm_signs[1:-1])
    ind_max = psf_spl(rdata['r']).argmax()
    star_position = rdata['r'][ind_max]
    zero_crossing_i =  np.where(zero_crossing)[0]
    for i in range(1,len(psf_signs)-1):
        if(psf_signs[i]<0.0 and psf_signs[i+1]>0.0 ):
            ind_min=i
        if(psf_signs[i]>0.0 and psf_signs[i+1]<0.0 ):
            ind_max=i
    FWHM = np.abs(rdata['r'][zero_crossing_i[1]] - rdata['r'][zero_crossing_i[0]])
    FWHM_for = np.abs(rdata['r'][ind_max] + rdata['r'][ind_min])
    
    return FWHM, FWHM_for, star_position

"""
correction_structures: this function cuts off the wierd structures inside three 
files after the reduction.
"""
def correction_structures(file_name, udata):
    """Correction of weird structure inside the image:"""
    if(file_name == "A11"):
        for i in range(165,275):
            sel_col = udata['x'] == i
            udata['f'][sel_col] = 0.0    
        for i in range(300,370):
            sel_col = udata['x'] == i
            udata['f'][sel_col] = 0.0       
            
        img_res = make_image(udata)
        #plot_func.plot_image(img_res, res_data, title="Model of Final Residues")
    
    if(file_name == "B11"):
        for i in range(70,260):
            sel_col = udata['x'] == i
            udata['f'][sel_col] = 0.0    
        
        img_res = make_image(udata)
        #plot_func.plot_image(img_res, res_data, title="Model of Final Residues") 
    
    if(file_name == "A12"):
        for i in range(7,46):
            sel_col = udata['x'] == i
            udata['f'][sel_col] = 0.0    
        
        img_res = make_image(udata)
        #plot_func.plot_image(img_res, res_data, title="Model of Final Residues")
        
    return udata


"""
Barycentric_Correction: this function esteems the BC of aumic and CRIRES at
eso in the four observations in order to work with an unique reference frame
 
"""
def Barycentric_Correction(aumic, eso, i):
    
    if(i==0 or i==20 or i==21 or i==22 or i==23 or i==24 or i==25 or i==26 or 
       i==28 or i==29):
        ti = Time(['2023-08-26T04:16:04'], format='isot', scale='utc')
        te = Time(['2023-08-26T05:12:48'], format='isot', scale='utc')
        barycorr_i= aumic.radial_velocity_correction(obstime=ti, 
                                                     location=eso).to(u.km/u.s)  
        barycorr_e= aumic.radial_velocity_correction(obstime=te, 
                                                     location=eso).to(u.km/u.s)  
        barycorr = (barycorr_e+barycorr_i)/2.0
    if(i==1 or i==2 or i==3 or i==4 or i==5 or i==6 or i==7 or i==8 or i==9):
        ti = Time(['2023-08-15T04:49:48'], format='isot', scale='utc')
        te = Time(['2023-08-15T05:49:03'], format='isot', scale='utc')
        barycorr_i= aumic.radial_velocity_correction(obstime=ti, 
                                                     location=eso).to(u.km/u.s)  
        barycorr_e= aumic.radial_velocity_correction(obstime=te, 
                                                     location=eso).to(u.km/u.s)  
        barycorr = (barycorr_e+barycorr_i)/2.0
    if(i==10 or i==11 or i==12 or i==13 or i==14 or i==15 or i==16 or i==17 or
       i==18 or i==19 or i==27):
        ti = Time(['2023-08-23T04:37:29'], format='isot', scale='utc')
        te = Time(['2023-08-23T05:45:44'], format='isot', scale='utc')
        barycorr_i= aumic.radial_velocity_correction(obstime=ti,
                                                     location=eso).to(u.km/u.s)  
        barycorr_e= aumic.radial_velocity_correction(obstime=te, 
                                                     location=eso).to(u.km/u.s)  
        barycorr = (barycorr_e+barycorr_i)/2.0 
    if(i==30 or i==31 or i==32 or i==33 or i==34 or i==35 or i==36 or i==37 or
       i==38 or i==39):
        ti = Time(['2023-08-27T03:51:38'],format='isot', scale='utc')
        te = Time(['2023-08-27T04:47:50'], format='isot', scale='utc')
        barycorr_i= aumic.radial_velocity_correction(obstime=ti, 
                                                     location=eso).to(u.km/u.s)  
        barycorr_e= aumic.radial_velocity_correction(obstime=te, 
                                                     location=eso).to(u.km/u.s)  
        barycorr = (barycorr_e+barycorr_i)/2.0  
            
    return barycorr


"""
delta_w: is a function used to identify the peak of std of flux and define 
the range of wavelength to cut of in order to eliminate the telluric lines 
from stellar spectrum
    
"""
def delta_w(w, null_w, w_copy, cut_w, ind_max, dw=0.1):
    w_peak = w_copy[ind_max]
    cut_w = np.where(np.abs(cut_w-w_peak)>dw, cut_w ,null_w)
    sel_w = np.abs(w_copy-w_peak)>dw
      
    return sel_w, cut_w
"""
telluric_lines_cut: is a function used to identify the telluric lines and to 
redefine the new array without the telluric lines 
    
"""    
def telluric_lines_cut(w, std_flux, limit, dw ):
    w_telluric = []
    
    for i in range(0,len(w)):
        std_flux_max = np.max(std_flux)
        ind_max = std_flux.argmax()
        if(std_flux_max<limit):break
        w_telluric.append(w[ind_max])
        w_peak = w[ind_max]
        sel_w = np.abs(w-w_peak)>dw
        std_flux = std_flux[sel_w]
        w = w[sel_w]
         
    
    print("TL: ",w_telluric)
    return w, std_flux, w_telluric


def master_spectrum_fit(master_wavelength, master_spec, master_weight):
    
    master_wavelength = (np.array(master_wavelength)).flatten()
    master_spec = (np.array(master_spec)).flatten()
    master_weight = (np.array(master_weight)).flatten()
    #print(master_wavelength)
    #print(len(master_wavelength))
    ind = np.argsort(master_wavelength)
    master_spec = master_spec[ind]
    master_weight = master_weight[ind]
    master_wavelength = np.sort(master_wavelength)
    #print(master_wavelength)
    knots = np.arange(master_wavelength[30], master_wavelength[-30], 0.015)
    master_spec_spl = sp.interpolate.LSQUnivariateSpline(master_wavelength, master_spec, knots, master_weight)
    
    return master_wavelength, master_spec, master_spec_spl












