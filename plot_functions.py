#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 16:58:51 2024

@author: beatricecaccherano
#Set of functions to plot spectroscopy data
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

"""Plot of PSF"""
def plot_psf(udata, psf, psf_spl, spec_spl, title, flag = 0):
    
    fig, ax1 = plt.subplots()
    plt.title(title)
    plt.xlabel('r - distance from slit centre')
    plt.ylabel('Flux')
    r = np.linspace(np.min(udata['r']),np.max(udata['r']), len(psf))
    ax1.plot(udata['r'], udata['f']/spec_spl(udata['w']),'.', markersize=1 )
    
    ax1.tick_params(color='blue', axis='x', labelsize=10)
    ax1.plot(udata['r'], psf_spl(udata['r']),'.', color = 'orange', markersize= 0.5 )
    fig.set_figheight(5)
    fig.set_figwidth(10)
    if flag==1:
        plt.title('PSF after SIGMA CLIPPING METHOD')
        ax1.plot(r, psf_spl(r),'.', c='red', markersize=0.5 )
        fig.set_figheight(15)
        fig.set_figwidth(10)
        
    plt.show()

"""Plot of SPECTRUM"""
def plot_spectrum(data, spec, spec_spl, flag=1):
    ind_w = np.argsort(data['w'])
    wdata = data[ind_w]
    fig, ax = plt.subplots()
    plt.title('Stellar Spectrum')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Flux')
    w = np.linspace(wdata['w'][30],wdata['w'][-30], 3000)
    if(flag == 1):
        ax.plot(w, spec,'.',color = 'red', markersize=1 )
    ax.plot(w, spec_spl(w),'.',color = 'green',ms=1)
    ax.tick_params(color='blue', axis='x', labelsize=10)
    fig.set_figheight(5)
    fig.set_figwidth(10)

    plt.show()
    
"""Plot of Spectrum CLUMP"""  
def plot_spectrum_clump(wavelength, w_with_cut, spec_clump_spl, spec_spl):    
    mean_flux=np.mean(spec_spl(wavelength))
    fig, ax = plt.subplots()
    #plt.title('Spline Function of Stellar Spectrum')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Flux')
    print(spec_spl(wavelength))
    ax.plot(wavelength, spec_spl(wavelength),'.',color = 'green', ms=1)
    ax.plot(w_with_cut,mean_flux*spec_clump_spl(w_with_cut),'.',color = 'red',ms=1)
    
    ax.tick_params(color='blue', axis='x', labelsize=10)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    
    plt.show()  
    
    
def plot_spectrum_clump_test(wavelength, spec_clump_spl, spec_spl):    
    mean_flux=np.mean(spec_clump_spl(wavelength))
    fig, ax = plt.subplots()
    #plt.title('Spline Function of Stellar Spectrum')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Flux')
    
    ax.plot(wavelength, mean_flux*spec_spl(wavelength),'.', color = 'green', markersize=1 )
    ax.plot(wavelength, spec_clump_spl(wavelength),'.',color = 'red',ms=1)
    ax.tick_params(color='blue', axis='x', labelsize=10)
    fig.set_figheight(5)
    fig.set_figwidth(10)

    plt.show()   
    
"""Plot of CLUMP and STELLAR SPECTRUMs"""
def plot_spectrum_clump_new(wavelength, wavelength_shift , spec_spl):
    
    fig, ax = plt.subplots()
    plt.title('Clump and Stellar Spectrum -- NEW!!')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Flux')
        
    ax.plot(wavelength, spec_spl(wavelength),'.',color = 'green',ms=1)
    ax.plot(wavelength, spec_spl(wavelength_shift),'.',color = 'red',ms=1)
       
    fig.set_figheight(5)
    fig.set_figwidth(10)

    plt.show()
    
"""Plot of CLUMP and STELLAR SPECTRUMs for SESSION 4"""
def plot_spectrum_clump_random(clump_data, udata, C_spec_spl, spec_spl,mean_RV):
    fig, ax = plt.subplots()
    plt.title('Clump and Stellar Spectrum')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Flux')
    w = np.linspace(np.min(udata['w'][10]),np.max(udata['w'][-10]), 3000)
    C_w = np.linspace(np.min(clump_data['w'][10]),np.max(clump_data['w'][-10]), 3000)
    #w_clump = np.linspace(np.min(clump_wavelength[100]),3944, 3000)
    ax.plot(w, spec_spl(w),'.',color = 'orange',ms=1)
    if(mean_RV>0.0):
        ax.plot(C_w, C_spec_spl(C_w),'.',color = 'red',ms=1)
    else:
        ax.plot(C_w, C_spec_spl(C_w),'.',color = 'blue',ms=1)
    fig.set_figheight(5)
    fig.set_figwidth(10)

    plt.show()

"""Plot of x-coordinate and r-distance from the center of the slit"""
def plot_x_r(data):
    fig, ax = plt.subplots()
    plt.title('x - r')
    plt.xlabel('x_pixel')
    plt.ylabel('r')
    ax.plot(data['x'], data['r'],'.', markersize=1 )
    ax.tick_params(color='blue', axis='x', labelsize=10)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    
    plt.show()
    
"""Plot of x-coordinate and y-coordinate"""
def plot_x_y(data):
    fig, ax = plt.subplots()
    plt.title('x - y')
    plt.xlabel('x_pixel')
    plt.ylabel('y_pixel')
    ax.plot(data['x'], data['y'],'.', markersize=1 )
    ax.tick_params(color='blue', axis='x', labelsize=10)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    
    plt.show()

"""3D Plot of x-coordinate, y-coordinate and flux"""
def plot_3D_flux_x_r(data):
    z = data['f']
    x = data['x']
    y = data['r']
     
    # Creating figure
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d") 
    # Creating plot
    ax.scatter3D(x, y, z, marker='.',s=0.1 )
    plt.title("Flux - r - x:")
    fig.set_figheight(5)
    fig.set_figwidth(10)
    plt.show()


def image_from_data(data):
    """Given a data table with 'x', 'r', and 'f' entries,
    this function returns a 2D array where each row of the
    table corresponds to a pixel (x,r) in the image. Pixel
    coordinates without value are given NaN values
    """
    fig, ax1 = plt.subplots()
    plt.title('Image from DATA:')
    fig.set_figheight(15)
    fig.set_figwidth(10)
    # Round r and to integers
    r_int = np.array(np.round(data['r']), dtype=int)
    x_int = np.array(np.round(data['x']), dtype=int)

    # Make smallest index 0
    r_int -= np.min(r_int)
    x_int -= np.min(x_int)

    # Create image array full of NaN
    image_array = np.full([np.max(r_int)+1, np.max(x_int)+1], np.nan)
    
    # Fill image with values from data table
    image_array[r_int, x_int] = data['f']
    
    ax1.pcolormesh(image_array)
    plt.show()

"""Plot of the CCF"""
def plot_ccf(velocity, ccf):
    fig, ax = plt.subplots()
    plt.title('ccf - velocity ')
    plt.xlabel('velocity [km/s]')
    plt.ylabel('Power')
    ax.plot(velocity, ccf,'.', color= 'blue', markersize=10 )
    #ax.plot(velocity, ccf_spl(velocity),'.', markersize=1 )
    ax.tick_params(color='blue', axis='x', labelsize=10)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    
    plt.show()

"""Plot of the IMAGE"""
def plot_image(image, data, fmin=None, fmax=None, title='ccf - velocity'):
    if fmin is None:
        fmin = np.nanmin(image)
    if fmax is None:
        fmax = np.nanmax(image)
    xmin, xmax = np.min(data['x']), np.max(data['x'])
    ymin, ymax = np.min(data['y']), np.max(data['y'])
    lenx = xmax - xmin + 1
    leny = ymax - ymin + 1
    fig, ax = plt.subplots()
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    x_data = np.linspace(xmin,xmax,lenx)
    y_data = np.linspace(ymin,ymax,leny)
    plt.pcolormesh(x_data, y_data, image, vmin = fmin, vmax = fmax)
    
    plt.colorbar()
    #plt.pcolormesh(image)
    ax.tick_params(color='blue', axis='x', labelsize=10)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    
    plt.show()
    



def make_image_v_r(img_ccf, r_position, velocity, title):
    
    fig, ax = plt.subplots()
    plt.title(title)
    plt.xlabel('Radial Velocity [km/s]')
    plt.ylabel('Separation [pixel units]')
    
    ax.tick_params(color='blue', axis='x', labelsize=10)
        
    bar=ax.pcolormesh(velocity, r_position, img_ccf, vmin=0, vmax=1)
    fig.colorbar(bar,ax=ax)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    
    plt.show()
    
    
    
def make_image_v_r_epochs(img_ccf, r_position, velocity, V_MAX, V_MIN, title):
    velocity = np.linspace(np.min(velocity), np.max(velocity), len(velocity)+1)
    r_position = np.linspace(np.min(r_position), np.max(r_position), len(r_position)+1)
    r_position_arcsec = r_position*0.0373
    fig, ax = plt.subplots()
    plt.title(title)
    plt.xlabel('Radial Velocity [km/s]')
    plt.ylabel('Separation [arcsec]')
    
    ax.tick_params(color='blue', axis='x', labelsize=10)
        
    bar=ax.pcolormesh(velocity[10:-10], r_position_arcsec, img_ccf[:,10:-10], shading='flat', vmin=V_MIN, vmax=V_MAX)
    fig.colorbar(bar,ax=ax)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    
    plt.show()

"""Image of group of spectrums for each wavelength range"""
def plot_image_group_spec(group_spec, wdata):
    #wdata = data sorted by w
    w_sorted = np.sort(wdata['w'])
    w = np.linspace(w_sorted[0], w_sorted[-1], group_spec.shape[1])
    group = np.arange(group_spec.shape[0])
    fig, ax = plt.subplots()
    plt.title("Group of spectrums")
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Group')
    
    ax.tick_params(color='blue', axis='x', labelsize=10)
        
    bar=ax.pcolormesh(w, group, group_spec)
    fig.colorbar(bar,ax=ax)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    
    plt.show()
    
"""Plot of group of spectrums for each wavelength range"""
def plot_group_spec(spec, w, len_spectrum):
    # w should be a 1D wavelength array matching spec columns
    w = np.asarray(w)
    fig2, ax2 = plt.subplots()
    plt.title('Stellar Spectrum1')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Flux')
    for i in range(0,len_spectrum):
        ax2.plot(w, spec[i,:],'-',ms=1)
        ax2.tick_params(color='blue', axis='x', labelsize=10)
    fig2.set_figheight(5)
    fig2.set_figwidth(10)
    plt.show()

"""plot of group of psf for each wavelength range"""
def plot_group_psf(psf, rdata, len_spectrum):
    colors = ['red', 'green', 'green', 'green', 'green', 'green', 'green',
              'green', 'green', 'green', 'lime', 'lime', 'lime', 'lime', 
              'lime', 'lime', 'lime', 'lime', 'lime', 'lime', 'red',
              'red', 'red', 'red', 'red', 'red', 'red', 'lime', 'red', 'red',
              'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue',
              'blue', 'blue']
    r_sorted = np.sort(rdata['r'])
    r = np.linspace(r_sorted[0], r_sorted[-1], psf.shape[1])
    fig2, ax2 = plt.subplots()
    plt.title('PSF')
    plt.xlabel('r [pixel units]')
    plt.ylabel('Flux')
    for i in range(0,len_spectrum):
        ax2.plot(r, psf[i,:],'-', c= colors[i],ms=1)
        ax2.tick_params(color='blue', axis='x', labelsize=10)
    fig2.set_figheight(5)
    fig2.set_figwidth(10)
    plt.show()
    
"""Plot of velocity shift across the four epoch"""
def plot_velocity_shift(velocity): 
    colors = ['red', 'green', 'green', 'green', 'green', 'green', 'green',
              'green', 'green', 'green', 'lime', 'lime', 'lime', 'lime', 
              'lime', 'lime', 'lime', 'lime', 'lime', 'lime', 'red',
              'red', 'red', 'red', 'red', 'red', 'red', 'lime', 'red', 'red',
              'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue', 'blue',
              'blue', 'blue']

    group = np.linspace(0,39,40)
    fig1, ax1 = plt.subplots()
    plt.title('VELOCITY SHIFT plot')
    plt.xlabel('velocity shift')
    plt.ylabel('group')
    ax1.scatter(velocity, group, c= colors)   
    ax1.tick_params(color='blue', axis='x', labelsize=10)
    fig1.set_figheight(5)
    fig1.set_figwidth(10)
    plt.show()
    
    
"""Plot of the std of flux for a fixed wavelength along the 40 spectrums of 
the same wavelength range"""  
def plot_std(wdata, std_flux, mean_std_flux):
    #wdata = data sorted by w
    w = np.linspace(wdata['w'][2000],wdata['w'][-2000], 3000)
    mean_std_flux = np.mean(std_flux)
    fig1, ax1 = plt.subplots()
    plt.title('standard deviation plot')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('standard deviation')
    ax1.plot(w, std_flux ,'-',ms=1)
    
    ax1.tick_params(color='blue', axis='x', labelsize=10)
    fig1.set_figheight(5)
    fig1.set_figwidth(10)
    plt.show()
    

def plot_spectrum_after_cut(spec, w_group, len_spectrum):
    fig2, ax2 = plt.subplots()
    plt.title('Stellar Spectrum after telluric lines removal')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Flux')
    for i in range(0,len_spectrum):
        spec[i]/= np.median(spec[i])#normalization
        ax2.plot(w_group[i],spec[i] ,'.',ms=1)
        ax2.tick_params(color='blue', axis='x', labelsize=10)
    fig2.set_figheight(5)
    fig2.set_figwidth(10)
    plt.show()    
    
def plot_master_spectrum(master_wavelength, master_spec, master_spec_spl):
    fig, ax = plt.subplots()
    plt.title('Master Stellar Spectrum')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Flux')
    #ax.plot(master_wavelength, master_spec,'.',ms=1)
    w = np.linspace(master_wavelength[2800],master_wavelength[-3500], 3000)
    ax.plot(w, master_spec_spl(w),'-',color = 'green', ms=1)
    ax.tick_params(color='blue', axis='x', labelsize=10)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    plt.show()    
        


def plot_std_no_tell(w_no_tell, w_telluric, std_flux, limit):
    plt.plot(w_no_tell,std_flux, '-', c='orange')
    plt.hlines(y = limit, xmin=np.min(w_no_tell),xmax=np.max(w_no_tell),  color = 'red')
    test = np.zeros(len(w_telluric))
    plt.plot(w_telluric,test, '*', c='pink')
    plt.show()


def plot_Keplerian_Limit(d_clumps_AU, velocity, vel_limit_1, vel_1, vel_limit_2, vel_2, vel_limit_3, vel_3):
    fig, ax = plt.subplots()
    plt.title('Maximum Limit Of Keplerian Velocity')
    plt.ylabel('Velocity [km/s] ')
    plt.xlabel('Position [AU] ')
    ax.plot(d_clumps_AU,velocity,'-',color = 'gray', ms=1)   
    ax.hlines(y = vel_limit_1, xmin=np.min(d_clumps_AU),xmax=np.max(d_clumps_AU),  color = 'green')
    ax.plot(d_clumps_AU,vel_1,'*',color = 'green', ms=1)   
    ax.hlines(y = vel_limit_2, xmin=np.min(d_clumps_AU),xmax=np.max(d_clumps_AU),  color = 'red')
    ax.plot(d_clumps_AU,vel_2,'*',color = 'red', ms=1)  
    ax.hlines(y = vel_limit_3, xmin=np.min(d_clumps_AU),xmax=np.max(d_clumps_AU),  color = 'cyan')
    ax.plot(d_clumps_AU,vel_3,'*',color = 'cyan', ms=1)
    ax.tick_params(color='blue', axis='x', labelsize=10)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    plt.show() 
    
    
def plot_circular_ring(d_clumps_arcsec, kepl_vel_R, kepl_vel_B, title ):
    fig, ax = plt.subplots()
    plt.title(title)
    plt.xlabel('Velocity [km/s] ')
    plt.ylabel('Position [arcsec] ')
    ax.plot(0.0,0.0,'*',color = 'gold', ms=10)
    ax.plot(kepl_vel_R, -d_clumps_arcsec,'-',color = 'green', ms=1)
    ax.plot(kepl_vel_B,d_clumps_arcsec,'-',color = 'green', ms=1) 
    ax.tick_params(color='blue', axis='x', labelsize=10)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    plt.show()   

def plot_ring(dx, vel,  title):
    fig, ax = plt.subplots()
    plt.title(title)
    plt.xlabel('Velocity [km/s] ')
    plt.ylabel('Position [arcsec] ')
    ax.plot(0.0,0.0,'*',color = 'gold', ms=10)
    plt.plot(vel, dx, '-',color = 'green', ms=1)
    ax.tick_params(color='blue', axis='x', labelsize=10)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    plt.show() 

def plot_ring_multiple(dx, vel1, vel2, vel3,  title):
    fig, ax = plt.subplots()
    plt.title(title)
    plt.xlabel('Velocity [km/s] ')
    plt.ylabel('Position [arcsec] ')
    ax.plot(0.0,0.0,'*',color = 'gold', ms=10)
    plt.plot(vel1, dx, '-',color = 'blue', ms=1)
    plt.plot(vel2, dx, '-',color = 'green', ms=1)
    plt.plot(vel3, dx, '-',color = 'red', ms=1)
    ax.tick_params(color='blue', axis='x', labelsize=10)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    plt.show()


def plot_expanding_klep_ring(d_clumps_arcsec, exp_vel_R_cw, exp_vel_B_cw, title):
    fig, ax = plt.subplots()
    plt.title(title)
    plt.xlabel('Velocity [km/s] ')
    plt.ylabel('Position [arcsec] ')
    ax.plot(0.0,0.0,'*',color = 'gold', ms=10)
    ax.plot(exp_vel_R_cw, -d_clumps_arcsec,'-',color = 'green', ms=1)
    ax.plot(exp_vel_R_cw, d_clumps_arcsec,'-',color = 'lightgreen', ms=1) 
    
    ax.plot(exp_vel_B_cw, -d_clumps_arcsec,'-',color = 'lightgreen', ms=1)
    ax.plot(exp_vel_B_cw, d_clumps_arcsec,'-',color = 'green', ms=1) 
    
    ax.tick_params(color='blue', axis='x', labelsize=10)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    plt.show()
    

def plot_expanding_klep_doppler(d_clumps_arcsec, vel_cw_R, vel_cw_B,title):
    fig, ax = plt.subplots()
    plt.title(title)
    plt.xlabel('Velocity [km/s] ')
    plt.ylabel('Position [arcsec] ')
    ax.plot(0.0,0.0,'*',color = 'gold', ms=10)
    ax.plot(vel_cw_R, -d_clumps_arcsec, '-',color = 'red', ms=1)
    ax.plot(vel_cw_R, d_clumps_arcsec,'-',color = 'salmon', ms=1) 
    
    ax.plot(vel_cw_B, -d_clumps_arcsec,'-',color = 'dodgerblue', ms=1)
    ax.plot(vel_cw_B, d_clumps_arcsec,'-',color = 'blue', ms=1) 
    
    ax.tick_params(color='blue', axis='x', labelsize=10)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    plt.show()
    
def plot_dr_corr(w0_array, dr_array, wdata, dr_spl):
    fig, ax = plt.subplots()
    plt.title('dr correction for each interval dw ')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('dr [pixel unit]')
    ax.plot(w0_array, dr_array,'.',color = 'green', ms=3)
    ax.plot(wdata['w'], dr_spl(wdata['w']),'-',color = 'red', ms=1)    
    fig.set_figheight(5)
    fig.set_figwidth(10)
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    