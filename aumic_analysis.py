#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:28:30 2024

@author: beatricecaccherano
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
import os
import numpy as np
import scipy as sp 
import time
import matplotlib.pyplot as plt 

import spectroscopy_functions as spec_fun
import plot_functions as plot_func

st = time.time()
root = 'C:/Users/alice/Documents/Stage Suède/data'
#files = ['AUMic000_A11.npy','AUMic001_A11.npy','AUMic002_A11.npy','AUMic003_A11.npy']


"""
FLAG TO activate if you are working with files A11, A12,  B11 in order to 
cut the strange structure which compare insiede the image.
"""
file_name ="A13"

"""
Creation of the list of files whcih you want analyse: 
    Write the name of the first file of the group, the group number (=group_num)
    and the group (=group).
"""
file = 'AUMic000_A26.npy'
group = '00'
group_num = int(group)

files = [file]

SESSION = 1
 
if(SESSION==1): 
    for i in range(0,2):
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
    # Build a consistent wavelength grid that is inside the coverage of every file
    # Using larger margin (50 points) to avoid edge effects and outliers
    w_common_min = -np.inf
    w_common_max = np.inf
    for filename in files:
        temp = np.load(os.path.join(root, filename))
        indw_temp = np.argsort(temp['w'])
        w_sorted_temp = temp['w'][indw_temp]
        w_common_min = max(w_common_min, w_sorted_temp[50])
        w_common_max = min(w_common_max, w_sorted_temp[-50])
    w = np.linspace(w_common_min, w_common_max, 3000)
    vel_shift = np.empty(40)
    FWHM = np.empty(40)
    group_spec = np.empty([40,3000])
    one_frame_spec = np.empty([40,3000])
    std_flux = np.empty([2,3000])
    j = 0
    for file in files:
        file_path = os.path.join(root, file)
        udata = np.load(file_path)
    
        print("AU Mic code is running with '", file, " ':" )
        
        """Cleaning data:"""
        clean_data = spec_fun.clean_outliers(udata)
        
        """Rough Stellar Spectrum - RSS"""
        interp_spectrum, err_interp_spectrum = spec_fun.rough_spectrum(clean_data,1)
        
        """
        PSF define by RSS: this PSF is useful to have the first rough PSF function 
        to start the analysis.
        """
        psf_spl, psf, err_psf= spec_fun.fit_psf(clean_data, interp_spectrum, 0.0)
        #plot_func.plot_psf(clean_data, psf, psf_spl, interp_spectrum)
        
        
        psf_values = psf_spl(clean_data['r'])
        ind_max = np.argmax(psf_values)
        r_star = clean_data['r'][ind_max] #position of the star 
        
        ind = np.argsort(udata['r'])
        rdata = udata[ind]
        
        """Model of Data:"""
        model_data = rdata.copy()#copy of data
        model_data['f'] = (interp_spectrum(rdata['w'])*psf_spl(rdata['r']))# replace the flux with the MODEL
        img = spec_fun.make_image(model_data)#image of the model data
        #plot_func.plot_image(img, model_data,
        #                     fmin=np.min(model_data['f']),
        #                     fmax=np.max(model_data['f']),
        #                     title="Model of Data")
        
        """Model of Residues:"""
        res = (rdata['f']-(interp_spectrum(rdata['w'])*psf_spl(rdata['r'])))/rdata['e']
        
        res_data = rdata.copy() #copy of data
        res_data['f'] = res # replace the flux with the residuals
        img_res = spec_fun.make_image(res_data)#create an image of the residuals
        #plot_func.plot_image(img_res, res_data,
        #                     fmin=np.min(res_data['f']),
        #                     fmax=np.max(res_data['f']),
        #                     title="Model of Residues")
        
        
        """Stellar Spectrum:"""

        # Use the cleaned data and a narrow radius window around the stellar PSF center
        M = 3
        for i in range(1,M):
            spec_spl, spec, err_spec, weight, wavelength = spec_fun.fit_spec(clean_data, psf_spl, r_star, dr=5.0)
            knots=np.arange(rdata['r'][10],rdata['r'][-10],0.5)
            psf_spl, psf, err_psf =  spec_fun.fit_psf_rough(rdata, spec_spl, knots)
         
        """Studing of the DATA NOISY """  
        max_psf = np.max(psf_spl(rdata['r']))
        half_max_psf = max_psf*0.5
        fwhm_psf = psf_spl(rdata['r']) - half_max_psf
        
        for i in range(1,len(fwhm_psf)-1):
            if(fwhm_psf[i]<0.0 and fwhm_psf[i+1]>0.0 ):
                ind_min=i+1
            if(fwhm_psf[i]>0.0 and fwhm_psf[i+1]<0.0 ):
                ind_max=i
        
        FWHM[j] = rdata['r'][ind_max] - rdata['r'][ind_min]
        #print('FWHM[',j,']',FWHM[j]) #modif
        
    
        """Studing of OBSERVATION EPOCHS and TELLURIC LINES """    
        vel_array = np.linspace(-10,10,40)
        group_spec[j,:]=spec_spl(w)#array to produce image of all the spectrums
        if(j==0):
            reference_spec = spec_spl(w)
            
        ccf_array=spec_fun.cross_correlation_velocity_clump(w, reference_spec, 
                                                            spec_spl, vel_array)
        mean_ccf = np.mean(ccf_array)
        plot_func.plot_ccf(vel_array, ccf_array/mean_ccf)
        ind_max = ccf_array.argmax()  #un peu simple non ?
        vel_shift[j] = vel_array[ind_max]
        w_shift = np.empty([40,3000])
        """# De-shift wavelengths for evaluation, but clip to file's wavelength range to avoid extrapolation
        doppler_factor = 1 + (vel_shift[j] / (sp.constants.c * 0.001))
        w_deshift = w / doppler_factor
        # Get this file's wavelength range
        w_file_min = np.min(udata['w'])
        w_file_max = np.max(udata['w'])
        # Clip de-shifted wavelengths to stay within the file's range
        w_deshift_clipped = np.clip(w_deshift, w_file_min, w_file_max)
        one_frame_spec[j,:] = spec_spl(w_deshift_clipped)  # evaluate at clipped de-shifted wavelengths"""
        if(vel_shift[j]!=-10.0):
            w_shift[j,:] = spec_fun.doppler_shift(w, vel_shift[j])
            one_frame_spec[j,:]=spec_spl(w_shift[j,:])#array to produce the image of all the spectrum in an unique frame
        else:
            one_frame_spec[j,:]=spec_spl(w)  
        j+=1
        
        
        """Final PSF and Spectrum plots:""" 
        #plot_func.plot_psf(rdata, psf, psf_spl, spec_spl,'Final PSF')
        #plot_func.plot_spectrum(udata, spec, spec_spl, 0)
       
        
        """Final image of the residuals:"""
        final_res = (rdata['f']-(spec_spl(rdata['w'])*psf_spl(rdata['r'])))/rdata['e']
        res_data = rdata.copy()
        res_data['f'] = final_res 
        img_res = spec_fun.make_image(res_data)
        #plot_func.plot_image(img_res, res_data,
        #                     fmin=np.min(res_data['f']),
        #                     fmax=np.max(res_data['f']),
        #                     title="Model of Final Residues")
        
        
        """Correction of weird structure inside the image:"""
        if(file_name == "A11"):
            for i in range(165,275):
                sel_col = res_data['x'] == i
                res_data['f'][sel_col] = 0.0    
            for i in range(300,370):
                sel_col = res_data['x'] == i
                res_data['f'][sel_col] = 0.0       
                
            img_res = spec_fun.make_image(res_data)
            plot_func.plot_image(img_res, res_data, title="Model of Final Residues")
        
        if(file_name == "B11"):
            for i in range(70,260):
                sel_col = res_data['x'] == i
                res_data['f'][sel_col] = 0.0    
            
            img_res = spec_fun.make_image(res_data)
            plot_func.plot_image(img_res, res_data, title="Model of Final Residues") 
        
        if(file_name == "A12"):
            for i in range(7,46):
                sel_col = res_data['x'] == i
                res_data['f'][sel_col] = 0.0    
            
            img_res = spec_fun.make_image(res_data)
            plot_func.plot_image(img_res, res_data, title="Model of Final Residues")        
              
        """Definition of IMAGE of the clumps in term of Radial Velocity and Separtion"""
        #spec_fun.image_separation_velocity_test(res_data, spec_spl, psf_spl) #modif 
        
    indw = np.argsort(udata['w']) #modif
    wdata = udata[indw] #modif
    w = np.linspace(wdata['w'][10], wdata['w'][-10], 3000) #modif
    #w = np.linspace(np.min(wdata['w']),np.max(wdata['w']), 3000)
    
    std_flux[0,:] = w
    std_flux[1,:] = np.std(one_frame_spec, axis=0)
    fig, ax = plt.subplots()
    plt.title('standard deviation plot')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('standard deviation')
    # Plot std but skip the problematic edge points to avoid border outliers
    ax.plot(w[50:-50], np.std(one_frame_spec, axis=0)[50:-50],'-',ms=1)
    ax.tick_params(color='blue', axis='x', labelsize=10)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    plt.show()
    
    # Now normalize and clip for plotting
    one_frame_spec /= np.median(one_frame_spec, axis=1)[:, None]  # normalization
    one_frame_spec = np.clip(one_frame_spec, 0.5, 2)  # clip outliers
    
    """PLOT of all the spectrums of the group"""
    group_spec/=np.median(group_spec, axis = 1)[:,None]#normalization
    # Clip extreme outliers to improve plot visibility
    group_spec = np.clip(group_spec, 0.5, 2)
    #plot_func.plot_group_spec(group_spec, w, len_spectrum)
    fig, ax = plt.subplots()
    plt.title('Stellar Spectrum')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Flux')
    for i in range(0,len_spectrum):
        ax.plot(w, group_spec[i,:],'-',ms=1)
        ax.tick_params(color='blue', axis='x', labelsize=10)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    plt.show()
    w_s = np.linspace(wdata['w'][100], wdata['w'][-30], 3000)
    """PLOT of all the spectrums in an UNIQUE REFERENCE FRAME"""
    #plot_func.plot_group_spec(one_frame_spec, w, len_spectrum)
    fig, ax = plt.subplots()
    plt.title('Stellar Spectrum in the SAME Reference Frame')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Flux')
    # one_frame_spec already normalized and clipped earlier
    for i in range(0,len_spectrum):
        ax.plot(w, one_frame_spec[i,:],'-',ms=1)
        ax.tick_params(color='blue', axis='x', labelsize=10)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
"""
SESSION 2: this section of the code analyses AUMic data group B15 to identify
the three epochs of observation.
"""
if (SESSION==2):
    for i in range(0,39):
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
    line1_position = np.zeros(len_spectrum)
    line2_position = np.zeros(len_spectrum)
    line3_position = np.zeros(len_spectrum)
    line4_position = np.zeros(len_spectrum)
    line5_position = np.zeros(len_spectrum)
    line6_position = np.zeros(len_spectrum)
    line7_position = np.zeros(len_spectrum)
    line8_position = np.zeros(len_spectrum)
    line9_position = np.zeros(len_spectrum)
    k = 0
    for file in files:
        file_path = os.path.join(root, file)
        udata = np.load(file_path)
    
        print("AU Mic code is running with '", file, " ':" )
        
        """Cleaning data:"""
        clean_data = spec_fun.clean_outliers(udata)
        
        """Rough Stellar Spectrum - RSS"""
        interp_spectrum, err_interp_spectrum = spec_fun.rough_spectrum(clean_data,1)
        
        """
        PSF define by RSS: this PSF is useful to have the first rough PSF function 
        to start the analysis.
        """
        psf_spl, psf, err_psf= spec_fun.fit_psf(clean_data, interp_spectrum, 0.0)
        #plot_func.plot_psf(clean_data, psf, psf_spl, interp_spectrum)
        
        """
        psf_max = np.max(psf_spl(clean_data['r']))
        print("Max of PSF: ",psf_max)
        ind_max = psf_spl(clean_data['r']) == psf_max
        r_star = clean_data['r'][ind_max] #position of the star 
        print("Position of star: ",r_star)
        plot_func.plot_psf(clean_data, psf, psf_spl, interp_spectrum)
        print(udata['r'])
        udata['r'] -= r_star
        print(udata['r'])
        """
        
        ind = np.argsort(udata['r'])
        rdata = udata[ind] 
        
        """Model of Data:"""
        model_data = rdata.copy()#copy of data
        model_data['f'] = (interp_spectrum(rdata['w'])*psf_spl(rdata['r']))# replace the flux with the MODEL
        img = spec_fun.make_image(model_data)#image of the model data
        plot_func.plot_image(img, model_data, title="Model of Data")
        
        """Model of Residues:"""
        res = (rdata['f']-(interp_spectrum(rdata['w'])*psf_spl(rdata['r'])))/rdata['e']
        
        res_data = rdata.copy() #copy of data
        res_data['f'] = res # replace the flux with the residuals
        img_res = spec_fun.make_image(res_data)#create an image of the residuals
        plot_func.plot_image(img_res, res_data, title="Model of Residues")
        
        
        """Stellar Spectrum:"""
        residues=[]
        M = 3
        for i in range(1,M):
            spec_spl, spec, err_spec, weight, wavelength = spec_fun.fit_spec(udata, psf_spl, r_star)
            knots=np.arange(rdata['r'][10],rdata['r'][-10],0.5)
            psf_spl, psf, err_psf =  spec_fun.fit_psf_rough(rdata, spec_spl, knots)
         
            
        w = np.linspace(np.min(udata['w'][10]),np.max(udata['w'][-10]), 3000) 
        array_spectrum.append(spec_spl(w))
        """Final PSF and Spectrum plots:""" 
        #plot_func.plot_psf(rdata, psf, psf_spl, spec_spl)
        #plot_func.plot_spectrum(udata, spec, spec_spl, 0)
        
        """REGION 1: (1545.68->1545.78)nm"""
        w1 = np.arange(1545.68,1545.78,0.02)
        len_w1 = len(w1)
        for i in range(0, len_w1):
            #min_flux1 = np.min(spec_spl(w1))
            ind_min_flux1 = spec_spl(w1).argmin()
            line1_position[k] = w1[ind_min_flux1]
                
        """REGION 2: (1549.44->1549.56)nm"""
        w2 = np.arange(1549.44,1549.56,0.02)
        len_w2 = len(w2)
        for i in range(0, len_w2):
            #min_flux1 = np.min(spec_spl(w1))
            ind_min_flux2 = spec_spl(w2).argmin()
            line2_position[k] = w2[ind_min_flux2]
               
        """REGION 3: (1548.29->1548.36)nm"""
        w3 = np.arange(1548.29, 1548.36,0.01)
        len_w3 = len(w3)
        for i in range(0, len_w3):
            ind_min_flux3 = spec_spl(w3).argmin()
            line3_position[k] = w3[ind_min_flux3]
            
        """REGION 4: (1544.02,1544.14)nm"""
        w4 = np.arange(1544.02,1544.14,0.02)
        len_w4 = len(w4)
        for i in range(0, len_w4):
            ind_min_flux4 = spec_spl(w4).argmin()
            line4_position[k] = w4[ind_min_flux4]
               
        k+=1
        
        """Final image of the residuals:"""
        final_res = (rdata['f']-(spec_spl(rdata['w'])*psf_spl(rdata['r'])))/rdata['e']
        res_data = rdata.copy()
        res_data['f'] = final_res 
        img_res = spec_fun.make_image(res_data)
        plot_func.plot_image(img_res, res_data, title="Model of Final Residues")
        
        """Correction of weird structure inside the image:"""
        if(file_name == "A11"):
            for i in range(165,275):
                sel_col = res_data['x'] == i
                res_data['f'][sel_col] = 0.0    
            for i in range(300,370):
                sel_col = res_data['x'] == i
                res_data['f'][sel_col] = 0.0       
                
            img_res = spec_fun.make_image(res_data)
            plot_func.plot_image(img_res, res_data, title="Model of Final Residues")
        
        if(file_name == "B11"):
            for i in range(70,260):
                sel_col = res_data['x'] == i
                res_data['f'][sel_col] = 0.0    
            
            img_res = spec_fun.make_image(res_data)
            plot_func.plot_image(img_res, res_data, title="Model of Final Residues") 
        
        if(file_name == "A12"):
            for i in range(7,46):
                sel_col = res_data['x'] == i
                res_data['f'][sel_col] = 0.0    
            
            img_res = spec_fun.make_image(res_data)
            plot_func.plot_image(img_res, res_data, title="Model of Final Residues")        
              
        """Definition of IMAGE of the clumps in term of Radial Velocity and Separtion"""
        spec_fun.image_separation_velocity_test(res_data, spec_spl, psf_spl)
        
    
    """PLOT of all the spectrums of the group"""
    fig, ax = plt.subplots()
    plt.title('Stellar Spectrum')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Flux')
    w = np.linspace(np.min(udata['w'][10]),np.max(udata['w'][-10]), 3000)
    c = 'green'
    for i in range(0,len_spectrum):
        ax.plot(w, array_spectrum[i],'.',color = c ,ms=1)
        ax.tick_params(color='blue', axis='x', labelsize=10)
        if(i==0):c = 'greenyellow'
        if(i==1):c = 'limegreen'
        
    """ ax.axvline(x = 1735.6370, color = 'orange')
    ax.axvline(x = 1736.1086, color = 'red')
    ax.axvline(x = 1736.69148, color = 'pink')
    ax.axvline(x = 1736.85848, color = 'purple')
    ax.axvline(x = 1739.81795, color = 'blue')
    ax.axvline(x = 1741.2748, color = 'cyan')
    ax.axvline(x = 1741.64647, color = 'olive')
    ax.axvline(x = 1744.44231, color = 'brown')"""
    fig.set_figheight(5)
    fig.set_figwidth(10)
    
    plt.show()
    print("REGION 1: ",line1_position)
    print("REGION 2: ",line2_position)
    print("REGION 3: ",line3_position)
    print("REGION 4: ",line4_position)

"""
SESSION 3: this section of the code analyses AUMic data group A11 to identify
the trhee epochs of observation.
"""
    
if (SESSION==3):
    for i in range(0,39):
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
    line1_position = np.zeros(len_spectrum)
    line2_position = np.zeros(len_spectrum)
    line3_position = np.zeros(len_spectrum)
    line4_position = np.zeros(len_spectrum)
    line5_position = np.zeros(len_spectrum)
    line6_position = np.zeros(len_spectrum)
    line7_position = np.zeros(len_spectrum)
    line8_position = np.zeros(len_spectrum)
    line9_position = np.zeros(len_spectrum)
    k = 0
    for file in files:
        file_path = os.path.join(root, file)
        udata = np.load(file_path)
    
        print("AU Mic code is running with '", file, " ':" )
        
        """Cleaning data:"""
        clean_data = spec_fun.clean_outliers(udata)
        
        """Rough Stellar Spectrum - RSS"""
        interp_spectrum, err_interp_spectrum = spec_fun.rough_spectrum(clean_data,0)
        
        """
        PSF define by RSS: this PSF is useful to have the first rough PSF function 
        to start the analysis.
        """
        psf_spl, psf, err_psf= spec_fun.fit_psf(clean_data, interp_spectrum, 0.0)
        #plot_func.plot_psf(clean_data, psf, psf_spl, interp_spectrum)
        
        """
        psf_max = np.max(psf_spl(clean_data['r']))
        print("Max of PSF: ",psf_max)
        ind_max = psf_spl(clean_data['r']) == psf_max
        r_star = clean_data['r'][ind_max] #position of the star 
        print("Position of star: ",r_star)
        plot_func.plot_psf(clean_data, psf, psf_spl, interp_spectrum)
        print(udata['r'])
        udata['r'] -= r_star
        print(udata['r'])
        """
        
        ind = np.argsort(udata['r'])
        rdata = udata[ind] 
        
        """Model of Data:"""
        model_data = rdata.copy()#copy of data
        model_data['f'] = (interp_spectrum(rdata['w'])*psf_spl(rdata['r']))# replace the flux with the MODEL
        img = spec_fun.make_image(model_data)#image of the model data
        plot_func.plot_image(img, model_data, title="Model of Data")
        
        """Model of Residues:"""
        res = (rdata['f']-(interp_spectrum(rdata['w'])*psf_spl(rdata['r'])))/rdata['e']
        
        res_data = rdata.copy() #copy of data
        res_data['f'] = res # replace the flux with the residuals
        img_res = spec_fun.make_image(res_data)#create an image of the residuals
        plot_func.plot_image(img_res, res_data, title="Model of Residues")
        
        
        """Stellar Spectrum:"""
        residues=[]
        M = 3
        for i in range(1,M):
            spec_spl, spec, err_spec, weight, wavelength = spec_fun.fit_spec(udata, psf_spl, r_star)
            knots=np.arange(rdata['r'][10],rdata['r'][-10],0.5)
            psf_spl, psf, err_psf =  spec_fun.fit_psf_rough(rdata, spec_spl, knots)
         
            
        w = np.linspace(np.min(udata['w'][10]),np.max(udata['w'][-10]), 3000) 
        array_spectrum.append(spec_spl(w))
        """Final PSF and Spectrum plots:""" 
        #plot_func.plot_psf(rdata, psf, psf_spl, spec_spl)
        #plot_func.plot_spectrum(udata, spec, spec_spl, 0)
        
        """REGION 1: """
        w1 = np.arange(1635.6, 1635.775,0.02)
        len_w1 = len(w1)
        for i in range(0, len_w1):
            #min_flux1 = np.min(spec_spl(w1))
            ind_min_flux1 = spec_spl(w1).argmin()
            line1_position[k] = w1[ind_min_flux1]
                
        """REGION 2: """
        w2 = np.arange(1636.875,1637.05,0.02)
        len_w2 = len(w2)
        for i in range(0, len_w2):
            #min_flux1 = np.min(spec_spl(w1))
            ind_min_flux2 = spec_spl(w2).argmin()
            line2_position[k] = w2[ind_min_flux2]
               
        """REGION 3: """
        w3 = np.arange(1637.225, 1637.375,0.02)
        len_w3 = len(w3)
        for i in range(0, len_w3):
            ind_min_flux3 = spec_spl(w3).argmin()
            line3_position[k] = w3[ind_min_flux3]
            
        """REGION 4: """
        w4 = np.arange(1642.86,1642.98,0.02)
        len_w4 = len(w4)
        for i in range(0, len_w4):
            ind_min_flux4 = spec_spl(w4).argmin()
            line4_position[k] = w4[ind_min_flux4]
            
        """REGION 5: """
        w5 = np.arange(1642.30,1642.40,0.02)
        len_w5 = len(w5)
        for i in range(0, len_w4):
            ind_min_flux5 = spec_spl(w5).argmin()
            line5_position[k] = w5[ind_min_flux5]
            
        """REGION 6: """
        w6 = np.arange(1638.16,1638.30,0.02)
        len_w6 = len(w6)
        for i in range(0, len_w6):
            ind_min_flux6 = spec_spl(w6).argmin()
            line6_position[k] = w6[ind_min_flux6]
            
        """REGION 7: """
        w7 = np.arange(1641.70,1641.86,0.02)
        len_w7 = len(w7)
        for i in range(0, len_w7):
            ind_min_flux7 = spec_spl(w7).argmin()
            line7_position[k] = w7[ind_min_flux7]
            
        """REGION 8: """
        w8 = np.arange(1643.075,1643.250,0.02)
        len_w8 = len(w8)
        for i in range(0, len_w8):
            ind_min_flux8 = spec_spl(w8).argmin()
            line8_position[k] = w8[ind_min_flux8]
         
        """REGION 9: """
        w9 = np.arange(1634.15,1634.30,0.02)
        len_w9 = len(w9)
        for i in range(0, len_w9):
            ind_min_flux9 = spec_spl(w9).argmin()
            line9_position[k] = w9[ind_min_flux9]
            
        k+=1
        
        """Final image of the residuals:"""
        final_res = (rdata['f']-(spec_spl(rdata['w'])*psf_spl(rdata['r'])))/rdata['e']
        res_data = rdata.copy()
        res_data['f'] = final_res 
        img_res = spec_fun.make_image(res_data)
        plot_func.plot_image(img_res, res_data, title="Model of Final Residues")
        
        """Correction of weird structure inside the image:"""
        if(file_name == "A11"):
            for i in range(165,275):
                sel_col = res_data['x'] == i
                res_data['f'][sel_col] = 0.0    
            for i in range(300,370):
                sel_col = res_data['x'] == i
                res_data['f'][sel_col] = 0.0       
                
            img_res = spec_fun.make_image(res_data)
            plot_func.plot_image(img_res, res_data, title="Model of Final Residues")
        
        if(file_name == "B11"):
            for i in range(70,260):
                sel_col = res_data['x'] == i
                res_data['f'][sel_col] = 0.0    
            
            img_res = spec_fun.make_image(res_data)
            plot_func.plot_image(img_res, res_data, title="Model of Final Residues") 
        
        if(file_name == "A12"):
            for i in range(7,46):
                sel_col = res_data['x'] == i
                res_data['f'][sel_col] = 0.0    
            
            img_res = spec_fun.make_image(res_data)
            plot_func.plot_image(img_res, res_data, title="Model of Final Residues")        
              
        """Definition of IMAGE of the clumps in term of Radial Velocity and Separtion"""
        spec_fun.image_separation_velocity_test(res_data, spec_spl, psf_spl)    
    
    
    """PLOT of all the spectrums of the group"""
    fig, ax = plt.subplots()
    plt.title('Stellar Spectrum')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Flux')
    w = np.linspace(np.min(udata['w'][10]),np.max(udata['w'][-10]), 3000)
    c = 'green'
    for i in range(0,len_spectrum):
        ax.plot(w, array_spectrum[i],'.',color = c ,ms=1)
        ax.tick_params(color='blue', axis='x', labelsize=10)
        if(i==0):c = 'greenyellow'
        if(i==1):c = 'limegreen'
        
    """ax.axvline(x = 1751.547, color = 'orange')
    ax.axvline(x = 1755.155, color = 'red')
    ax.axvline(x = 1756.821, color = 'pink')
    ax.axvline(x = 1757.452, color = 'purple')"""
    fig.set_figheight(5)
    fig.set_figwidth(10)
    
    plt.show()
    print("REGION 1: ",line1_position, "average:", np.mean(line1_position))
    print("REGION 2: ",line2_position, "average:", np.mean(line2_position))
    print("REGION 3: ",line3_position, "average:", np.mean(line3_position))
    print("REGION 4: ",line4_position, "average:", np.mean(line4_position))
    print("REGION 5: ",line5_position, "average:", np.mean(line5_position))
    print("REGION 6: ",line6_position, "average:", np.mean(line6_position))
    print("REGION 7: ",line7_position, "average:", np.mean(line7_position))
    print("REGION 8: ",line8_position, "average:", np.mean(line8_position))
    print("REGION 9: ",line9_position, "average:", np.mean(line9_position))
    
    
"""
SESSION 4: this section of the code analyses AUMic data group A23 to identify
the trhee epochs of observation.
"""
if (SESSION==4):
    for i in range(0,2):
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
    line1_position = np.zeros(len_spectrum)
    line2_position = np.zeros(len_spectrum)
    line3_position = np.zeros(len_spectrum)
    line4_position = np.zeros(len_spectrum)
    line5_position = np.zeros(len_spectrum)
    k = 0
    for file in files:
        file_path = os.path.join(root, file)
        udata = np.load(file_path)
    
        print("AU Mic code is running with '", file, " ':" )
        
        """Cleaning data:"""
        clean_data = spec_fun.clean_outliers(udata)
        
        """Rough Stellar Spectrum - RSS"""
        interp_spectrum, err_interp_spectrum = spec_fun.rough_spectrum(clean_data,1)
        
        """
        PSF define by RSS: this PSF is useful to have the first rough PSF function 
        to start the analysis.
        """
        psf_spl, psf, err_psf= spec_fun.fit_psf(clean_data, interp_spectrum, 0.0)
        #plot_func.plot_psf(clean_data, psf, psf_spl, interp_spectrum)
        
        """
        psf_max = np.max(psf_spl(clean_data['r']))
        print("Max of PSF: ",psf_max)
        ind_max = psf_spl(clean_data['r']) == psf_max
        r_star = clean_data['r'][ind_max] #position of the star 
        print("Position of star: ",r_star)
        plot_func.plot_psf(clean_data, psf, psf_spl, interp_spectrum)
        print(udata['r'])
        udata['r'] -= r_star
        print(udata['r'])
        """
        
        ind = np.argsort(udata['r'])
        rdata = udata[ind] 
        
        """Model of Data:"""
        model_data = rdata.copy()#copy of data
        model_data['f'] = (interp_spectrum(rdata['w'])*psf_spl(rdata['r']))# replace the flux with the MODEL
        img = spec_fun.make_image(model_data)#image of the model data
        plot_func.plot_image(img, model_data, title="Model of Data")
        
        """Model of Residues:"""
        res = (rdata['f']-(interp_spectrum(rdata['w'])*psf_spl(rdata['r'])))/rdata['e']
        
        res_data = rdata.copy() #copy of data
        res_data['f'] = res # replace the flux with the residuals
        img_res = spec_fun.make_image(res_data)#create an image of the residuals
        plot_func.plot_image(img_res, res_data, title="Model of Residues")
        
        
        """Stellar Spectrum:"""
        residues=[]
        M = 3
        for i in range(1,M):
            spec_spl, spec, err_spec, weight, wavelength = spec_fun.fit_spec(udata, psf_spl, r_star)
            knots=np.arange(rdata['r'][10],rdata['r'][-10],0.5)
            psf_spl, psf, err_psf =  spec_fun.fit_psf_rough(rdata, spec_spl, knots)
         
            
        w = np.linspace(np.min(udata['w'][10]),np.max(udata['w'][-10]), 3000) 
        array_spectrum.append(spec_spl(w))
        """Final PSF and Spectrum plots:""" 
        #plot_func.plot_psf(rdata, psf, psf_spl, spec_spl)
        #plot_func.plot_spectrum(udata, spec, spec_spl, 0)
        
        """REGION 1: (1649.00->1649.25)nm"""
        w1 = np.arange(1649.00,1649.25,0.02)
        len_w1 = len(w1)
        for i in range(0, len_w1):
            #min_flux1 = np.min(spec_spl(w1))
            ind_min_flux1 = spec_spl(w1).argmin()
            line1_position[k] = w1[ind_min_flux1]
                
        """REGION 2: (1654.76->1654.90)nm"""
        w2 = np.arange(1654.76,1654.90,0.02)
        len_w2 = len(w2)
        for i in range(0, len_w2):
            #min_flux1 = np.min(spec_spl(w1))
            ind_min_flux2 = spec_spl(w2).argmin()
            line2_position[k] = w2[ind_min_flux2]
               
        """REGION 3: (1650.78->1650.88)nm"""
        w3 = np.arange(1650.78,1650.88,0.02)
        len_w3 = len(w3)
        for i in range(0, len_w3):
            ind_min_flux3 = spec_spl(w3).argmin()
            line3_position[k] = w3[ind_min_flux3]
            
        """REGION 4: (1650.22 ->1650.36)nm"""
        w4 = np.arange(1650.22, 1650.36,0.02)
        len_w4 = len(w4)
        for i in range(0, len_w4):
            ind_min_flux4 = spec_spl(w4).argmin()
            line4_position[k] = w4[ind_min_flux4]
            
        """REGION 5: (1750.46-->1750.58)nm"""
        w5 = np.arange(1653.025,1653.20,0.02)
        len_w5 = len(w5)
        for i in range(0, len_w4):
            ind_min_flux5 = spec_spl(w5).argmin()
            line5_position[k] = w5[ind_min_flux5]
               
        k+=1
        
        """Final image of the residuals:"""
        final_res = (rdata['f']-(spec_spl(rdata['w'])*psf_spl(rdata['r'])))/rdata['e']
        res_data = rdata.copy()
        res_data['f'] = final_res 
        img_res = spec_fun.make_image(res_data)
        plot_func.plot_image(img_res, res_data, title="Model of Final Residues")
        
        """Correction of weird structure inside the image:"""
        if(file_name == "A11"):
            for i in range(165,275):
                sel_col = res_data['x'] == i
                res_data['f'][sel_col] = 0.0    
            for i in range(300,370):
                sel_col = res_data['x'] == i
                res_data['f'][sel_col] = 0.0       
                
            img_res = spec_fun.make_image(res_data)
            plot_func.plot_image(img_res, res_data, title="Model of Final Residues")
        
        if(file_name == "B11"):
            for i in range(70,260):
                sel_col = res_data['x'] == i
                res_data['f'][sel_col] = 0.0    
            
            img_res = spec_fun.make_image(res_data)
            plot_func.plot_image(img_res, res_data, title="Model of Final Residues") 
        
        if(file_name == "A12"):
            for i in range(7,46):
                sel_col = res_data['x'] == i
                res_data['f'][sel_col] = 0.0    
            
            img_res = spec_fun.make_image(res_data)
            plot_func.plot_image(img_res, res_data, title="Model of Final Residues")        
              
        """Definition of IMAGE of the clumps in term of Radial Velocity and Separtion"""
        spec_fun.image_separation_velocity_test(res_data, spec_spl, psf_spl)
        
    
    """PLOT of all the spectrums of the group"""
    fig, ax = plt.subplots()
    plt.title('Stellar Spectrum')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Flux')
    w = np.linspace(np.min(udata['w'][10]),np.max(udata['w'][-10]), 3000)
    c = 'green'
    for i in range(0,len_spectrum):
        ax.plot(w, array_spectrum[i],'.',color = c ,ms=1)
        ax.tick_params(color='blue', axis='x', labelsize=10)
        if(i==0):c = 'greenyellow'
        if(i==1):c = 'limegreen'
        
    ax.axvline(x = 1648.223, color = 'orange')
    ax.axvline(x = 1650.957, color = 'red')
    ax.axvline(x = 1653.727, color = 'pink')
    ax.axvline(x = 1649.746, color = 'purple')
    ax.axvline(x = 1650.198, color = 'blue')
    """ax.axvline(x = 1741.2748, color = 'cyan')
    ax.axvline(x = 1741.64647, color = 'olive')
    ax.axvline(x = 1744.44231, color = 'brown')"""
    fig.set_figheight(5)
    fig.set_figwidth(10)
    
    plt.show()
    print("REGION 1: ",line1_position)
    print("REGION 2: ",line2_position)
    print("REGION 3: ",line3_position)
    print("REGION 4: ",line4_position)
    print("REGION 5: ",line5_position)

"""
SESSION 5: this section of the code analyses AUMic data group B34 to identify
the trhee epochs of observation.
"""
if (SESSION==5):
    for i in range(0,2):
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
    line1_position = np.zeros(len_spectrum)
    line2_position = np.zeros(len_spectrum)
    line3_position = np.zeros(len_spectrum)
    line4_position = np.zeros(len_spectrum)
    line5_position = np.zeros(len_spectrum)
    k = 0
    for file in files:
        file_path = os.path.join(root, file)
        udata = np.load(file_path)
    
        print("AU Mic code is running with '", file, " ':" )
        
        """Cleaning data:"""
        clean_data = spec_fun.clean_outliers(udata)
        
        """Rough Stellar Spectrum - RSS"""
        interp_spectrum, err_interp_spectrum = spec_fun.rough_spectrum(clean_data,1)
        
        """
        PSF define by RSS: this PSF is useful to have the first rough PSF function 
        to start the analysis.
        """
        psf_spl, psf, err_psf= spec_fun.fit_psf(clean_data, interp_spectrum, 0.0)
        #plot_func.plot_psf(clean_data, psf, psf_spl, interp_spectrum)
        
        """
        psf_max = np.max(psf_spl(clean_data['r']))
        print("Max of PSF: ",psf_max)
        ind_max = psf_spl(clean_data['r']) == psf_max
        r_star = clean_data['r'][ind_max] #position of the star 
        print("Position of star: ",r_star)
        plot_func.plot_psf(clean_data, psf, psf_spl, interp_spectrum)
        print(udata['r'])
        udata['r'] -= r_star
        print(udata['r'])
        """
        
        ind = np.argsort(udata['r'])
        rdata = udata[ind] 
        
        """Model of Data:"""
        model_data = rdata.copy()#copy of data
        model_data['f'] = (interp_spectrum(rdata['w'])*psf_spl(rdata['r']))# replace the flux with the MODEL
        img = spec_fun.make_image(model_data)#image of the model data
        plot_func.plot_image(img, model_data, title="Model of Data")
        
        """Model of Residues:"""
        res = (rdata['f']-(interp_spectrum(rdata['w'])*psf_spl(rdata['r'])))/rdata['e']
        
        res_data = rdata.copy() #copy of data
        res_data['f'] = res # replace the flux with the residuals
        img_res = spec_fun.make_image(res_data)#create an image of the residuals
        plot_func.plot_image(img_res, res_data, title="Model of Residues")
        
        
        """Stellar Spectrum:"""
        residues=[]
        M = 3
        for i in range(1,M):
            spec_spl, spec, err_spec, weight, wavelength = spec_fun.fit_spec(udata, psf_spl, r_star)
            knots=np.arange(rdata['r'][10],rdata['r'][-10],0.5)
            psf_spl, psf, err_psf =  spec_fun.fit_psf_rough(rdata, spec_spl, knots)
         
            
        w = np.linspace(np.min(udata['w'][10]),np.max(udata['w'][-10]), 3000) 
        array_spectrum.append(spec_spl(w))
        """Final PSF and Spectrum plots:""" 
        #plot_func.plot_psf(rdata, psf, psf_spl, spec_spl)
        #plot_func.plot_spectrum(udata, spec, spec_spl, 0)
        
        """REGION 1: (1567.00->1567.98)nm"""
        w1 = np.arange(1567.00,1567.98,0.02)
        len_w1 = len(w1)
        for i in range(0, len_w1):
            #min_flux1 = np.min(spec_spl(w1))
            ind_min_flux1 = spec_spl(w1).argmin()
            line1_position[k] = w1[ind_min_flux1]
                
        """REGION 2: (1654.76->1654.90)nm"""
        w2 = np.arange(1569.32, 1569.42,0.02)
        len_w2 = len(w2)
        for i in range(0, len_w2):
            #min_flux1 = np.min(spec_spl(w1))
            ind_min_flux2 = spec_spl(w2).argmin()
            line2_position[k] = w2[ind_min_flux2]
               
        """REGION 3: (1650.78->1650.88)nm"""
        w3 = np.arange(1568.38,1568.52,0.02)
        len_w3 = len(w3)
        for i in range(0, len_w3):
            ind_min_flux3 = spec_spl(w3).argmin()
            line3_position[k] = w3[ind_min_flux3]
            
        """REGION 4: (1650.22 ->1650.36)nm"""
        w4 = np.arange(1573.79, 1573.86,0.02)
        len_w4 = len(w4)
        for i in range(0, len_w4):
            ind_min_flux4 = spec_spl(w4).argmin()
            line4_position[k] = w4[ind_min_flux4]
            
        """REGION 5: (1750.46-->1750.58)nm"""
        w5 = np.arange(1572.77,1572.85,0.02)
        len_w5 = len(w5)
        for i in range(0, len_w4):
            ind_min_flux5 = spec_spl(w5).argmin()
            line5_position[k] = w5[ind_min_flux5]
               
        k+=1
        
        """Final image of the residuals:"""
        final_res = (rdata['f']-(spec_spl(rdata['w'])*psf_spl(rdata['r'])))/rdata['e']
        res_data = rdata.copy()
        res_data['f'] = final_res 
        img_res = spec_fun.make_image(res_data)
        plot_func.plot_image(img_res, res_data, title="Model of Final Residues")
        
        """Correction of weird structure inside the image:"""
        if(file_name == "A11"):
            for i in range(165,275):
                sel_col = res_data['x'] == i
                res_data['f'][sel_col] = 0.0    
            for i in range(300,370):
                sel_col = res_data['x'] == i
                res_data['f'][sel_col] = 0.0       
                
            img_res = spec_fun.make_image(res_data)
            plot_func.plot_image(img_res, res_data, title="Model of Final Residues")
        
        if(file_name == "B11"):
            for i in range(70,260):
                sel_col = res_data['x'] == i
                res_data['f'][sel_col] = 0.0    
            
            img_res = spec_fun.make_image(res_data)
            plot_func.plot_image(img_res, res_data, title="Model of Final Residues") 
        
        if(file_name == "A12"):
            for i in range(7,46):
                sel_col = res_data['x'] == i
                res_data['f'][sel_col] = 0.0    
            
            img_res = spec_fun.make_image(res_data)
            plot_func.plot_image(img_res, res_data, title="Model of Final Residues")        
              
        """Definition of IMAGE of the clumps in term of Radial Velocity and Separtion"""
        spec_fun.image_separation_velocity_test(res_data, spec_spl, psf_spl)
        
    
    """PLOT of all the spectrums of the group"""
    fig, ax = plt.subplots()
    plt.title('Stellar Spectrum')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Flux')
    w = np.linspace(np.min(udata['w'][10]),np.max(udata['w'][-10]), 3000)
    c = 'green'
    for i in range(0,len_spectrum):
        ax.plot(w, array_spectrum[i],'.',color = c ,ms=1)
        ax.tick_params(color='blue', axis='x', labelsize=10)
        if(i==0):c = 'greenyellow'
        if(i==1):c = 'limegreen'
        
    ax.axvline(x = 1572.022, color = 'orange')
    ax.axvline(x = 1571.712, color = 'red')
    ax.axvline(x = 1571.117, color = 'pink')
    """ax.axvline(x = 1649.746, color = 'purple')
    ax.axvline(x = 1650.198, color = 'blue')
    ax.axvline(x = 1741.2748, color = 'cyan')
    ax.axvline(x = 1741.64647, color = 'olive')
    ax.axvline(x = 1744.44231, color = 'brown')"""
    fig.set_figheight(5)
    fig.set_figwidth(10)
    
    plt.show()
    print("REGION 1: ",line1_position)
    print("REGION 2: ",line2_position)
    print("REGION 3: ",line3_position)
    print("REGION 4: ",line4_position)
    print("REGION 5: ",line5_position)

"""
SESSION 6: this section of the code analyses AUMic data group A33 to identify
the trhee epochs of observation.
"""
if (SESSION==6):
    for i in range(0,2):
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
    line1_position = np.zeros(len_spectrum)
    line2_position = np.zeros(len_spectrum)
    line3_position = np.zeros(len_spectrum)
    line4_position = np.zeros(len_spectrum)
    k = 0
    for file in files:
        file_path = os.path.join(root, file)
        udata = np.load(file_path)
    
        print("AU Mic code is running with '", file, " ':" )
        
        """Cleaning data:"""
        clean_data = spec_fun.clean_outliers(udata)
        
        """Rough Stellar Spectrum - RSS"""
        interp_spectrum, err_interp_spectrum = spec_fun.rough_spectrum(clean_data,0)
        
        """
        PSF define by RSS: this PSF is useful to have the first rough PSF function 
        to start the analysis.
        """
        psf_spl, psf, err_psf= spec_fun.fit_psf(clean_data, interp_spectrum, 0.0)
        plot_func.plot_psf(clean_data, psf, psf_spl, interp_spectrum)
        
        """
        psf_max = np.max(psf_spl(clean_data['r']))
        print("Max of PSF: ",psf_max)
        ind_max = psf_spl(clean_data['r']) == psf_max
        r_star = clean_data['r'][ind_max] #position of the star 
        print("Position of star: ",r_star)
        plot_func.plot_psf(clean_data, psf, psf_spl, interp_spectrum)
        print(udata['r'])
        udata['r'] -= r_star
        print(udata['r'])
        """
        
        ind = np.argsort(udata['r'])
        rdata = udata[ind] 
        
        """Model of Data:"""
        model_data = rdata.copy()#copy of data
        model_data['f'] = (interp_spectrum(rdata['w'])*psf_spl(rdata['r']))# replace the flux with the MODEL
        img = spec_fun.make_image(model_data)#image of the model data
        plot_func.plot_image(img, model_data, title="Model of Data")
        
        """Model of Residues:"""
        res = (rdata['f']-(interp_spectrum(rdata['w'])*psf_spl(rdata['r'])))/rdata['e']
        
        res_data = rdata.copy() #copy of data
        res_data['f'] = res # replace the flux with the residuals
        img_res = spec_fun.make_image(res_data)#create an image of the residuals
        plot_func.plot_image(img_res, res_data, title="Model of Residues")
        
        
        """Stellar Spectrum:"""
        residues=[]
        M = 3
        for i in range(1,M):
            spec_spl, spec, err_spec, weight, wavelength = spec_fun.fit_spec(udata, psf_spl, r_star)
            knots=np.arange(rdata['r'][10],rdata['r'][-10],0.5)
            psf_spl, psf, err_psf =  spec_fun.fit_psf_rough(rdata, spec_spl, knots)
         
            
        w = np.linspace(np.min(udata['w'][10]),np.max(udata['w'][-10]), 3000) 
        array_spectrum.append(spec_spl(w))
        """Final PSF and Spectrum plots:""" 
        plot_func.plot_psf(rdata, psf, psf_spl, spec_spl)
        #plot_func.plot_spectrum(udata, spec, spec_spl, 0)
        
        """REGION 1: (1617.850->1618.025)nm"""
        w1 = np.arange(1617.850,1618.025,0.02)
        len_w1 = len(w1)
        for i in range(0, len_w1):
            #min_flux1 = np.min(spec_spl(w1))
            ind_min_flux1 = spec_spl(w1).argmin()
            line1_position[k] = w1[ind_min_flux1]
                
        """REGION 2: (1614.800-> 1614.975)nm"""
        w2 = np.arange(1614.800, 1614.975,0.02)
        len_w2 = len(w2)
        for i in range(0, len_w2):
            #min_flux1 = np.min(spec_spl(w1))
            ind_min_flux2 = spec_spl(w2).argmin()
            line2_position[k] = w2[ind_min_flux2]
               
        """REGION 3: (1617.12->1617.26)nm"""
        w3 = np.arange(1617.12,1617.26,0.02)
        len_w3 = len(w3)
        for i in range(0, len_w3):
            ind_min_flux3 = spec_spl(w3).argmin()
            line3_position[k] = w3[ind_min_flux3]
            
        """REGION 4: (1650.22 ->1650.36)nm"""
        w4 = np.arange(1612.80, 1612.98,0.02)
        len_w4 = len(w4)
        for i in range(0, len_w4):
            ind_min_flux4 = spec_spl(w4).argmin()
            line4_position[k] = w4[ind_min_flux4]
            
        
               
        k+=1
        
        """Final image of the residuals:"""
        final_res = (rdata['f']-(spec_spl(rdata['w'])*psf_spl(rdata['r'])))/rdata['e']
        res_data = rdata.copy()
        res_data['f'] = final_res 
        img_res = spec_fun.make_image(res_data)
        plot_func.plot_image(img_res, res_data, title="Model of Final Residues")
        
        """Correction of weird structure inside the image:"""
        if(file_name == "A11"):
            for i in range(165,275):
                sel_col = res_data['x'] == i
                res_data['f'][sel_col] = 0.0    
            for i in range(300,370):
                sel_col = res_data['x'] == i
                res_data['f'][sel_col] = 0.0       
                
            img_res = spec_fun.make_image(res_data)
            plot_func.plot_image(img_res, res_data, title="Model of Final Residues")
        
        if(file_name == "B11"):
            for i in range(70,260):
                sel_col = res_data['x'] == i
                res_data['f'][sel_col] = 0.0    
            
            img_res = spec_fun.make_image(res_data)
            plot_func.plot_image(img_res, res_data, title="Model of Final Residues") 
        
        if(file_name == "A12"):
            for i in range(7,46):
                sel_col = res_data['x'] == i
                res_data['f'][sel_col] = 0.0    
            
            img_res = spec_fun.make_image(res_data)
            plot_func.plot_image(img_res, res_data, title="Model of Final Residues")        
              
        """Definition of IMAGE of the clumps in term of Radial Velocity and Separtion"""
        spec_fun.image_separation_velocity_test(res_data, spec_spl, psf_spl)
        
    
    """PLOT of all the spectrums of the group"""
    fig, ax = plt.subplots()
    plt.title('Stellar Spectrum')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Flux')
    w = np.linspace(np.min(udata['w'][10]),np.max(udata['w'][-10]), 3000)
    c = 'green'
    for i in range(0,len_spectrum):
        ax.plot(w, array_spectrum[i],'.',color = c ,ms=1)
        ax.tick_params(color='blue', axis='x', labelsize=10)
        if(i==0):c = 'greenyellow'
        if(i==1):c = 'limegreen'
        
    ax.axvline(x = 1611.019, color = 'orange')
    ax.axvline(x = 1611.511, color = 'red')
    ax.axvline(x = 1612.019, color = 'pink')
    ax.axvline(x = 1612.519, color = 'purple')
    ax.axvline(x = 1613.044, color = 'blue')
    ax.axvline(x = 1613.570, color = 'cyan')
    ax.axvline(x = 1614.107, color = 'olive')
    ax.axvline(x = 1614.645, color = 'brown')
    fig.set_figheight(5)
    fig.set_figwidth(10)
    
    plt.show()
    print("REGION 1: ",line1_position)
    print("REGION 2: ",line2_position)
    print("REGION 3: ",line3_position)
    print("REGION 4: ",line4_position)
   
"""
SESSION 7: this section of the code analyses AUMic data group B13 to identify
the trhee epochs of observation.
"""
if (SESSION==7):
    for i in range(0,2):
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
    line1_position = np.zeros(len_spectrum)
    line2_position = np.zeros(len_spectrum)
    line3_position = np.zeros(len_spectrum)
    line4_position = np.zeros(len_spectrum)
    line5_position = np.zeros(len_spectrum)
    k = 0
    for file in files:
        file_path = os.path.join(root, file)
        udata = np.load(file_path)
    
        print("AU Mic code is running with '", file, " ':" )
        
        """Cleaning data:"""
        clean_data = spec_fun.clean_outliers(udata)
        
        """Rough Stellar Spectrum - RSS"""
        interp_spectrum, err_interp_spectrum = spec_fun.rough_spectrum(clean_data,1)
        
        """
        PSF define by RSS: this PSF is useful to have the first rough PSF function 
        to start the analysis.
        """
        psf_spl, psf, err_psf= spec_fun.fit_psf(clean_data, interp_spectrum, 0.0)
        #plot_func.plot_psf(clean_data, psf, psf_spl, interp_spectrum)
        
        """
        psf_max = np.max(psf_spl(clean_data['r']))
        print("Max of PSF: ",psf_max)
        ind_max = psf_spl(clean_data['r']) == psf_max
        r_star = clean_data['r'][ind_max] #position of the star 
        print("Position of star: ",r_star)
        plot_func.plot_psf(clean_data, psf, psf_spl, interp_spectrum)
        print(udata['r'])
        udata['r'] -= r_star
        print(udata['r'])
        """
        
        ind = np.argsort(udata['r'])
        rdata = udata[ind] 
        
        """Model of Data:"""
        model_data = rdata.copy()#copy of data
        model_data['f'] = (interp_spectrum(rdata['w'])*psf_spl(rdata['r']))# replace the flux with the MODEL
        img = spec_fun.make_image(model_data)#image of the model data
        plot_func.plot_image(img, model_data, title="Model of Data")
        
        """Model of Residues:"""
        res = (rdata['f']-(interp_spectrum(rdata['w'])*psf_spl(rdata['r'])))/rdata['e']
        
        res_data = rdata.copy() #copy of data
        res_data['f'] = res # replace the flux with the residuals
        img_res = spec_fun.make_image(res_data)#create an image of the residuals
        plot_func.plot_image(img_res, res_data, title="Model of Residues")
        
        
        """Stellar Spectrum:"""
        residues=[]
        M = 3
        for i in range(1,M):
            spec_spl, spec, err_spec, weight, wavelength = spec_fun.fit_spec(udata, psf_spl, r_star)
            knots=np.arange(rdata['r'][10],rdata['r'][-10],0.5)
            psf_spl, psf, err_psf =  spec_fun.fit_psf_rough(rdata, spec_spl, knots)
         
            
        w = np.linspace(np.min(udata['w'][10]),np.max(udata['w'][-10]), 3000) 
        array_spectrum.append(spec_spl(w))
        """Final PSF and Spectrum plots:""" 
        #plot_func.plot_psf(rdata, psf, psf_spl, spec_spl)
        #plot_func.plot_spectrum(udata, spec, spec_spl, 0)
        
        """REGION 1: (1567.00->1567.98)nm"""
        w1 = np.arange(1634.125,1634.300,0.02)
        len_w1 = len(w1)
        for i in range(0, len_w1):
            #min_flux1 = np.min(spec_spl(w1))
            ind_min_flux1 = spec_spl(w1).argmin()
            line1_position[k] = w1[ind_min_flux1]
                
        """REGION 2: (1654.76->1654.90)nm"""
        w2 = np.arange(1634.54, 1634.66,0.02)
        len_w2 = len(w2)
        for i in range(0, len_w2):
            #min_flux1 = np.min(spec_spl(w1))
            ind_min_flux2 = spec_spl(w2).argmin()
            line2_position[k] = w2[ind_min_flux2]
               
        """REGION 3: (1650.78->1650.88)nm"""
        w3 = np.arange(1636.60,1636.70,0.02)
        len_w3 = len(w3)
        for i in range(0, len_w3):
            ind_min_flux3 = spec_spl(w3).argmin()
            line3_position[k] = w3[ind_min_flux3]
            
        """REGION 4: (1650.22 ->1650.36)nm"""
        w4 = np.arange(1640.88, 1641.02,0.02)
        len_w4 = len(w4)
        for i in range(0, len_w4):
            ind_min_flux4 = spec_spl(w4).argmin()
            line4_position[k] = w4[ind_min_flux4]
            
        """REGION 5: (1750.46-->1750.58)nm"""
        w5 = np.arange(1641.675,1641.85,0.02)
        len_w5 = len(w5)
        for i in range(0, len_w4):
            ind_min_flux5 = spec_spl(w5).argmin()
            line5_position[k] = w5[ind_min_flux5]
               
        k+=1
        
        """Final image of the residuals:"""
        final_res = (rdata['f']-(spec_spl(rdata['w'])*psf_spl(rdata['r'])))/rdata['e']
        res_data = rdata.copy()
        res_data['f'] = final_res 
        img_res = spec_fun.make_image(res_data)
        plot_func.plot_image(img_res, res_data, title="Model of Final Residues")
        
        """Correction of weird structure inside the image:"""
        if(file_name == "A11"):
            for i in range(165,275):
                sel_col = res_data['x'] == i
                res_data['f'][sel_col] = 0.0    
            for i in range(300,370):
                sel_col = res_data['x'] == i
                res_data['f'][sel_col] = 0.0       
                
            img_res = spec_fun.make_image(res_data)
            plot_func.plot_image(img_res, res_data, title="Model of Final Residues")
        
        if(file_name == "B11"):
            for i in range(70,260):
                sel_col = res_data['x'] == i
                res_data['f'][sel_col] = 0.0    
            
            img_res = spec_fun.make_image(res_data)
            plot_func.plot_image(img_res, res_data, title="Model of Final Residues") 
        
        if(file_name == "A12"):
            for i in range(7,46):
                sel_col = res_data['x'] == i
                res_data['f'][sel_col] = 0.0    
            
            img_res = spec_fun.make_image(res_data)
            plot_func.plot_image(img_res, res_data, title="Model of Final Residues")        
              
        """Definition of IMAGE of the clumps in term of Radial Velocity and Separtion"""
        spec_fun.image_separation_velocity_test(res_data, spec_spl, psf_spl)
        
    
    """PLOT of all the spectrums of the group"""
    fig, ax = plt.subplots()
    plt.title('Stellar Spectrum')
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Flux')
    w = np.linspace(np.min(udata['w'][10]),np.max(udata['w'][-10]), 3000)
    c = 'green'
    for i in range(0,len_spectrum):
        ax.plot(w, array_spectrum[i],'.',color = c ,ms=1)
        ax.tick_params(color='blue', axis='x', labelsize=10)
        if(i==0):c = 'greenyellow'
        if(i==1):c = 'limegreen'
        
    """ax.axvline(x = 1572.022, color = 'orange')
    ax.axvline(x = 1571.712, color = 'red')
    ax.axvline(x = 1571.117, color = 'pink')
    x.axvline(x = 1649.746, color = 'purple')
    ax.axvline(x = 1650.198, color = 'blue')
    ax.axvline(x = 1741.2748, color = 'cyan')
    ax.axvline(x = 1741.64647, color = 'olive')
    ax.axvline(x = 1744.44231, color = 'brown')"""
    fig.set_figheight(5)
    fig.set_figwidth(10)
    
    plt.show()
    print("REGION 1: ",line1_position)
    print("REGION 2: ",line2_position)
    print("REGION 3: ",line3_position)
    print("REGION 4: ",line4_position)
    print("REGION 5: ",line5_position)
    
    
    
    
    
    
    
    
    
    
    
    
 

    
    
et = time.time()
elapsed_time = et- st
print('\n++++++++++++++++++++++++++++\nExecution time:', elapsed_time, 'seconds')


