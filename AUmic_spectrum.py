#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from file_manager import FileManager
import scipy as sp
from PyAstronomy import pyasl

# First run the pipeline to get the master spectrum of one region 

def plot_whole_spectrum(file_path):
	arr = np.load(file_path, allow_pickle=True)
	plt.plot(arr[:,0], arr[:,1])
	plt.xlabel('Wavelength (nm)')
	plt.ylabel('Flux')
	plt.title('AUmicspectrum')
	plt.show()

def plot_spectrum(file_path, min_w, max_w):
    print(f"Plotting spectrum for wavelength range: {min_w} nm to {max_w} nm")
    arr = np.load(file_path, allow_pickle=True)
    mask = (arr[:,0] >= min_w) & (arr[:,0] <= max_w)
    w = arr[mask][:,0]
    f = arr[mask][:,1]
    plt.plot(w,f)
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Flux')
    plt.title('AUmicspectrum')
    plt.show()

def remove_envelope_fit(model_spec, master_spec, min_w, max_w, w_telluric, plot_bool):
    """
    Remove envelope to have a flat baseline.
    Fit a spline for the flat master spectrum in order to later compute ccf
    """
    def w_f(spec):
        w,f = spec[:,0], spec[:,1]
        mask = (w >= min_w) & (w <= max_w)
        w,f = w[mask], f[mask]
        ind = np.argsort(w)
        w,f = w[ind],f[ind]

        n_bins = 50
        edges = np.linspace(w[0], w[-1], n_bins + 1)
        w_knots = []
        f_knots = []
        for i in range(n_bins):
            sel = (w >= edges[i]) & (w < edges[i + 1])
            if np.any(sel):
                w_knots.append(0.5 * (edges[i] + edges[i + 1]))
                f_knots.append(np.percentile(f[sel], 95.0))

        w_knots = np.asarray(w_knots, dtype=float)
        f_knots = np.asarray(f_knots, dtype=float)
        envelope = np.interp(w, w_knots, f_knots)
        envelope = np.clip(envelope, 1e-12, np.inf)
        return w, f / envelope, mask

    w_master, f_master, mask_master = w_f(master_spec)

    # Correction to vacuum wavelengths
    #w_master = w_master/1.00027 

    # Building splines
    w_model, f_model, _ = w_f(model_spec)
    master_weight = master_spec[:,2][mask_master]
    knots = np.arange(w_master[30], w_master[-30], 0.015)
    master_spec_spl = sp.interpolate.LSQUnivariateSpline(w_master, f_master, knots, master_weight) # for CCF
    knots = np.arange(w_model[30], w_model[-30], 0.015)
    model_spec_spl = sp.interpolate.LSQUnivariateSpline(w_model, f_model, knots) # for rotation broadening

    # Rotational broadening for model spectrum
    w_rot = np.linspace(np.min(w_model), np.max(w_model),len(w_model)) # evenly spaced wavelengths to perform rotational broadening
    f_for_rot = model_spec_spl(w_rot)
    f_model = pyasl.fastRotBroad(w_rot, f_for_rot, 0.3, 7) # limb darkening value ?
    f_model = f_model[15:-15] # sensitive values
    w_model = w_rot[15:-15]

    # Telluric removal for master spectrum plot
    """for w_peak in w_telluric :
        f_master[np.abs(w_master-w_peak) <= 0.15] = None"""

    master_flat = np.empty((len(w_master), 2))
    model_flat_rot = np.empty((len(w_model), 2))
    master_flat[:,0],master_flat[:,1] = w_master,f_master
    model_flat_rot[:,0],model_flat_rot[:,1] = w_model,f_model

    if plot_bool :
        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 7))

        # Realign master spectrum to model
        v_align = 0
        w_master, w_telluric = doppler_shift(w_master, v_align), doppler_shift(np.asarray(w_telluric), v_align) # realign spectra
        
        ax[0].plot(w_master, f_master)
        # Plot tellurics (useless if removed before)
        for w_peak in w_telluric :
             ax[0].plot(w_master[np.abs(w_master-w_peak) <= 0.15], f_master[np.abs(w_master-w_peak) <= 0.15], 'r-', linewidth=2)
        ax[0].set_ylabel('Corrected Flux')
        ax[0].set_title('Master Spectrum')
        #ax[1].plot(w_model,model_spec_spl(w_model))
        #ax[1].set_ylabel('Corrected Flux')
        #ax[1].set_title('Model Spectrum Spline')
        ax[1].plot(w_model, f_model)
        ax[1].set_ylabel('Corrected Flux')
        ax[1].set_title('Model Spectrum after rotational broadening')
        ax[1].set_xlabel('Wavelength (nm)')
        #w_shift = doppler_shift(w_model, float(20))
        #ax[2].plot(w_model[:-50], master_spec_spl(w_shift[:-50]))
        #ax[2].set_ylabel('Corrected Flux')
        #ax[2].set_title('Master shifted by v=20km/s')
        fig.suptitle('AUmicspectrum with Envelope Removed')
        plt.show()
    return model_flat_rot, master_flat, master_spec_spl

def doppler_shift(wavelength: np.ndarray, velocity_kms: float):
        doppler = 1.0 + (velocity_kms / (sp.constants.c * 0.001))
        return wavelength * doppler

def ccf(wavelength: np.ndarray, master_spec_spl: np.ndarray, model_spec: np.ndarray, vel_array: np.ndarray, w_telluric: list):
    ccf_array = np.zeros(len(vel_array))

    # Telluric removal for CCF
    wavelength_copy = wavelength.copy()
    for w_peak in w_telluric :
        wavelength[np.abs(wavelength-w_peak) <= 0.15] = np.nan

    for i, vel in enumerate(vel_array):
        w_shift = doppler_shift(wavelength_copy, float(vel))
        #w_shift_all = doppler_shift(wavelength_copy, float(vel))
        for w_peak in w_telluric :
            w_shift[np.abs(w_shift-w_peak) <= 0.15] = np.nan
        template = master_spec_spl(w_shift)
        model_norm = (model_spec - np.mean(model_spec)) / np.std(model_spec)
        template_norm = (template - np.nanmean(template)) / np.nanstd(template)
        #if i==100 :
        #    plt.plot(wavelength_copy, template_norm)
        #    plt.show()
        ccf_array[i] = float(np.nansum(model_norm * template_norm)) # or np.mean
    vel = vel_array[np.argmax(ccf_array)]
    plt.plot(vel_array, ccf_array)
    plt.xlabel('Velocity (km/s)')
    plt.ylabel('Cross-correlation')
    plt.show()
    return ccf_array,vel

if __name__ == '__main__':
    region = 'A22'
    file_path_model = 'C:/Users/alice/Documents/Stage Suède/spec3700_45.npy'
    file_path_master = f'C:/Users/alice/Documents/Stage Suède/plots/master_spectrum/{region}/master_img.npy'
    file_manager = FileManager('C:/Users/alice/Documents/Stage Suède/data/', region=region, nb_files=1)

    dataset = file_manager.load_data()
    first_data = dataset.items[0]
    min_w, max_w = np.min(first_data.raw_values['w']), np.max(first_data.raw_values['w'])
    #w_telluric = [1683.9371394, 1690.49612308, 1690.63308561, 1687.29272153, 1691.45866535, 1690.89559714, 1693.87072779] # for A12
    #w_telluric = [1609.97739643, 1610.45767904, 1610.9444961, 1611.43784762, 1611.94100083, 1612.48009355, 1612.97017784, 1613.49946888, 1614.02875992, 1614.57438709, 1615.1232815]
    w_telluric = []
    model_spec = np.load(file_path_model, allow_pickle=True) # 2 columns (wavelength, flux)
    master_spec = np.load(file_path_master, allow_pickle=True) # 3 columns (wavelength, flux, weight)

    #plot_whole_spectrum(file_path_master)
    #plot_spectrum(file_path_model, min_w, max_w)
    
    model_flat_rot, master_flat, master_spec_spl = remove_envelope_fit(model_spec, master_spec, min_w, max_w, w_telluric, plot_bool=True)
    ccf_array,vel = ccf(model_flat_rot[:,0], master_spec_spl, model_flat_rot[:,1], np.linspace(-20, 20, 1001), w_telluric)
    print(f"Relative speed between references frames {vel:.2f} km/s")