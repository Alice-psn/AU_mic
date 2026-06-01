#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Orchestration layer for steps 1 to 5.
This module defines the main pipeline class and the main function to run the analysis.
"""
from file_manager import FileManager
from plotting import Plotter
from spectroscopy_processing import SpectroscopyProcessing
from data_model import Data, Dataset
import numpy as np

def run_step1(dataset, spec_process, plotter):
    print("Step 1")
    plotter.plot_initial_data(dataset)
    spec_process.rss(dataset)
    plotter.plot_rss_dataset(dataset)

    spec_process.psf_rough(dataset)
    plotter.plot_psf_dataset(dataset, 'rough')

    spec_process.sigma_clipping_dataset(dataset, niter=0, klip=10)
    plotter.plot_residuals_dataset(dataset)
    plotter.plot_clipped_data_dataset(dataset)

    spec_process.refine_spectrum_psf(dataset, m_iter=3, dr=5.0)
    plotter.plot_spectrum_dataset(dataset)
    plotter.plot_psf_dataset(dataset, 'final')

    print("Step 1.1")
    spec_process.minimize_dr(dataset)
    plotter.plot_interp_dr_dataset(dataset)



def run_step1_2(dataset, spec_process, plotter):
    print("Step 1.2")
    plotter.plot_initial_data(dataset)

    spec_process.psf_rough(dataset)
    plotter.plot_psf_dataset(dataset, 'rough')
    spec_process.refine_spectrum_psf(dataset, m_iter=3, dr=5.0)
    plotter.plot_spectrum_dataset(dataset)
    plotter.plot_psf_dataset(dataset, 'final 1D')
    
    spec_process.psf_2d(dataset)
    spec_process.refine_spectrum_psf_2d(dataset, dr=5.0)
    plotter.plot_psf_dataset(dataset, '2D')
    plotter.plot_slice_psf_2d(dataset)

    #spec_process.FWHM_estimation(dataset)
    print("Step 2")
    spec_process.group_spectra(dataset)
    plotter.plot_group_spectra(dataset)
    plotter.plot_std_spectrum(dataset)
    
    spec_process.telluric_cut(dataset)
    plotter.plot_telluric_cut_spectra(dataset)
    spec_process.mss(dataset)
    plotter.plot_mss(dataset)

    print("Step 3")
    #spec_process.correction_structures(dataset)
    spec_process.final_residuals(dataset)
    plotter.plot_residuals_dataset(dataset,'Final Residuals')

    spec_process.ccf_image_separation_velocity(dataset,'model')
    plotter.plot_velocity_separation_image(dataset)

def run_step4(dataset_A,dataset_B,spec_process,plotter):
    print("Step 4")
    spec_process.group_velocity_separation(dataset_A, dataset_B)
    plotter.plot_group_velocity_separation(dataset_A, dataset_B)



def main():
    root = 'C:/Users/alice/Documents/Stage Suède/data/'
    plot_root = 'C:/Users/alice/Documents/Stage Suède/plots/'
    stellar_model_path = 'C:/Users/alice/Documents/Stage Suède/spec3700_45.npy'
    plotter = Plotter('save',plot_root) #output_mode can be 'save', 'show', 'both' or 'off' (default is 'show')
    spec_process = SpectroscopyProcessing()

    file_manager_A = FileManager(root,region='A12',nb_files=2, stellar_model_path=stellar_model_path)
    #file_manager_B = FileManager(root,region='B13',nb_files=39, stellar_model_path=stellar_model_path)

    dataset_A = file_manager_A.load_data()
    #dataset_B = file_manager_B.load_data()
    run_step1(dataset_A, spec_process, plotter)
    #run_step1(dataset_B, spec_process, plotter)

    dataset_step1_2_A = dataset_A.build_step1_2_dataset()
    #dataset_step1_2_B = dataset_B.build_step1_2_dataset()
    run_step1_2(dataset_step1_2_A, spec_process, plotter)
    #run_step1_2(dataset_step1_2_B, spec_process, plotter)

    #run_step4(dataset_step1_2_A, dataset_step1_2_B, spec_process, plotter)

if __name__ == "__main__":
    main()