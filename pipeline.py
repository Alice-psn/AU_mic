#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""High-level orchestration layer for steps 1 to 5.
This module defines the main pipeline class and the main function to run the analysis.
"""
from file_manager import FileManager
from plotting import Plotter
from spectroscopy_processing import SpectroscopyProcessing

def run_step1(dataset, spec_process, plotter):
    plotter.plot_initial_data(dataset)

    spec_process.rss(dataset)
    plotter.plot_rss_dataset(dataset)

    spec_process.psf_rough(dataset)
    plotter.plot_psf_dataset(dataset, 'rough')

    spec_process.sigma_clipping_dataset(dataset)
    plotter.plot_residuals_dataset(dataset)
    plotter.plot_clipped_data_dataset(dataset)

    spec_process.refine_spectrum_psf(dataset, m_iter=3, dr=5.0)
    plotter.plot_spectrum_dataset(dataset)
    plotter.plot_psf_dataset(dataset, 'final')

    spec_process.minimize_dr(dataset)
    plotter.plot_interp_dr_dataset(dataset)


def main(): #plot all spectra related to a region
    root = 'C:/Users/alice/Documents/Stage Suède/data/'
    plot_root = 'C:/Users/alice/Documents/Stage Suède/plots/'
    region = 'A12'
    file_manager = FileManager(root,region,nb_files=8)
    plotter = Plotter('off',plot_root) #output_mode can be 'save', 'show', 'both' or 'off' (default is 'show')
    spec_process = SpectroscopyProcessing()

    dataset = file_manager.load_data()
    run_step1(dataset, spec_process, plotter)

if __name__ == "__main__":
    main()