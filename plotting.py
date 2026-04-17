#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plotting helpers collected into a dedicated service class."""

import spectroscopy_functions as spec_fun
import plot_functions as plot_func
from data_model import Data,Dataset

import numpy as np
import os
import matplotlib.pyplot as plt

class Plotter:
    """A class to handle plotting of various stages of the AU Mic pipeline.
    
    Attributes :
        plot_bool: bool  -- A flag to enable or disable plotting
    """
    def __init__(self, plot_bool:bool):
        self.plot_bool = plot_bool

    def plot_image(self, image, data:Data, fmin=None, fmax=None, title="Flux data as function of pixels"):
        """ Plot the image of the flux data as function of pixels"""
        if fmin is None:
            fmin = np.nanmin(image)
        if fmax is None:
            fmax = np.nanmax(image)
        xmin, xmax = np.min(data.values['x']), np.max(data.values['x'])
        ymin, ymax = np.min(data.values['y']), np.max(data.values['y'])
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

    def make_image(self, data:Data, col='f'):
        """ Returns the original image of the CCD 
        """
        xmin, xmax = data.stats['xmin'], data.stats['xmax']
        ymin, ymax = data.stats['ymin'], data.stats['ymax']
        lenx = xmax - xmin + 1
        leny = ymax - ymin + 1
        # Make image full of NaN
        img = np.empty((leny, lenx))#matrix with shape leny*lenx
        img.fill(np.NaN)#matrix of NaN
        # Replace NaNs in image with values from table
        img[data.values['y'] - ymin, data.values['x'] - xmin] = data.values[col]
        return img


    def plot_initial_data(self, dataset:Dataset):
        if self.plot_bool:
            for i in range(len(dataset.items)):
                data = dataset.items[i]
                img = self.make_image(data)
                file_path = data.file_id
                self.plot_image(img, data, fmin=data.stats['fmin'], fmax=data.stats['fmax'], title=f"Flux data {file_path} as function of pixels ")
    
    def plot_psf(self, udata, psf, psf_spl, spec_spl0, title):
        pass


