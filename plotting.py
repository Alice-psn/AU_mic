#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plotting helpers collected into a dedicated service class."""

from data_model import Data, Dataset

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

from spectroscopy_processing import SpectroscopyProcessing

class Plotter:
    """A class to handle plotting of various stages of the AU Mic pipeline.
    """
    def __init__(self, output_mode: str = 'show', output_root: str = 'C:/Users/alice/Documents/Stage Suède/plots/'):
        self.output_mode = output_mode
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_name(name: str) -> str:
        return re.sub(r'[^A-Za-z0-9._-]+', '_', name).strip('_')

    def _emit_figure(self, fig, category: str, filename: str, region: str | None = None):
        if self.output_mode in ('save', 'both'):
            category_dir = self.output_root / category
            if region:
                category_dir = category_dir / self._safe_name(region)
            category_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(category_dir / f"{self._safe_name(filename)}.png", dpi=150, bbox_inches='tight')
        if self.output_mode in ('show', 'both'):
            plt.show()
        plt.close(fig)

    def plot_image(self, image, data:Data, fmin=None, fmax=None, title="Flux data as function of pixels", show_colorbar: bool = False, category: str = 'initial_data'):
        """ Plot the image of the flux data as function of pixels"""
        if self.output_mode == 'off':
            return
        if fmin is None:
            fmin = np.nanmin(image)
        if fmax is None:
            fmax = np.nanmax(image)
        xmin, xmax = np.min(data.raw_values['x']), np.max(data.raw_values['x'])
        ymin, ymax = np.min(data.raw_values['y']), np.max(data.raw_values['y'])
        fig, ax = plt.subplots()
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        mappable = ax.imshow(image, origin='lower', aspect='auto', extent=[xmin, xmax, ymin, ymax], vmin=fmin, vmax=fmax)
        if show_colorbar:
            fig.colorbar(mappable, ax=ax)
        ax.tick_params(color='blue', axis='x', labelsize=10)
        fig.set_figheight(5)
        fig.set_figwidth(10)
        self._emit_figure(fig, category=category, filename=title, region=data.region)


    def plot_initial_data(self, dataset:Dataset):
        """ Plot the initial flux data for each file in the dataset. No calculations
        """
        for data in dataset.items:
            img = SpectroscopyProcessing.make_image(data)
            file_path = data.file_id
            self.plot_image(img, data, fmin=data.stats['fmin'], fmax=data.stats['fmax'], title=f"Flux data {file_path} as function of pixels ", show_colorbar=False)

    def plot_rss(self, wavelength, spectrum, title="Rough Stellar Spectrum - RSS", region: str | None = None):
        if self.output_mode == 'off':
            return
        fig, ax = plt.subplots()
        plt.title(title)
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Flux')
        ax.plot(wavelength, spectrum, '.', color='green', markersize=1)
        ax.tick_params(color='blue', axis='x', labelsize=10)
        fig.set_figheight(5)
        fig.set_figwidth(10)
        self._emit_figure(fig, category='rss', filename=title, region=region)

    def plot_rss_dataset(self, dataset: Dataset):
        if self.output_mode == 'off':
            return
        for data in dataset.items:
            self.plot_rss(
                data.derived['rss_wavelength'],
                data.derived['rss_flux'],
                title=f"Rough Stellar Spectrum - RSS - {data.file_id}",
                region=data.region,
            )

    def plot_spectrum(self, data: Data, spec, spec_spl, title=None):
        if self.output_mode == 'off':
            return
        values = data.derived.get('clipped_values', data.derived.get('clean_values', data.raw_values))
        if len(values) < 60:
            return
        ind_w = np.argsort(values['w'])
        wdata = values[ind_w]
        fig, ax = plt.subplots()
        plt.title('Final Spectrum' if title is None else title)
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('Flux')
        w = np.linspace(wdata['w'][30], wdata['w'][-30], 3000)
        ax.plot(w, spec_spl(w), '.', color='green', ms=1)
        ax.tick_params(color='blue', axis='x', labelsize=10)
        fig.set_figheight(5)
        fig.set_figwidth(10)
        file_title = 'Final Spectrum' if title is None else title
        self._emit_figure(fig, category='spectrum', filename=file_title, region=data.region)

    def plot_spectrum_dataset(self, dataset: Dataset):
        if self.output_mode == 'off':
            return
        for data in dataset.items:
            spec = data.derived.get('spec')
            spec_spl = data.derived.get('spec_spl')
            if spec is None or spec_spl is None:
                continue
            self.plot_spectrum(data, spec, spec_spl, title=f"Final Spectrum - {data.file_id}")
    
    def plot_psf(self, data, psf, psf_spl, spec_spl0, title=None):
        if self.output_mode == 'off':
            return
        values = data.derived.get('clean_values', data.raw_values)
        fig, ax1 = plt.subplots()
        plt.title('Final PSF' if title is None else title)
        plt.xlabel('r - distance from slit centre')
        plt.ylabel('Flux')
        r = np.linspace(np.min(values['r']), np.max(values['r']), len(psf))
        ax1.plot(values['r'], values['f']/spec_spl0(values['w']), '.', markersize=1)
        ax1.tick_params(color='blue', axis='x', labelsize=10)
        ax1.plot(values['r'], psf_spl(values['r']), '.', color='orange', markersize=0.5)
        fig.set_figheight(5)
        fig.set_figwidth(10)
        file_title = 'PSF' if title is None else title
        self._emit_figure(fig, category='psf', filename=file_title, region=data.region)

    def plot_psf_dataset(self, dataset: Dataset, stage: str = ''):
        if self.output_mode == 'off':
            return
        stage = stage.lower()
        if stage == 'rough':
            psf_key = 'psf_rough'
            psf_spl_key = 'psf_rough_spl'
            title_prefix = 'Rough PSF'
        else:
            psf_key = 'psf'
            psf_spl_key = 'psf_spl'
            title_prefix = 'Final PSF'

        for data in dataset.items:
            psf = data.derived.get(psf_key)
            psf_spl = data.derived.get(psf_spl_key)
            spec_spl0 = data.derived.get('rss_interp')
            if psf is None or psf_spl is None or spec_spl0 is None:
                continue
            self.plot_psf(data, psf, psf_spl, spec_spl0, title=f"{title_prefix} - {data.file_id}")

    def plot_residuals(self, data: Data, title=None):
        if self.output_mode == 'off':
            return
        image = data.derived.get('residual_image')
        residual_data = data.derived.get('residual_data')
        if image is None or residual_data is None:
            return
        plot_title = 'Residuals' if title is None else title
        self.plot_image(
            image,
            residual_data,
            title=plot_title,
            show_colorbar=True,
            category='residuals',
        )

    def plot_residuals_dataset(self, dataset: Dataset):
        if self.output_mode == 'off':
            return
        for data in dataset.items:
            self.plot_residuals(data, title=f"Residuals - {data.file_id}")

    def plot_clipped_data(self, data: Data, title=None):
        if self.output_mode == 'off':
            return
        clipped_values = data.derived.get('clipped_values')
        if clipped_values is None or len(clipped_values) == 0:
            return

        xmin, xmax = np.min(clipped_values['x']), np.max(clipped_values['x'])
        ymin, ymax = np.min(clipped_values['y']), np.max(clipped_values['y'])
        lenx = xmax - xmin + 1
        leny = ymax - ymin + 1
        img = np.empty((leny, lenx))
        img.fill(np.NaN)
        img[clipped_values['y'] - ymin, clipped_values['x'] - xmin] = clipped_values['f']

        # Keep plot extents based on the full frame for easier visual comparison.
        full_image = np.empty((data.stats['ymax'] - data.stats['ymin'] + 1, data.stats['xmax'] - data.stats['xmin'] + 1))
        full_image.fill(np.NaN)
        full_image[clipped_values['y'] - data.stats['ymin'], clipped_values['x'] - data.stats['xmin']] = clipped_values['f']
        plot_title = 'Clipped Data' if title is None else title
        self.plot_image(full_image,data,fmin=np.min(clipped_values['f']),fmax=np.max(clipped_values['f']),title=plot_title,show_colorbar=True,category='clipped_data')

    def plot_clipped_data_dataset(self, dataset: Dataset):
        if self.output_mode == 'off':
            return
        for data in dataset.items:
            self.plot_clipped_data(data, title=f"Clipped Data - {data.file_id}")


