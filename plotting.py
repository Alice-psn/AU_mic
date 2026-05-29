#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plotting helpers collected into a dedicated service class."""

from data_model import Data, Dataset

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re
import scipy as sp

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

    def plot_image(self, image, data:Data, fmin=None, fmax=None, title="Flux data as function of pixels", show_colorbar: bool = True, category: str = 'initial_data'):
        """ Plot the image of the flux data as function of pixels"""
        if self.output_mode == 'off':
            return
        if fmin is None:
            fmin = np.nanmin(image)
        if fmax is None:
            fmax = np.nanmax(image)
        xmin, xmax = data.stats['xmin'], data.stats['xmax']
        ymin, ymax = data.stats['ymin'], data.stats['ymax']
        fig, ax = plt.subplots()
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        x_data = np.linspace(xmin, xmax, image.shape[1])
        y_data = np.linspace(ymin, ymax, image.shape[0])
        bar = plt.pcolormesh(x_data, y_data, image, vmin=fmin, vmax=fmax, shading='auto')
        if show_colorbar:
            fig.colorbar(bar, ax=ax)
        ax.tick_params(color='blue', axis='x', labelsize=10)
        fig.set_figheight(5)
        fig.set_figwidth(10)
        self._emit_figure(fig, category=category, filename=title, region=data.region)

    def plot_initial_data(self, dataset:Dataset):
        """ Plot the initial flux data for each file in the dataset
        """
        for data in dataset.items:
            img = SpectroscopyProcessing.make_image(data)
            file_path = data.file_id
            self.plot_image(img, data, fmin=data.stats['fmin'], fmax=data.stats['fmax'],
                            title=f"Flux data {file_path} as function of pixels ", show_colorbar=True)

    """
    ************ Spectra *************
    """

    def plot_rss(self, wavelength: np.ndarray, spectrum: np.ndarray, title="Rough Stellar Spectrum - RSS", region: str = None):
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

    def plot_spectrum(self, data: Data, spec_spl, title=None):
        if self.output_mode == 'off':
            return
        values = data.derived.get('clipped_values', data.derived.get('clean_values', data.raw_values))
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
            spec_spl = data.derived.get('spec_spl')
            if spec_spl is None:
                continue
            self.plot_spectrum(data, spec_spl, title=f"Final Spectrum - {data.file_id}")
    
    """
    ************ PSF *************
    """

    def plot_psf(self, data: Data, psf_spl, spectrum_spl, title=None):
        if self.output_mode == 'off':
            return
        values = data.derived.get('clipped_values', data.derived.get('clean_values', data.raw_values)) #not clear
        fig, ax1 = plt.subplots()
        plt.title('Final PSF' if title is None else title)
        plt.xlabel('r - distance from slit centre')
        plt.ylabel('Flux')
        ax1.plot(values['r'], values['f']/spectrum_spl(values['w']), '.', markersize=1)
        ax1.tick_params(color='blue', axis='x', labelsize=10)
        ax1.plot(values['r'], psf_spl(values['r']), '.', color='orange', markersize=0.5)
        fig.set_figheight(5)
        fig.set_figwidth(10)
        file_title = 'PSF' if title is None else title
        self._emit_figure(fig, category='psf', filename=file_title, region=data.region)

    def plot_psf_2d(self, data: Data, psf_2d_spl, title=None):
        """Plot 2D PSF as a heatmap over (r, w) grid."""
        if self.output_mode == 'off':
            return
        
        psf_2d_r = data.derived.get('psf_2d_r')
        psf_2d_w = data.derived.get('psf_2d_w')
        
        if psf_2d_r is None or psf_2d_w is None or len(psf_2d_r) == 0:
            return
        
        # Create regular grid for evaluation
        r_min, r_max = np.min(psf_2d_r), np.max(psf_2d_r)
        w_min, w_max = np.min(psf_2d_w), np.max(psf_2d_w)
        nr, nw = 100, 100
        r_grid = np.linspace(r_min, r_max, nr)
        w_grid = np.linspace(w_min, w_max, nw)
        R, W = np.meshgrid(r_grid, w_grid)
        
        # Evaluate spline on grid
        Z = psf_2d_spl(R, W, grid=False)
        
        # Clip to physical bounds [0, 0.35]
        #Z_clipped = np.clip(Z, 0, 0.35)
        
        # Create heatmap
        fig, ax = plt.subplots()
        plt.title('2D PSF (r, w)' if title is None else title)
        plt.xlabel('r - distance from star centre')
        plt.ylabel('Wavelength [nm]')
        bar = ax.pcolormesh(r_grid, w_grid, Z, shading='auto', cmap='viridis', vmin=0, vmax=0.35)
        fig.colorbar(bar, ax=ax, label='Normalized PSF')
        ax.tick_params(color='blue', axis='x', labelsize=10)
        fig.set_figheight(7)
        fig.set_figwidth(10)
        file_title = '2D PSF' if title is None else title
        self._emit_figure(fig, category='psf_2d', filename=file_title, region=data.region)

    def plot_slice_psf_2d(self, dataset: Dataset):
        if self.output_mode == 'off':
            return

        for data in dataset.items:
            psf_2d_spl = data.derived.get('psf_2d_spl')
            if psf_2d_spl is None:
                continue

            psf_w = data.derived.get('psf_2d_w') # data sorted by w
            psf_r = data.derived.get('psf_2d_r') # data sorted by r
            psf_vals = data.derived.get('psf_2d')
            if psf_r is None or psf_w is None or psf_vals is None:
                continue

            w_slice = float(np.median(psf_w))
            r_min, r_max = float(np.min(psf_r)), float(np.max(psf_r))
            r_grid = np.linspace(r_min, r_max, 400)

            fig, ax = plt.subplots()
            plot_title = f"PSF slice - {data.file_id}"
            plt.title(plot_title)
            plt.xlabel('r - distance from slit centre')
            plt.ylabel('Normalized PSF')

            # 2D spline slice at w_slice
            z_grid = np.asarray(psf_2d_spl(r_grid, w_slice), dtype=float)
            ax.plot(r_grid, z_grid, '-', color='green', lw=1.5, label='2D PSF slice w=w_slice')

            # Data points near the selected wavelength
            tol = max(1e-6, (np.nanmax(psf_w) - np.nanmin(psf_w)) / 50.0) # 1/50 of the total wavelength range
            sel = np.isfinite(psf_w) & (np.abs(psf_w - w_slice) <= tol)
            if np.any(sel):
                ax.plot(np.asarray(psf_r)[sel], np.asarray(psf_vals)[sel], '.', color='black', ms=3, label='data points')

            # 1D radial PSF
            psf_1d = data.derived.get('psf_spl') or data.derived.get('psf_rough_spl')
            if psf_1d is not None:
                p1d = np.asarray(psf_1d(r_grid), dtype=float)
                ax.plot(r_grid, p1d, '--', color='red', lw=1.0, label='1D PSF')

            # Plot knots
            r_knots = data.derived.get('r_knots')
            if r_knots is not None and len(r_knots) > 0:
                r_knots = np.asarray(r_knots, dtype=float)
                k_vals = np.asarray(psf_2d_spl(r_knots, w_slice), dtype=float)
                ax.plot(r_knots, k_vals, 'x', color='blue', ms=6, label='knots')

            ax.legend(loc='best')
            ax.tick_params(color='blue', axis='x', labelsize=10)
            fig.set_figheight(5)
            fig.set_figwidth(10)
            self._emit_figure(fig, category='psf', filename=plot_title, region=data.region)

    def plot_psf_dataset(self, dataset: Dataset, stage: str = ''):
        if self.output_mode == 'off':
            return
        stage = stage.lower()
        # 2D PSF case
        if '2d' in stage:
            for data in dataset.items:
                psf_2d_spl = data.derived.get('psf_2d_spl')
                if psf_2d_spl is None:
                    continue
                self.plot_psf_2d(data, psf_2d_spl, title=f"2D PSF - {data.file_id}")
            return
        
        # 1D PSF cases
        if stage == 'rough':
            psf_spl_key = 'psf_rough_spl'
            spec_spl_key = 'rss_interp'
            title_prefix = 'Rough PSF'
        else:
            psf_spl_key = 'psf_spl'
            spec_spl_key = 'spec_spl'
            title_prefix = 'Final PSF'

        for data in dataset.items:
            psf_spl = data.derived.get(psf_spl_key)
            spectrum_spl = data.derived.get(spec_spl_key)
            if psf_spl is None or spectrum_spl is None:
                continue
            self.plot_psf(data, psf_spl, spectrum_spl, title=f"{title_prefix} - {data.file_id}")

    """
    ************ Residuals *************
    """

    def plot_residuals(self, data: Data, title=None):
        if self.output_mode == 'off':
            return
        image = data.derived['residual_image']
        residual_data = data.derived['residual_data']
        if image is None or residual_data is None:
            return
        plot_title = 'Residuals' if title is None else title
        self.plot_image(image,residual_data,fmin=-10.0,fmax=10.0,title=plot_title,show_colorbar=True,category='residuals')

    def plot_residuals_dataset(self, dataset: Dataset, title_prefix: str = 'Residuals'):
        if self.output_mode == 'off':
            return
        for data in dataset.items:
            self.plot_residuals(data, title=f"{title_prefix} - {data.file_id}")

    """
    ************ Clipped data *************
    """

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

    """
    ************ r correction *************
    """

    def plot_interp_dr(self, dr_spl, w0_array, dr_array, wdata, knots, title='dr correction for each interval dw', region: str = None):
        if self.output_mode == 'off':
            return
        fig, ax = plt.subplots()
        plt.title(title)
        plt.xlabel('Wavelength [nm]')
        plt.ylabel('dr [pixel unit]')
        ax.plot(w0_array, dr_array, '.', color='green', ms=3, label='Best dr per dw')
        ax.plot(wdata['w'], dr_spl(wdata['w']), '-', color='red', ms=1, label='dr spline')

        if knots is not None and len(knots) > 0:
            ax.plot(knots, dr_spl(knots), 'x', color='black', ms=5, label='Knots')

        ax.tick_params(color='blue', axis='x', labelsize=10)
        ax.legend(loc='best')
        fig.set_figheight(5)
        fig.set_figwidth(10)
        self._emit_figure(fig, category='interp_dr', filename=title, region=region)

    def plot_interp_dr_dataset(self, dataset: Dataset, title=None):
        if self.output_mode == 'off':
            return
        for data in dataset.items:
            dr_spl = data.derived.get('dr_spl')
            w0_array = data.derived.get('dr_w0')
            dr_array = data.derived.get('dr_values')
            wdata = data.derived.get('dr_wdata')
            knots = data.derived.get('dr_knots')
            if dr_spl is None or w0_array is None or dr_array is None or wdata is None:
                continue
            plot_title = f"dr correction for each interval dw - {data.file_id}" if title is None else title
            self.plot_interp_dr(dr_spl, w0_array, dr_array, wdata, knots, title=plot_title, region=data.region)

    """
    ************ Group spectra *************
    """

    def plot_group_spectra(self, dataset: Dataset):
        if self.output_mode == 'off':
            return

        w = dataset.products.get('step2_w_grid')
        r = dataset.products.get('step2_r_grid')
        group_spec = dataset.products.get('group_spec')
        group_psf = dataset.products.get('group_psf')
        one_frame_spec = dataset.products.get('one_frame_spec')
        if w is None or group_spec is None or one_frame_spec is None:
            return

        eps = 1e-12
        group_spec_norm = group_spec / (np.nanmedian(group_spec, axis=1, keepdims=True) + eps)
        one_frame_norm = one_frame_spec / (np.nanmedian(one_frame_spec, axis=1, keepdims=True) + eps)
        group = np.arange(group_spec_norm.shape[0])

        fig, ax = plt.subplots()
        ax.set_title('Step 2 - Group spectra image')
        ax.set_xlabel('Wavelength [nm]')
        ax.set_ylabel('Frame index')
        bar = ax.pcolormesh(w, group, group_spec_norm, shading='auto')
        fig.colorbar(bar, ax=ax)
        fig.set_figheight(5)
        fig.set_figwidth(10)
        self._emit_figure(fig, category='group_spectra', filename='group_spectra_image', region=dataset.region)

        fig, ax = plt.subplots()
        ax.set_title('Step 2 - Group spectra')
        ax.set_xlabel('Wavelength [nm]')
        ax.set_ylabel('Normalized flux')
        for i in range(group_spec_norm.shape[0]):
            ax.plot(w, group_spec_norm[i], lw=0.8, alpha=0.7)
        fig.set_figheight(5)
        fig.set_figwidth(10)
        self._emit_figure(fig, category='group_spectra', filename='group_spectra_lines', region=dataset.region)

        fig, ax = plt.subplots()
        ax.set_title('Step 2 - One reference frame image')
        ax.set_xlabel('Wavelength [nm]')
        ax.set_ylabel('Frame index')
        bar = ax.pcolormesh(w, group, one_frame_norm, shading='auto')  #one_frame_norm(w) actually processes w_shift 
        fig.colorbar(bar, ax=ax)
        fig.set_figheight(5)
        fig.set_figwidth(10)
        self._emit_figure(fig, category='group_spectra', filename='one_frame_spectra_image', region=dataset.region)

        fig, ax = plt.subplots()
        ax.set_title('Step 2 - One reference frame spectra')
        ax.set_xlabel('Wavelength [nm]')
        ax.set_ylabel('Normalized flux')
        for i in range(one_frame_norm.shape[0]):
            ax.plot(w, one_frame_norm[i], lw=0.8, alpha=0.7)
        fig.set_figheight(5)
        fig.set_figwidth(10)
        self._emit_figure(fig, category='group_spectra', filename='one_frame_spectra_lines', region=dataset.region)

        if group_psf is not None and r is not None:
            fig, ax = plt.subplots()
            ax.set_title('Step 2 - Group PSF')
            ax.set_xlabel('r - distance from slit centre')
            ax.set_ylabel('Flux')
            for i in range(group_psf.shape[0]):
                ax.plot(r, group_psf[i], lw=0.8, alpha=0.7)
            fig.set_figheight(5)
            fig.set_figwidth(10)
            self._emit_figure(fig, category='group_spectra', filename='group_psf_lines', region=dataset.region)

    def plot_std_spectrum(self, dataset: Dataset):
        if self.output_mode == 'off':
            return

        w = dataset.products.get('step2_w_grid')
        one_frame_spec = dataset.products.get('one_frame_spec')
        if w is None or one_frame_spec is None:
            return

        one_frame_norm = one_frame_spec / (np.nanmedian(one_frame_spec, axis=1, keepdims=True) + 1e-12)
        std_flux = np.nanstd(one_frame_norm, axis=0)
        mean_std_flux = float(np.nanmean(std_flux))

        fig, ax = plt.subplots()
        ax.set_title('Step 2 - Std of one-frame spectra')
        ax.set_xlabel('Wavelength [nm]')
        ax.set_ylabel('Std flux')
        ax.plot(w, std_flux, ms=2, color='green', label='std spectrum')
        ax.axhline(mean_std_flux, color='red', lw=1.0, label='mean std')
        ax.legend(loc='best')
        fig.set_figheight(5)
        fig.set_figwidth(10)
        self._emit_figure(fig, category='group_spectra', filename='std_spectrum', region=dataset.region)

    def plot_telluric_cut_spectra(self, dataset: Dataset):
        if self.output_mode == 'off':
            return

        cut_spec = dataset.products.get('cut_spec')
        cut_wavelength = dataset.products.get('cut_wavelength')
        if cut_spec is None or cut_wavelength is None:
            return

        fig, ax = plt.subplots()
        ax.set_title('Stellar Spectra after telluric lines removal')
        ax.set_xlabel('Wavelength [nm]')
        ax.set_ylabel('Flux')

        for spec, wavelength in zip(cut_spec, [cut_wavelength] * len(cut_spec)):
            if len(spec) == 0:
                continue
            med = np.nanmedian(spec)
            if not np.isfinite(med) or med == 0.0:
                med = 1.0
            ax.plot(wavelength, spec / med, '.', ms=1)

        ax.tick_params(color='blue', axis='x', labelsize=10)
        fig.set_figheight(5)
        fig.set_figwidth(10)
        self._emit_figure(fig, category='group_spectra', filename='spectra_after_telluric_cut', region=dataset.region)

    def plot_mss(self, dataset: Dataset):
        """
        Plot the Master Stellar Spectrum from the fitted spline.
        """
        master_wavelength = dataset.products['master_wavelength_BRF']
        master_spec_spl = dataset.products['master_spec_spl']
        master_weight = dataset.products['master_weight']
        #master_spec = dataset.products['master_spec']
        
        # Evaluate spline over wavelength range [2800:-3500] indices
        w_eval = np.linspace(master_wavelength[2800],  master_wavelength[-3500], 3000)
        #w_eval = dataset.products['cut_wavelength']
        flux_eval = master_spec_spl(w_eval)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(w_eval, flux_eval, '-', color='green', linewidth=1)
        
        ax.set_xlabel('Wavelength [nm]', fontsize=10)
        ax.set_ylabel('Flux', fontsize=10)
        ax.set_title('Master Stellar Spectrum', fontsize=11)
        ax.tick_params(color='blue', axis='x', labelsize=10)
        
        # Save master_spec as .npy in same directory as image
        if self.output_mode in ('save', 'both'):
            category_dir = self.output_root / 'master_spectrum'
            category_dir = category_dir / self._safe_name(dataset.region)
            category_dir.mkdir(parents=True, exist_ok=True)
            master_img = np.empty((len(master_wavelength), 3))
            master_img[:, 0] = master_wavelength
            master_img[:, 1] = master_spec_spl(master_wavelength)
            master_img[:, 2] = master_weight
            np.save(category_dir / 'master_img.npy', master_img)
        
        self._emit_figure(fig, category='master_spectrum', filename='master_stellar_spectrum', region=dataset.region)

    def plot_velocity_separation_image(self, dataset: Dataset):
        if self.output_mode == 'off':
            return

        for data in dataset.items:
            img_ccf = data.derived.get('img_separation_velocity')
            r_position = data.derived.get('r_position_centered')
            velocity = data.derived.get('vel_array')

            if img_ccf is None or r_position is None or velocity is None:
                continue

            fig, ax = plt.subplots()
            ax.set_title(f"Radial Velocity - Separation Image - {data.file_id}")
            ax.set_xlabel('Radial Velocity [km/s]')
            ax.set_ylabel('Separation [pixel units]')
            ax.tick_params(color='blue', axis='x', labelsize=10)

            bar = ax.pcolormesh(velocity, r_position, img_ccf, vmin=0.0, vmax=1.0, shading='auto')
            fig.colorbar(bar, ax=ax)
            fig.set_figheight(5)
            fig.set_figwidth(10)

            self._emit_figure(fig, category='step3_velocity_separation', filename=f'velocity_separation_{data.file_id}', region=data.region)

    def _plot_step4_map(self, img_ccf: np.ndarray, r_position: np.ndarray, vel_array: np.ndarray, title: str, region: str | None = None):
        if self.output_mode == 'off':
            return

        img = np.asarray(img_ccf, dtype=float)
        finite_vals = img[np.isfinite(img)]
        if finite_vals.size == 0:
            return

        vmax = float(np.nanpercentile(finite_vals, 99.0))
        vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = float(np.nanmax(finite_vals)) if finite_vals.size > 0 else 1.0
            if not np.isfinite(vmax) or vmax <= vmin:
                vmax = 1.0

        fig, ax = plt.subplots()
        ax.set_title(title)
        ax.set_xlabel('Radial Velocity [km/s]')
        ax.set_ylabel('Separation [arcsec]')

        r_arcsec = np.asarray(r_position, dtype=float) * 0.0373
        bar = ax.pcolormesh(vel_array, r_arcsec, img, shading='auto', vmin=vmin, vmax=vmax)
        fig.colorbar(bar, ax=ax)
        fig.set_figheight(5)
        fig.set_figwidth(10)
        self._emit_figure(fig, category='step4_velocity_separation', filename=title, region=region)

    def plot_group_velocity_separation(self, dataset_A: Dataset, dataset_B: Dataset):
        if self.output_mode == 'off':
            return

        products = dataset_A.products if dataset_A.products.get('step4_night_slit_maps') is not None else dataset_B.products

        night_slit_maps = products.get('step4_night_slit_maps')
        night_combined = products.get('step4_night_combined')
        all_nights = products.get('step4_all_nights_combined')
        r_ref = products.get('step4_r_position')
        v_ref = products.get('step4_vel_array')

        if night_slit_maps is None or night_combined is None or all_nights is None or r_ref is None or v_ref is None:
            return

        region_label = products.get('step4_region', f"{dataset_A.region}_{dataset_B.region}")
        night_keys = sorted(night_slit_maps.keys())

        # 1) Per night and slit
        for night in night_keys:
            for slit in sorted(night_slit_maps[night].keys()):
                title = f"Step4 - Night {night} - Slit {slit}"
                self._plot_step4_map(night_slit_maps[night][slit], r_ref, v_ref, title, region_label)

        # 2) Slits combined per night
        for night in night_keys:
            title = f"Step4 - Night {night} - Slits Combined"
            self._plot_step4_map(night_combined[night], r_ref, v_ref, title, region_label)

        # 3) Nights combined
        self._plot_step4_map(all_nights, r_ref, v_ref, 'Step4 - All Nights Combined', region_label)

