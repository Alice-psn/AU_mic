#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spectroscopy helpers collected into a dedicated service class."""

from data_model import Data, Dataset

import numpy as np
import scipy as sp
import warnings

class SpectroscopyProcessing:
    """A class to handle spectroscopy processing of the AU Mic pipeline.
    
    Attributes :
        params: dict  -- Processing settings shared across steps
    """
    def __init__(self, params: dict = None):
        self.params = {} if params is None else dict(params)

    @staticmethod
    def make_image(data: Data, col: str = 'f'):
        """ Return the image of raw values of the selected column as function of x and y.
        """
        xmin, xmax = data.stats['xmin'], data.stats['xmax']
        ymin, ymax = data.stats['ymin'], data.stats['ymax']
        lenx = xmax - xmin + 1
        leny = ymax - ymin + 1
        # Make image full of NaN
        img = np.empty((leny, lenx))#matrix with shape leny*lenx
        img.fill(np.NaN)#matrix of NaN
        # Replace NaNs in image with values from table
        img[data.raw_values['y'] - ymin, data.raw_values['x'] - xmin] = data.raw_values[col]
        return img
    
    def clean_outliers(self, data: Data, niter: int = 2, clip: float = 100):
        """Identify outliers in raw_data table and replace
        them with median of surrounding pixels.
        """
        # Identify outliers and replace them with NaN
        img = self.make_image(data)
        err = self.make_image(data, col='e')
        #image of elements which are the median between original flux and the flux of column before and after 
        filt = self.moving_median(img)
        
        #Criteria to select outliers 
        ind = np.abs(img-filt) > clip*err
        #where ind is true replace with NaN
        img[ind] = np.NaN

        # Interpolate over identified outliers in image
        cimg = self.clean_image(img, niter=niter)
        
        # Store the clean data inside the original object without touching raw_values
        xmin, ymin = data.stats['xmin'], data.stats['ymin']
        clean_values = data.raw_values.copy()
        clean_values['f'] = cimg[data.raw_values['y'] - ymin, data.raw_values['x'] - xmin]
        data.derived['clean_values'] = clean_values
        data.masks['outliers'] = ind
        data.derived['clean_image'] = cimg
        return data

    @staticmethod
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

    @staticmethod
    def clean_image(org_img, niter=2):
        """Clean image by replacing NaN with 
        median values (along axis=1)
        """
        img = org_img.copy()
        #create a image of median value
        for _ in range(niter):
            med_img = SpectroscopyProcessing.moving_median(img)
            img[np.isnan(org_img)] = med_img[np.isnan(org_img)]
        return img

    def _working_values(self, data: Data):
        return data.derived.get('clean_values', data.raw_values)  # Use clean values if available, otherwise fall back to raw values

    def rough_spectrum(self, data: Data):
        """Compute the rough stellar spectrum (RSS) by summing flux values along x-axis for each y position, 
        and then interpolate it as a function of wavelength."""
        values = self._working_values(data)
        x_min = data.stats['xmin']
        x_max = data.stats['xmax']
        x = values['x'].astype(np.int64)
        idx = x - x_min
        n_cols = x_max - x_min + 1

        # Iterate over range(xmin, xmax), so exclude xmax column.
        spectrum_all = np.bincount(idx, weights=values['f'], minlength=n_cols)
        err_all = np.sqrt(np.bincount(idx, weights=values['e'] ** 2, minlength=n_cols))
        spectrum = spectrum_all[:-1]
        err_spectrum = err_all[:-1]

        wavelength = []
        for xi in range(x_min, x_max):
            sel_col = x == xi
            if np.any(sel_col):
                w_med = np.median(values['w'][sel_col])
            else:
                w_med = np.nan
            wavelength.append(w_med)
        wavelength = np.asarray(wavelength)

        interp_spectrum = lambda w: np.interp(w, wavelength, spectrum)
        err_interp_spectrum = lambda w: np.interp(w, wavelength, err_spectrum)
        return interp_spectrum, err_interp_spectrum, wavelength, spectrum, err_spectrum

    def rss(self, dataset: Dataset, niter: int = 2, clip: float = 100):
        """Compute the rough stellar spectrum (RSS) for each data item in the dataset, and store it in the derived quantities."""
        for data in dataset.items:
            self.clean_outliers(data, niter=niter, clip=clip)
            interp_spectrum, err_interp_spectrum, wavelength, spectrum, err_spectrum = self.rough_spectrum(data)
            data.derived['rss_interp'] = interp_spectrum
            data.derived['rss_err_interp'] = err_interp_spectrum
            data.derived['rss_wavelength'] = np.asarray(wavelength)
            data.derived['rss_flux'] = np.asarray(spectrum)
            data.derived['rss_err'] = np.asarray(err_spectrum)
        return dataset
    
    def fit_psf_rough(self, data: Data, spec_fun=None):
        """Fit a rough PSF by dividing the flux values by the interpolated spectrum, and then fit a spline"""
        values = self._working_values(data)
        if spec_fun is None:
            spec_fun = data.derived.get('rss_interp')
        if spec_fun is None:
            raise ValueError("rss_interp is missing. Run rss() before fit_psf_rough().")

        ind = np.argsort(values['r'])
        rdata = values[ind]

        denom = spec_fun(rdata['w'])
        valid = np.isfinite(denom) & (denom != 0.0) & np.isfinite(rdata['f']) & np.isfinite(rdata['e'])
        rdata = rdata[valid]
        denom = denom[valid]

        psf = rdata['f'] / denom
        err_psf = rdata['e'] / denom
        with np.errstate(divide='ignore', invalid='ignore'):
            weight = (1.0 / err_psf) ** 2
        weight[~np.isfinite(weight)] = 0.0

        min_slit = int(np.min(rdata['r']))
        max_slit = int(np.max(rdata['r']))
        knots = np.arange(min_slit, max_slit, 1.0)

        if len(knots) < 1:
            raise ValueError("Could not build rough PSF knots.")

        psf_spl = sp.interpolate.LSQUnivariateSpline(rdata['r'], psf, knots, weight)
        return psf_spl, psf, err_psf
   
    def psf_rough(self, dataset: Dataset):
        for data in dataset.items:
            psf_rough_spl, psf_rough, err_psf_rough = self.fit_psf_rough(data)
            data.derived['psf_rough_spl'] = psf_rough_spl
            data.derived['psf_rough'] = np.asarray(psf_rough)
            data.derived['err_psf_rough'] = np.asarray(err_psf_rough)
        return dataset
    
    def compute_model_data(self, data: Data, rss_interp=None, psf_spl=None):
        values = data.raw_values
        if rss_interp is None:
            rss_interp = data.derived.get('rss_interp')
        if psf_spl is None:
            psf_spl = data.derived.get('psf_rough_spl')

        model_data = data.copy()
        model_data.raw_values['f'] = rss_interp(values['w']) * psf_spl(values['r'])
        return model_data
    
    def compute_residuals(self, data: Data, rss_interp=None, psf_spl=None):
        values = data.raw_values
        if rss_interp is None:
            rss_interp = data.derived.get('rss_interp')
        if psf_spl is None:
            psf_spl = data.derived.get('psf_rough_spl')

        res = (values['f'] - rss_interp(values['w']) * psf_spl(values['r'])) / values['e']
        res_data = data.copy()
        res_data.raw_values['f'] = res
        return res_data
    
    @staticmethod
    def sigma_clipping_mask(residual: np.ndarray, niter:int=100, klip:float=10):
        sel = np.ones(len(residual), dtype='?')
        for _ in range(niter):
            sigma = np.std(residual[sel])
            sel = np.abs(residual) < klip * sigma
        return sel

    def compute_sigma_clipping_mask(self, data: Data, rss_interp=None, psf_spl=None, niter:int=100, klip:float=10):
        res_data = self.compute_residuals(data, rss_interp, psf_spl)
        sel = self.sigma_clipping_mask(res_data.raw_values['f'], niter, klip)
        return sel

    def sigma_clipping(self, data: Data, niter:int=100, klip:float=10):
        values = data.raw_values
        rss_interp = data.derived.get('rss_interp')
        psf_spl = data.derived.get('psf_rough_spl')

        model_data = self.compute_model_data(data, rss_interp, psf_spl)
        res_data = self.compute_residuals(data, rss_interp, psf_spl)
        sel = self.sigma_clipping_mask(res_data.raw_values['f'], niter=niter, klip=klip)

        clipped_values = values[sel]
        data.derived['model_data'] = model_data
        data.derived['residual_data'] = res_data
        data.derived['residual_image'] = self.make_image(res_data)
        data.derived['sigma_mask'] = sel
        data.derived['clipped_values'] = clipped_values
        data.clipped_values = clipped_values
        data.masks['sigma_clip'] = sel
        return data

    def sigma_clipping_dataset(self, dataset: Dataset, niter:int=2, klip:float=10):
        for data in dataset.items:
            self.sigma_clipping(data, niter=niter, klip=klip)
        return dataset


    def fit_psf_from_values(self, values: dict, spec_fun):
        ind = np.argsort(values['r'])
        data_sorted = values[ind]

        denom = spec_fun(data_sorted['w'])
        valid = np.isfinite(denom) & (denom != 0.0) & np.isfinite(data_sorted['f']) & np.isfinite(data_sorted['e'])
        data_sorted = data_sorted[valid]

        psf = data_sorted['f'] / spec_fun(data_sorted['w'])
        err_psf = data_sorted['e'] / spec_fun(data_sorted['w'])
        with np.errstate(divide='ignore', invalid='ignore'):
            weight = (1.0 / err_psf) ** 2
        weight[~np.isfinite(weight)] = 0.0

        if len(data_sorted['r']) > 2:
            knots = np.arange(data_sorted['r'][1], data_sorted['r'][-1], 0.5)
        else:
            knots = np.arange(np.min(data_sorted['r']), np.max(data_sorted['r']), 0.5)
        if len(knots) < 1:
            raise ValueError("Could not build PSF knots.")

        psf_spl = sp.interpolate.LSQUnivariateSpline(data_sorted['r'], psf, knots, weight)
        return psf_spl, psf, err_psf

    def fit_spec(self, values: dict, psf_fun, r_star: float, dr: float=5.0, spec_resolution: float=0.015):
        ind = np.argsort(values['w'])
        wdata = values[ind]
        ind_dr = np.abs(wdata['r'] - r_star) < dr
        data = wdata[ind_dr]

        denom = psf_fun(data['r'])
        valid = np.isfinite(denom) & (denom != 0.0) & np.isfinite(data['f']) & np.isfinite(data['e'])
        data = data[valid]

        spec = data['f'] / psf_fun(data['r'])
        err_spec = data['e'] / psf_fun(data['r'])
        with np.errstate(divide='ignore', invalid='ignore'):
            weight = (psf_fun(data['r']) / err_spec) ** 2
        weight[~np.isfinite(weight)] = 0.0

        knots = np.arange(data['w'][10], data['w'][-10], spec_resolution)
        if len(knots) < 1:
            raise ValueError("Could not build spectrum knots.")

        wavelength = data['w']
        spec_spl = sp.interpolate.LSQUnivariateSpline(data['w'], spec, knots, weight)
        return spec_spl, spec, err_spec, weight, wavelength

    def refine_spectrum_psf(self, dataset: Dataset, m_iter: int = 3, dr: float = 5.0):
        for data in dataset.items:
            clipped_values = data.derived.get('clipped_values')
            psf_rough_spl = data.derived.get('psf_rough_spl')

            # Determine r_star after sigma-clipped selection.
            ind_max = np.argmax(psf_rough_spl(clipped_values['r']))
            r_star = clipped_values['r'][ind_max]
            data.derived['r_star'] = float(r_star)

            rdata = clipped_values[np.argsort(clipped_values['r'])]
            psf_spl = psf_rough_spl

            for _ in range(m_iter):
                spec_spl, spec, err_spec, weight, wavelength = self.fit_spec(clipped_values, psf_spl, r_star, dr=dr)
                psf_spl, psf, err_psf = self.fit_psf_from_values(rdata, spec_spl)

            data.derived['spec_spl'] = spec_spl
            data.derived['spec'] = np.asarray(spec)
            data.derived['err_spec'] = np.asarray(err_spec)
            data.derived['weight'] = np.asarray(weight)
            data.derived['wavelength'] = np.asarray(wavelength)
            data.derived['psf_spl'] = psf_spl
            data.derived['psf'] = np.asarray(psf)
            data.derived['err_psf'] = np.asarray(err_psf)
        return dataset

    def minimize_dr(self, dataset: Dataset):
        """Estimate dr(w) and apply r-correction for each file."""
        for item in dataset.items:
            psf_spl = item.derived.get('psf_spl')
            spec_spl = item.derived.get('spec_spl')
            if psf_spl is None or spec_spl is None:
                continue

            # Flux normalization
            ind = np.argsort(item.raw_values['w'])
            wdata = item.raw_values[ind].copy()
            wdata['f'] = wdata['f'] / spec_spl(wdata['w'])

            # Definition of wavelength range
            num_delta_w = 2 * int(np.max(wdata['w']) - np.min(wdata['w'])) + 1
            w0_array = np.zeros(num_delta_w)
            dr_array = np.zeros(num_delta_w)
            chi2_array = np.zeros(num_delta_w)
            w0 = float(np.min(wdata['w']) + 0.25)
            dw = 0.5

            # Selection of inner region of PSF.
            ind_max = np.argmax(psf_spl(wdata['r']))
            r0 = wdata['r'][ind_max]
            r_data = wdata[np.abs(wdata['r'] - r0) < 15.0]

            for i in range(num_delta_w):
                sel_data = r_data[np.abs(r_data['w'] - w0) < dw]
                best_dr = 0.0
                best_chi2 = np.nan

                if len(sel_data) > 0:
                    r = sel_data['r']
                    flux = sel_data['f']
                    err_flux = sel_data['e']
                    chi2 = lambda dr: np.sum((flux-psf_spl(r-dr))**2.0/err_flux)
                    res = sp.optimize.minimize_scalar(chi2, bounds=(-2.0, 2.0), method='bounded')
                    best_dr = float(res.x)
                    best_chi2 = float(res.fun)

                w0_array[i] = w0
                dr_array[i] = best_dr
                chi2_array[i] = best_chi2
                w0 += 0.5

            knots = np.linspace(w0_array[3], w0_array[-3], 7)
            dr_spl = sp.interpolate.LSQUnivariateSpline(w0_array, dr_array, knots)

            corrected_values = item.raw_values.copy()
            corrected_values['r'] = corrected_values['r'] - dr_spl(corrected_values['w'])

            interp_data = r_data.copy()
            interp_data['r'] = interp_data['r'] - dr_spl(interp_data['w'])

            item.derived['dr_w0'] = w0_array
            item.derived['dr_values'] = dr_array
            item.derived['dr_chi2'] = chi2_array
            item.derived['dr_knots'] = np.asarray(knots)
            item.derived['dr_spl'] = dr_spl
            item.derived['dr_wdata'] = wdata
            item.derived['dr_norm_data'] = r_data
            item.derived['interp_dr_data'] = interp_data
            item.derived['r_corrected_values'] = corrected_values
        return dataset