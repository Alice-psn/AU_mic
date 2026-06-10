#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spectroscopy helpers collected into a dedicated service class."""

from data_model import Data, Dataset

import numpy as np
import scipy as sp
import warnings
import matplotlib.pyplot as plt
from plotting import Plotter
from PyAstronomy import pyasl

try:
    from astropy.time import Time
    from astropy import coordinates as coord, units as u
except Exception:
    Time = None
    coord = None
    u = None

class SpectroscopyProcessing:
    """A class to handle spectroscopy processing of the AU Mic pipeline.
    
    Attributes :
        params: dict  -- Processing settings shared across steps
    """
    def __init__(self, params: dict = None):
        self.params = {} if params is None else dict(params)

    @staticmethod
    def _barycentric_correction_kms(index: int):
        if Time is None or coord is None or u is None:
            return 0.0

        aumic = coord.SkyCoord("20:45:09.5324974119", "-31:20:27.237889841", unit=(u.hourangle, u.deg), frame='icrs')
        eso = coord.EarthLocation.from_geodetic(lat=70.416666667 * u.deg, lon=-24.666667 * u.deg, height=2635 * u.m)

        if index in (0, 20, 21, 22, 23, 24, 25, 26, 28, 29):
            ti = Time(['2023-08-26T04:16:04'], format='isot', scale='utc')
            te = Time(['2023-08-26T05:12:48'], format='isot', scale='utc')
        elif index in (1, 2, 3, 4, 5, 6, 7, 8, 9):
            ti = Time(['2023-08-15T04:49:48'], format='isot', scale='utc')
            te = Time(['2023-08-15T05:49:03'], format='isot', scale='utc')
        elif index in (10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 27):
            ti = Time(['2023-08-23T04:37:29'], format='isot', scale='utc')
            te = Time(['2023-08-23T05:45:44'], format='isot', scale='utc')
        else:
            ti = Time(['2023-08-27T03:51:38'], format='isot', scale='utc')
            te = Time(['2023-08-27T04:47:50'], format='isot', scale='utc')

        barycorr_i = aumic.radial_velocity_correction(obstime=ti, location=eso).to(u.km / u.s)
        barycorr_e = aumic.radial_velocity_correction(obstime=te, location=eso).to(u.km / u.s)
        barycorr = (barycorr_e + barycorr_i) / 2.0
        return float(barycorr.value)

    @staticmethod
    def _estimate_fwhm(values: np.ndarray, psf_spl):
        ind = np.argsort(values['r'])
        rdata = values[ind]
        psf_vals = psf_spl(rdata['r'])
        half_max_psf = np.max(psf_vals) * 0.5
        fwhm_signs = np.sign(psf_vals - half_max_psf)
        zero_crossing = (fwhm_signs[0:-2] != fwhm_signs[1:-1])
        zero_crossing_i = np.where(zero_crossing)[0]

        star_position = float(rdata['r'][np.argmax(psf_vals)])
        fwhm = np.nan
        if len(zero_crossing_i) >= 2:
            fwhm = float(np.abs(rdata['r'][zero_crossing_i[1]] - rdata['r'][zero_crossing_i[0]]))

        psf_signs = psf_vals - half_max_psf
        ind_min = 0
        ind_max = len(psf_signs) - 1
        for i in range(1, len(psf_signs) - 1):
            if psf_signs[i] < 0.0 and psf_signs[i + 1] > 0.0:
                ind_min = i
            if psf_signs[i] > 0.0 and psf_signs[i + 1] < 0.0:
                ind_max = i
        fwhm_for = float(np.abs(rdata['r'][ind_max] + rdata['r'][ind_min]))
        return fwhm, fwhm_for, star_position

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
        img = self.make_image(data)
        err = self.make_image(data, col='e')
        #image of elements which are the median between original flux and the flux of column before and after 
        filt = self.moving_median(img)
        
        #Criteria to select outliers 
        ind = np.abs(img-filt) > clip*err
        img[ind] = np.NaN

        # Interpolate over identified outliers in image
        cimg = self.clean_image(img, niter=niter)
        
        # Store the clean data inside the original object
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

    """
    ************ Rough spectra and PSF *************
    """

    def rough_spectrum(self, data: Data):
        """Compute a rough 1D stellar spectrum by summing flux along each detector x column and assigning each column a median wavelength"""
        values = self._working_values(data)
        x_min = np.min(values['x'])
        x_max = np.max(values['x'])
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
    
    def fit_psf_rough(self, data: Data):
        """Fit a rough PSF by dividing the flux values by the interpolated spectrum, and then fit a spline"""
        values = self._working_values(data)
        spec_fun = data.derived.get('rss_interp')
        if spec_fun is None:
            raise ValueError("rss_interp is missing. Run rss() before fit_psf_rough().")

        ind = np.argsort(values['r'])
        rdata = values[ind]
        denom = spec_fun(rdata['w'])

        psf = rdata['f'] / denom
        err_psf = rdata['e'] / denom
        with np.errstate(divide='ignore', invalid='ignore'):
            weight = (1.0 / err_psf) ** 2
        weight[~np.isfinite(weight)] = 0.0

        # Use quantile-based knots to ensure Schoenberg-Whitney conditions are satisfied
        r_min, r_max = np.min(rdata['r']), np.max(rdata['r'])
        #n_knots = 50
        #knots = np.quantile(rdata['r'], np.linspace(0.1, 0.9, n_knots))
        knots = np.arange(r_min,r_max,1.0) # old code
        # Remove any knots exactly at or very close to data boundaries
        knots = knots[(knots > r_min + 0.1) & (knots < r_max - 0.1)]
        
        psf_spl = sp.interpolate.LSQUnivariateSpline(rdata['r'], psf, knots, weight)
        return psf_spl, psf, err_psf
   
    def psf_rough(self, dataset: Dataset):
        for data in dataset.items:
            psf_rough_spl, psf_rough, err_psf_rough = self.fit_psf_rough(data)
            data.derived['psf_rough_spl'] = psf_rough_spl
            data.derived['psf_rough'] = np.asarray(psf_rough)
            data.derived['err_psf_rough'] = np.asarray(err_psf_rough)
        return dataset
    
    """
    ************ Sigma clipping *************
    """

    # def compute_model_data(self, data: Data, rss_interp, psf_spl):
    #     values = data.raw_values
    #     model_data = data.copy()
    #     model_data.raw_values['f'] = rss_interp(values['w']) * psf_spl(values['r'])
    #     return model_data
    
    def compute_residuals(self, data: Data, rss_interp, psf_spl):
        values = data.raw_values
        res = (values['f'] - rss_interp(values['w']) * psf_spl(values['r'])) / values['e']
        res_data = data.copy()
        res_data.raw_values['f'] = res
        return res_data
    
    @staticmethod
    def sigma_clipping_mask(residual: np.ndarray, niter:int=2, klip:float=3):
        sel = np.ones(len(residual), dtype='?')
        for _ in range(niter):
            sigma = np.std(residual[sel])
            sel = np.abs(residual) < klip * sigma
        return sel

    def sigma_clipping(self, data: Data, niter:int=2, klip:float=3):
        """
        Perform sigma clipping on the data and compute the first residuals
        """
        values = data.raw_values
        rss_interp = data.derived.get('rss_interp')
        psf_spl = data.derived.get('psf_rough_spl')

        res_data = self.compute_residuals(data, rss_interp, psf_spl)
        sel = self.sigma_clipping_mask(res_data.raw_values['f'], niter=niter, klip=klip)

        clipped_values = values[sel]
        #data.derived['model_data'] = model_data
        data.derived['residual_data'] = res_data
        data.derived['residual_image'] = self.make_image(res_data)
        data.derived['sigma_mask'] = sel
        data.derived['clipped_values'] = clipped_values
        data.clipped_values = clipped_values
        data.masks['sigma_clip'] = sel
        return data

    def sigma_clipping_dataset(self, dataset: Dataset, niter:int=2, klip:float=3):
        for data in dataset.items:
            self.sigma_clipping(data, niter=niter, klip=klip)
        return dataset

    """
    ************ Final spectra and PSF *************
    """
    @staticmethod
    def fit_psf_from_values(values, spec_fun):
        """
        Derive PSF from spectrum 
        """
        ind = np.argsort(values['r'])
        data_sorted = values[ind]

        denom = spec_fun(data_sorted['w'])
        psf = data_sorted['f'] / denom
        err_psf = data_sorted['e'] / denom
        with np.errstate(divide='ignore', invalid='ignore'):
            weight = (1.0 / err_psf) ** 2
        weight[~np.isfinite(weight)] = 0.0

        knots = np.arange(data_sorted['r'][1], data_sorted['r'][-1], 0.5)
        psf_spl = sp.interpolate.LSQUnivariateSpline(data_sorted['r'], psf, knots, weight)
        return psf_spl, psf, err_psf
    
    @staticmethod
    def fit_spec(values, psf_fun, r_star: float, dr: float=15.0, spec_resolution: float=0.015):
        """
        Derive spectrum from PSF, considering only data around the centre of the star (dr distance)
        """
        ind = np.argsort(values['w'])
        wdata = values[ind]
        ind_dr = np.abs(wdata['r'] - r_star) < dr
        data = wdata[ind_dr]

        denom = psf_fun(data['r'])
        spec = data['f'] / denom
        err_spec = data['e'] / denom
        with np.errstate(divide='ignore', invalid='ignore'):
            weight = (psf_fun(data['r']) / err_spec) ** 2
        weight[~np.isfinite(weight)] = 0.0

        knots = np.arange(data['w'][10], data['w'][-10], spec_resolution)
        wavelength = data['w']
        spec_spl = sp.interpolate.LSQUnivariateSpline(data['w'], spec, knots, weight)
        return spec_spl, spec, err_spec, weight, wavelength

    def fit_spec_2d(self, data, dr: float=15.0, spec_resolution: float=0.015):
        """
        Derive spectrum from 2D PSF, considering only data around the centre of the star (dr distance)
        """
        psf_2d_spl = data.derived.get('psf_2d_spl')
        spec_spl = data.derived.get('spec_spl')
        r_eval, w_eval, f_eval = data.derived['psf_2d_r'], data.derived['psf_2d_w'], data.derived['psf_2d_f']
        second_arg = data.derived['second_arg']

        # Find r_star using 2D PSF at all wavelengths
        if 'f' in second_arg :
            psf_2d_vals = np.asarray(psf_2d_spl(r_eval, f_eval, grid=False), dtype=float)
        else :
            psf_2d_vals = np.asarray(psf_2d_spl(r_eval, w_eval, grid=False), dtype=float)
        ind_max = np.argmax(psf_2d_vals)
        r_star = r_eval[ind_max] # almost zero
        #print('r_star',r_star)

        values = data.raw_values.copy()
        ind = np.argsort(values['w'])
        wdata = values[ind] #[100:-100] if 'f'
        ind_dr = np.abs(wdata['r'] - r_star) < dr
        wdata = wdata[ind_dr]

        if 'f' in second_arg :
            spec_spl = data.derived['spec_spl']
            f = spec_spl(np.asarray(wdata['w']))
            f = f/np.median(f)
            if np.max(f)>np.max(f_eval):
                print(f'Warning: psf spline is evaluated outside its training values, maximum should be < {np.max(f_eval)} but is = {np.max(f)}')
            denom = psf_2d_spl(wdata['r'], f, grid=False)
        else :
            denom = psf_2d_spl(wdata['r'], wdata['w'], grid=False)
        spec = wdata['f'] / denom
        err_spec = wdata['e'] / denom
        with np.errstate(divide='ignore', invalid='ignore'):
            weight = (denom / err_spec) ** 2
        weight[~np.isfinite(weight)] = 0.0

        knots = np.arange(wdata['w'][10], wdata['w'][-10], spec_resolution)
        wavelength = wdata['w']
        spec_spl = sp.interpolate.LSQUnivariateSpline(wdata['w'], spec, knots, weight)

        data.derived['spec_spl'] = spec_spl
        data.derived['spec'] = np.asarray(spec)
        data.derived['err_spec'] = np.asarray(err_spec)
        data.derived['weight'] = np.asarray(weight)
        data.derived['wavelength'] = np.asarray(wavelength)



        ind = np.argsort(values['w'])
        temp_data = values[ind] #[100:-100]
        f = spec_spl(np.asarray(temp_data['w']))
        f = f/np.median(f)
        #print('fmax after spec',np.max(f))

        return

    def refine_spectrum_psf(self, dataset: Dataset, m_iter: int = 3, dr: float = 15.0):
        for data in dataset.items:
            clipped_values = data.derived.get('clipped_values')
            psf_rough_spl = data.derived.get('psf_rough_spl')

            # Determine r_star after sigma-clipped selection.
            ind_max = np.argmax(psf_rough_spl(clipped_values['r']))
            r_star = clipped_values['r'][ind_max]

            rdata = clipped_values[np.argsort(clipped_values['r'])]
            psf_spl = psf_rough_spl

            for _ in range(m_iter):
                spec_spl, spec, err_spec, weight, wavelength = SpectroscopyProcessing.fit_spec(clipped_values, psf_spl, r_star, dr=dr)
                psf_spl, psf, err_psf = SpectroscopyProcessing.fit_psf_from_values(rdata, spec_spl)

            data.derived['spec_spl'] = spec_spl
            data.derived['spec'] = np.asarray(spec)
            data.derived['err_spec'] = np.asarray(err_spec)
            data.derived['weight'] = np.asarray(weight)
            data.derived['wavelength'] = np.asarray(wavelength)
            data.derived['psf_spl'] = psf_spl
            data.derived['psf'] = np.asarray(psf)
            data.derived['err_psf'] = np.asarray(err_psf)
        return dataset

    def refine_spectrum_psf_2d(self, dataset: Dataset, second_arg: str, dr: float = 15.0):
        """Spectrum fit using 2D PSF (1 iteration for the moment)"""
        for data in dataset.items:
            self.fit_psf_2d(data,second_arg)
            self.fit_spec_2d(data, dr)
            self.fit_psf_2d(data,second_arg)
        return

    def fit_psf_2d(self, data, second_arg:str):
        """
        Compute 2D PSF as function of separation r and of second_arg ('f' for flux or 'w' for wavelength dependance).
        Flux dependance is actually spectrum dependance and is normalized by its median
        """
        values = data.raw_values.copy()
        spec_spl = data.derived.get('spec_spl')

        # Build normalized PSF samples on raw points (except edges)
        ind = np.argsort(values['w'])
        data_sorted = values[ind] # [100:-100] if 'f'
        denom = spec_spl(data_sorted['w'])

        psf = data_sorted['f'] / denom
        err_psf = data_sorted['e'] / denom
        with np.errstate(divide='ignore', invalid='ignore'):
            weight = (1.0 / err_psf) ** 2
        weight[~np.isfinite(weight)] = 0.0

        r = data_sorted['r']
        w = data_sorted['w']
        f = spec_spl(data_sorted['w'])
        f = f/np.median(f)

        n_knots_r, n_knots_w, n_knots_f = 60, 2, 2
        r_unique, w_unique, f_unique = np.unique(r), np.unique(w), np.unique(f)
        r0 = r[np.argmax(psf)]
        u = np.arcsinh((r_unique - r0) / 3.0) # more knots towards the centre
        r_knots = r0 + 3.0 * np.sinh(np.linspace(u[1], u[-2], n_knots_r))
        max_left = r0 - r_unique[1] if len(r_unique) > 1 else 0.0
        max_right = r_unique[-2] - r0 if len(r_unique) > 1 else 0.0
        #u = np.linspace(-1.0, 1.0, n_knots_r)
        #r_knots = np.where(u < 0.0, r0 - max_left * (np.abs(u) ** 2), r0 + max_right * (np.abs(u) ** 2))
        w_knots = np.linspace(w_unique[100],w_unique[-100],n_knots_w)
        f_knots = np.quantile(f_unique, np.linspace(0, 1, n_knots_f+2)[1:-1])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if 'f' in second_arg: 
                psf_2d_spl = sp.interpolate.LSQBivariateSpline(r, f, psf, r_knots, f_knots, w=weight)
            else :
                psf_2d_spl = sp.interpolate.LSQBivariateSpline(r, w, psf, r_knots, w_knots, w=weight)
            
        data.derived['psf_2d_spl'] = psf_2d_spl
        data.derived['r_knots'] = r_knots
        data.derived['psf_2d'] = psf  # Raw normalized PSF
        #data.derived['err_psf_2d'] = 1.0 / np.sqrt(weight + 1e-12)
        data.derived['psf_2d_r'] = r
        data.derived['psf_2d_w'] = w
        data.derived['psf_2d_f'] = f
        data.derived['second_arg'] = second_arg
        return psf_2d_spl

    """
    ************ r correction *************
    """

    def minimize_dr(self, dataset: Dataset):
        """Estimate dr(w) and apply r-correction for each file."""
        for data in dataset.items:
            psf_spl = data.derived.get('psf_spl')
            spec_spl = data.derived.get('spec_spl')
            if psf_spl is None or spec_spl is None:
                continue

            # Flux normalization
            ind = np.argsort(data.raw_values['w'])
            w_data = data.raw_values[ind].copy()
            w_data['f'] = w_data['f'] / spec_spl(w_data['w'])

            # Definition of wavelength range
            num_delta_w = 2 * int(np.max(w_data['w']) - np.min(w_data['w'])) + 1
            w0_array = np.zeros(num_delta_w)
            dr_array = np.zeros(num_delta_w)
            chi2_array = np.zeros(num_delta_w)
            w0 = float(np.min(w_data['w']) + 0.25)
            dw = 0.5
            dr = 15.0

            # Selection of inner region of PSF.
            ind_max = np.argmax(psf_spl(w_data['r']))
            r0 = w_data['r'][ind_max]
            r_data = w_data[np.abs(w_data['r'] - r0) < dr]

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

            corrected_values = data.raw_values.copy()
            corrected_values['r'] = corrected_values['r'] - dr_spl(corrected_values['w']) 
            ind_max = np.argmax(psf_spl(corrected_values['r']))
            r0 = corrected_values['r'][ind_max]
            corrected_values['r'] = corrected_values['r'] - r0 # center the psf on the center of the star

            data.derived['dr_w0'] = w0_array
            data.derived['dr_values'] = dr_array
            data.derived['dr_knots'] = np.asarray(knots)
            data.derived['dr_spl'] = dr_spl
            data.derived['dr_wdata'] = w_data
            data.derived['r_corrected_values'] = corrected_values # these values will be used for the rest of the code
        return dataset

    """
    ************ Telluric cut and structure correction *************
    """

    def FWHM_estimation(self, dataset: Dataset):
        fwhm_arr = []
        fwhm_for_arr = []
        star_pos_arr = []
        for data in dataset.items:
            psf_spl = data.derived.get('psf_spl')
            if psf_spl is None:
                fwhm_arr.append(np.nan)
                fwhm_for_arr.append(np.nan)
                star_pos_arr.append(np.nan)
                continue

            fwhm, fwhm_for, star_position = self._estimate_fwhm(data.raw_values, psf_spl)
            data.derived['fwhm'] = float(fwhm)
            data.derived['fwhm_for'] = float(fwhm_for)
            data.derived['star_position'] = float(star_position)
            fwhm_arr.append(fwhm)
            fwhm_for_arr.append(fwhm_for)
            star_pos_arr.append(star_position)

        dataset.products['fwhm'] = np.asarray(fwhm_arr)
        dataset.products['fwhm_for'] = np.asarray(fwhm_for_arr)
        dataset.products['star_position'] = np.asarray(star_pos_arr)
        return dataset

    def group_spectra(self, dataset: Dataset):
        if len(dataset.items) == 0:
            return dataset

        n_files = len(dataset.items)
        n_wave = 3000

        group_spec = np.full((n_files, n_wave), np.nan, dtype=float)
        group_psf = np.full((n_files, n_wave), np.nan, dtype=float)
        one_frame_spec = np.full((n_files, n_wave), np.nan, dtype=float)
        w_shift = np.full((n_files, n_wave), np.nan, dtype=float)
        barycorr = np.full(n_files, np.nan, dtype=float)
        vel = np.full(n_files, np.nan, dtype=float)

        master_wavelength_BRF = []
        master_spec = []
        master_weight = []
        wavelength_grid = None
        r_grid = None

        for j, data in enumerate(dataset.items):
            spec_spl = data.derived.get('spec_spl')
            psf_spl = data.derived.get('psf_spl')
            if spec_spl is None or psf_spl is None:
                continue

            w_ind = np.argsort(data.raw_values['w'])
            wdata = data.raw_values[w_ind]
            r_ind = np.argsort(data.raw_values['r'])
            rdata = data.raw_values[r_ind]

            i0 = 2000 if len(wdata) > 4001 else 0
            i1 = -2000 if len(wdata) > 4001 else -1
            w = np.linspace(wdata['w'][i0], wdata['w'][i1], n_wave)

            r0 = 2000 if len(rdata) > 4001 else 0
            r1 = -2000 if len(rdata) > 4001 else -1
            r = np.linspace(rdata['r'][r0], rdata['r'][r1], n_wave)

            group_spec[j, :] = spec_spl(w)
            group_psf[j, :] = psf_spl(r) # still 1D PSF

            if wavelength_grid is None:
                wavelength_grid = w.copy()
                r_grid = r.copy()

            barycorr[j] = self._barycentric_correction_kms(j)
            vel[j] = -barycorr[j]
            w_shift[j, :] = self.doppler_shift(w, vel[j])
            one_frame_spec[j, :] = spec_spl(w_shift[j, :])

            spec = data.derived['spec']
            wavelength_data_dr = data.derived['wavelength'] # to make it correspond to the weights
            weight = data.derived['weight']
            if spec is not None and wavelength_data_dr is not None and weight is not None and len(spec) > 0:
                median_spec = np.median(spec)
                if np.isfinite(median_spec) and median_spec != 0.0:
                    master_spec.extend((spec / median_spec).tolist())
                    master_wavelength_shift = self.doppler_shift(wavelength_data_dr, -vel[j]) #was initially -vel[j] 
                    master_wavelength_BRF.extend(master_wavelength_shift.tolist())
                    master_weight.extend(weight.tolist())

        dataset.products['step2_w_grid'] = wavelength_grid
        dataset.products['step2_r_grid'] = r_grid
        dataset.products['group_spec'] = group_spec
        dataset.products['group_psf'] = group_psf
        dataset.products['one_frame_spec'] = one_frame_spec
        dataset.products['master_wavelength_BRF'] = np.asarray(master_wavelength_BRF) # barycentric reference frame
        dataset.products['master_spec'] = np.asarray(master_spec)
        dataset.products['master_weight'] = np.asarray(master_weight)
        return dataset
    
    def doppler_shift(self, wavelength: np.ndarray, velocity_kms: float):
        doppler = 1.0 + (velocity_kms / (sp.constants.c * 0.001))
        return wavelength * doppler

    def telluric_cut(self, dataset: Dataset):
        """
        Return the peak of tellurics in any reference frame close to the Earth reference frame, what matters is the masked region around
        """
        if len(dataset.items) == 0:
            return dataset

        w_grid = dataset.products.get('step2_w_grid') # ERF common grid
        one_frame_spec = dataset.products.get('one_frame_spec')
        if w_grid is None or one_frame_spec is None:
            return dataset

        limit = self.params.get('telluric_cut_limit', 0.042)
        dw = self.params.get('telluric_cut_dw', 0.1)

        one_frame_norm = one_frame_spec / (np.nanmedian(one_frame_spec, axis=1, keepdims=True) + 1e-12)
        std_flux = np.nanstd(one_frame_norm, axis=0)

        w_work = np.asarray(w_grid).copy()
        std_work = std_flux.copy()
        w_telluric = []

        while len(std_work) > 0:
            std_flux_max = np.nanmax(std_work)
            if std_flux_max < limit:
                break

            ind_max = int(np.nanargmax(std_work))
            w_peak = float(w_work[ind_max])
            w_telluric.append(w_peak)

            sel_w = np.abs(w_work - w_peak) > dw
            w_work = w_work[sel_w]
            std_work = std_work[sel_w]

        sel_cut_w = np.ones(len(w_grid), dtype=bool)
        for w_peak in w_telluric:
            sel_cut_w &= np.abs(w_grid - w_peak) > dw

        cut_wavelength = np.asarray(w_grid)[sel_cut_w]
        cut_spec = []
        for j, data in enumerate(dataset.items):
            one_frame = one_frame_spec[j, :]
            cut_spec.append(one_frame[sel_cut_w])
            #data.derived['telluric_mask'] = sel_cut_w
            #data.derived['telluric_cut_wavelength'] = cut_wavelength
            #data.derived['telluric_cut_spec'] = one_frame[sel_cut_w]

        dataset.products['std_flux'] = std_flux
        dataset.products['mean_std_flux'] = float(np.nanmean(std_flux))
        dataset.products['w_telluric'] = np.asarray(w_telluric)
        #dataset.products['telluric_mask'] = sel_cut_w
        #dataset.products['w_no_tell'] = w_work
        #dataset.products['std_flux_no_tell'] = std_work
        dataset.products['cut_wavelength'] = cut_wavelength
        dataset.products['cut_spec'] = cut_spec
        return dataset
    
    def mss(self, dataset: Dataset):
        """
        Calculate the Master Stellar Spectrum from accumulated master data.
        """
        # Extract accumulated master data
        master_wavelength_BRF = np.asarray(dataset.products['master_wavelength_BRF']).flatten()
        master_spec = np.asarray(dataset.products['master_spec']).flatten()
        master_weight = np.asarray(dataset.products['master_weight']).flatten()
        
        # Sort by wavelength
        ind = np.argsort(master_wavelength_BRF)
        master_spec = master_spec[ind]
        master_weight = master_weight[ind]
        master_wavelength_BRF = np.sort(master_wavelength_BRF)
        
        # Build knots with spacing 0.015 nm, avoiding edges
        knots = np.arange(master_wavelength_BRF[30], master_wavelength_BRF[-30], 0.015)
        
        # Fit spline with weights
        master_spec_spl = sp.interpolate.LSQUnivariateSpline(master_wavelength_BRF, master_spec, knots, master_weight)
        
        # Store results in dataset.products
        dataset.products['master_wavelength_BRF'] = master_wavelength_BRF
        #dataset.products['master_spec'] = master_spec
        dataset.products['master_spec_spl'] = master_spec_spl
        dataset.products['master_weight'] = master_weight
        
        return dataset

    def final_residuals(self, dataset: Dataset):
        """
        Compute final residuals using r-corrected data (in data.raw_values)
        Shift the residuals to barycentric reference frame
        """
        for j,data in enumerate(dataset.items):
            spec_spl = data.derived.get('spec_spl')
            psf_2d_spl = data.derived.get('psf_2d_spl')
            psf_spl = data.derived.get('psf_spl')
            r_corrected_values = data.raw_values.copy()

            if spec_spl is None or (psf_2d_spl is None and psf_spl is None) or r_corrected_values is None or len(r_corrected_values) == 0:
                continue

            # Optionally apply structure masking (on r-corrected values).
            use_corrected = bool(data.derived.get('structure_correction_applied', False))
            if use_corrected:
                values = data.derived.get('structure_corrected_values', r_corrected_values)
            else:
                values = r_corrected_values

            # Removing the edge points
            #ind_w = np.argsort(values['w'])
            #rdata = values[ind_w][100:-100]
            ind_r = np.argsort(values) # np.argsort(rdata['r'])
            rdata = values[ind_r]
            second_arg = data.derived['second_arg']
            if psf_2d_spl is not None:
                if 'f' in second_arg:
                    f = spec_spl(rdata['w'])
                    f = f/np.median(f)
                    print('max_f', np.max(f))
                    psf_eval = np.asarray(psf_2d_spl(rdata['r'], f, grid=False), dtype=float)
                else:
                    psf_eval = np.asarray(psf_2d_spl(rdata['r'], rdata['w'], grid=False), dtype=float)
            else:
                psf_eval = np.asarray(psf_spl(rdata['r']), dtype=float)
            model_flux = spec_spl(rdata['w']) * psf_eval
            denom = np.asarray(rdata['e'], dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                final_res = np.where(denom != 0.0, (rdata['f'] - model_flux) / denom, np.nan)

            # Applied structure correction: set masked pixels to zero in residual.
            if use_corrected:
                structure_mask = data.derived.get('structure_mask')
                if structure_mask is not None and len(structure_mask) == len(values):
                    mask_sorted = np.asarray(structure_mask)[ind_r]
                    final_res[mask_sorted] = 0.0

            corrected_table = rdata.copy()
            corrected_table['f'] = final_res

            # Shift to BRF
            barycorr = self._barycentric_correction_kms(j)
            #print(barycorr,'km/s')
            #print(corrected_table['w'][0],'before shift')
            corrected_table['w'] = self.doppler_shift(corrected_table['w'],barycorr)
            #print(corrected_table['w'][0],'after shift')
            res_data_BRF = Data(
                raw_values=corrected_table,
                source_path=data.source_path,
                file_id=data.file_id,
                epoch_index=data.epoch_index,
                region=data.region,
                masks=data.masks,
                derived=data.derived,
            )

            residual_image = self.make_image(res_data_BRF)
            data.derived['residual_data'] = res_data_BRF
            data.derived['residual_image'] = residual_image

        return dataset
    
    def correction_structures(self, dataset: Dataset):
        """Correction of weird structure inside the image"""
        correction_ranges = {
            'A11': ((165, 275), (300, 370)),
            'A12': ((7, 46),),
            'B11': ((70, 260),),
        }

        for data in dataset.items:
            r_corrected_values = data.raw_values
            corrected_values = r_corrected_values.copy()
            structure_mask = np.zeros(len(corrected_values), dtype=bool)
            ranges = correction_ranges.get(data.region, ())

            for start_x, end_x in ranges:
                for x_val in range(start_x, end_x):
                    sel_col = corrected_values['x'] == x_val
                    corrected_values['f'][sel_col] = 0.0
                    structure_mask[sel_col] = True

            data.derived['structure_corrected_values'] = corrected_values
            data.derived['structure_mask'] = structure_mask
            data.derived['structure_correction_applied'] = data.region in correction_ranges


        return dataset
    
    """
    ************ CCF *************
    """

    @staticmethod
    def cross_correlation(shift_spec: np.ndarray, template_spec: np.ndarray):
        shift_spec_norm = (shift_spec - np.mean(shift_spec)) / np.std(shift_spec)
        template_spec_norm = (template_spec - np.mean(template_spec)) / np.std(template_spec)
        return float(np.sum(shift_spec_norm * template_spec_norm))

    def cross_correlation_velocity_clump(self, wavelength: np.ndarray, spec_clump: np.ndarray, template_spec, vel_array: np.ndarray):
        ccf_array = np.zeros(len(vel_array))
        for i, vel in enumerate(vel_array):
            w_shift = self.doppler_shift(wavelength, float(vel))
            ccf_array[i] = self.cross_correlation(spec_clump, template_spec(w_shift))
        return ccf_array

    def ccf_image_separation_velocity(self, dataset: Dataset, star_spec: str):
        """
        Compute cross correlation between the residuals and the stellar spectrum (master or model), star_spec is 'master' or 'model'.
        If 'model', fit a spline of the stellar model
        """
        if len(dataset.items) == 0:
            return dataset

        if 'model' in star_spec and dataset.stellar_model_path is not None :
            stellar_model_path = dataset.stellar_model_path
            model_spec = np.load(stellar_model_path, allow_pickle=True)
            w = model_spec[:,0]
            f = model_spec[:,1]
            f = f/np.median(f)
            first_data = dataset.items[0]
            min_w, max_w = np.min(first_data.raw_values['w']), np.max(first_data.raw_values['w'])
            mask = (w >= min_w-0.1) & (w <= max_w+0.1) #get some points outside for ccf shift 
            w,f = w[mask], f[mask]
            knots = np.arange(w[30], w[-30], 0.015)
            spec_spl = sp.interpolate.LSQUnivariateSpline(w,f, knots)

            # Rotational broadening
            w_rot = np.linspace(np.min(w), np.max(w),len(w)) # evenly spaced wavelengths to perform rotational broadening
            f_for_rot = spec_spl(w_rot)
            f_rot = pyasl.fastRotBroad(w_rot, f_for_rot, 0.3, 7)
            f_rot= f_rot[15:-15]
            w_rot = w_rot[15:-15]
            knots_rot = np.arange(w[30], w[-30], 0.015)
            spec_spl = sp.interpolate.LSQUnivariateSpline(w_rot,f_rot, knots_rot)

            dataset.products['stellar_model_spec_spl'] = spec_spl
        else :
            spec_spl = dataset.products.get('master_spec_spl')

        w_telluric = np.asarray(dataset.products.get('w_telluric', np.array([])), dtype=float)
        #print(w_telluric)
        vmax = float(self.params.get('step3_vmax', 100.0))
        vmin = float(self.params.get('step3_vmin', -100.0))
        lenv = int(vmax - vmin + 1)
        vel_array = np.linspace(vmin, vmax, lenv)
        delta_w = float(self.params.get('step3_delta_w', 0.1)) # was 0.15, now is covering barycentric motion 
        dr = float(self.params.get('step3_dr', 2.0)) # why 2.0 ?
        delta = vmax / (sp.constants.c * 0.001)

        for j, data in enumerate(dataset.items):
            residual_data_BRF = data.derived.get('residual_data')
            psf_2d_spl = data.derived.get('psf_2d_spl')
            psf_spl = data.derived.get('psf_spl')
            if residual_data_BRF is None or (psf_2d_spl is None and psf_spl is None):
                continue
            
            values = residual_data_BRF.raw_values
            rmin = float(np.min(values['r']))
            rmax = float(np.max(values['r']))
            r_position = np.arange(rmin, rmax, 1.0)
            lenr = len(r_position)

            img_ccf_erf = np.full((lenr, lenv), np.nan, dtype=float)

            # barycorr_kms = self._barycentric_correction_kms(j)
            # shift_values = values.copy()
            # shift_values['w'] = self.doppler_shift(values['w'], -barycorr_kms)

            w = np.linspace(np.min(values['w']) * (1.0 + delta), np.max(values['w']) * (1.0 - delta), 3000) #safe grid for CCF, CHANGE FOR LARGER
            w_with_cut = np.asarray(w, dtype=float).copy()

            if 'model' in star_spec and dataset.stellar_model_path is not None :
                for w_peak in w_telluric:
                    w_with_cut = w_with_cut[np.abs(w_with_cut - w_peak) > delta_w] # mask to cover tellurics move due to barycentric correction
            else : 
                for w_peak in w_telluric:
                    w_with_cut = w_with_cut[np.abs(w_with_cut - w_peak) > delta*np.max(w)+delta_w] # mask to cover tellurics move due to cross-correlation shift of master spectrum
            # TEMPORARILY NO MASKING
            #w_with_cut = np.asarray(w, dtype=float).copy()
            print(w_telluric)
            for i, r_pos in enumerate(r_position):
                spec_clump_spl = self._clump_spectrum(data, values, r_pos, dr) # residual spectrum at given r_pos position, extrapolation at edges, to use on restricted range
                # Plot residuals spectrum
                """plt.plot(w_with_cut, spec_clump_spl(w_with_cut),'r.')
                plt.title(f'Residuals spectrum around r={r_pos}')
                plt.show()"""
                # Shift mask back to ERF
                #w_with_cut = self.doppler_shift(w_with_cut, barycorr_kms)
                residuals_with_cut = spec_clump_spl(w_with_cut)
                """if i==0:
                    plt.plot(w_with_cut, residuals_with_cut,'r.')
                    plt.show()"""
                """tellurics = np.load(f'C:/Users/alice/Documents/Stage Suède/plots/tellurics/{data.region}/{data.epoch_index}_tellurics.npy', allow_pickle=True)
                w_tell, tell_array = tellurics[:,0], tellurics[:,1]
                tell_spl = sp.interpolate.LSQUnivariateSpline(w_tell, tell_array, np.arange(w_tell[10], w_tell[-10], 0.015)) # Create spline for telluric transmission
                tell_array = tell_spl(w_with_cut)
                residuals_with_cut = residuals_with_cut / tell_array"""
                ccf_array = self.cross_correlation_velocity_clump(w_with_cut, residuals_with_cut, spec_spl, vel_array)
                img_ccf_erf[i, :] = ccf_array

            if psf_2d_spl is not None:
                ind_max = int(np.nanargmax(psf_2d_spl(values['r'], values['w'], grid=False)))
            else:
                ind_max = int(np.nanargmax(psf_spl(values['r'])))
            r_star = float(values['r'][ind_max])
            r_position_centered = r_position - r_star

            # Row-wise normalization with edge exclusion in velocity
            v_slice = img_ccf_erf[:, 10:-10] if lenv > 20 else img_ccf_erf
            row_min = np.nanmin(v_slice, axis=1)
            row_max = np.nanmax(v_slice, axis=1)
            denom = row_max - row_min
            valid_rows = np.isfinite(denom) & (denom > 0.0)

            for i in range(lenr):
                if valid_rows[i]:
                    img_ccf_erf[i, :] = (img_ccf_erf[i, :] - row_min[i]) / (denom[i] + 1e-12)

            data.derived['img_separation_velocity'] = img_ccf_erf
            data.derived['r_position_centered'] = r_position_centered
            data.derived['vel_array'] = vel_array

            if j == 0:
                dataset.products['img_separation_velocity'] = img_ccf_erf
                dataset.products['r_position_centered'] = r_position_centered
                dataset.products['vel_array'] = vel_array

        return dataset

    @staticmethod
    def fit_step(w, f, weight):
        edges = np.arange(w[10], w[-10], 0.015)
        idx = np.digitize(w, edges) - 1  # bin index 0..nbins-1
        nbins = len(edges) - 1
        num = np.zeros(nbins)
        den = np.zeros(nbins)
        for i in range(nbins):
            mask = idx == i
            if np.any(mask):
                num[i] = np.sum(weight[mask] * f[mask])
                den[i] = np.sum(weight[mask])
        vals = np.divide(num, den, out=np.zeros_like(num), where=den != 0)

        def step_fun(x_eval):
            x_eval = np.asarray(x_eval)
            idx_eval = np.digitize(x_eval, edges) - 1
            idx_eval = np.clip(idx_eval, 0, nbins - 1)
            return vals[idx_eval]
        return step_fun, vals

    def _clump_spectrum(self, data: Data, residual_values: np.ndarray, r_pos: float, dr: float):
        """
        Build spline of the residual spectrum at a position r_pos
        """
        ind = np.argsort(residual_values['w'])
        wdata_BRF = residual_values[ind]
        sel_dr = np.abs(wdata_BRF['r'] - r_pos) < dr
        res_data = wdata_BRF[sel_dr]

        step = float(self.params.get('step3_spec_resolution', 0.015))
        w_min = res_data['w'][10]
        w_max = res_data['w'][-10]
        knots = np.arange(w_min, w_max, step)

        with np.errstate(divide='ignore', invalid='ignore'):
            weight = (1.0 / res_data['e']) ** 2
        weight[~np.isfinite(weight)] = 0.0
        #spec_clump_spl = self.fit_step(res_data['w'], res_data['f'], weight)[0]
        spec_clump_spl = sp.interpolate.LSQUnivariateSpline(res_data['w'], res_data['f'], knots, weight, k=3)
        """plt.plot(res_data['w'], spec_clump_spl(res_data['w']),'r')
        plt.plot(res_data['w'],res_data['f'],'b.')
        plt.title(f'Residuals spectrum around r={r_pos}')
        plt.show()"""
        plotter = Plotter('save','C:/Users/alice/Documents/Stage Suède/plots/')
        plotter.send_residuals(data, residual_values, r_pos, dr)
        return spec_clump_spl
        
    @staticmethod
    def _step4_night_from_epoch(epoch_index: int | None):
        if epoch_index is None:
            return 'n1'
        idx = int(epoch_index)
        if idx in (1, 2, 3, 4, 5, 6, 7, 8, 9):
            return 'n1'
        if idx in (10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 27):
            return 'n2'
        if idx in (0, 20, 21, 22, 23, 24, 25, 26, 28, 29):
            return 'n3'
        if idx in (30, 31, 32, 33, 34, 35, 36, 37, 38, 39):
            return 'n4'
        return 'n1'

    @staticmethod
    def _step4_slit_from_region(region: str | None):
        if region is None or len(region) == 0:
            return 'UNK'
        first = str(region)[0].upper()
        if first in ('A', 'B'):
            return first
        return 'UNK'

    @staticmethod
    def _step4_resize_map(img: np.ndarray, target_shape: tuple[int, int]):
        img = np.nan_to_num(np.asarray(img, dtype=float), nan=0.0)
        nr_t, nv_t = target_shape
        nr, nv = img.shape
        if (nr, nv) == (nr_t, nv_t):
            return img

        r_src = np.linspace(0.0, 1.0, nr)
        r_dst = np.linspace(0.0, 1.0, nr_t)
        v_src = np.linspace(0.0, 1.0, nv)
        v_dst = np.linspace(0.0, 1.0, nv_t)

        tmp = np.empty((nr_t, nv), dtype=float)
        for j in range(nv):
            tmp[:, j] = np.interp(r_dst, r_src, img[:, j])

        out = np.empty((nr_t, nv_t), dtype=float)
        for i in range(nr_t):
            out[i, :] = np.interp(v_dst, v_src, tmp[i, :])

        return out

    def group_velocity_separation(self, dataset_A: Dataset, dataset_B: Dataset):
        valid_items = []
        for dataset in (dataset_A, dataset_B):
            for data in dataset.items:
                img = data.derived.get('img_separation_velocity')
                r_pos = data.derived.get('r_position_centered')
                vel = data.derived.get('vel_array')
                if img is None or r_pos is None or vel is None:
                    continue
                valid_items.append(data)

        if len(valid_items) == 0:
            return dataset_A, dataset_B

        ref_data = valid_items[0]
        ref_img = np.asarray(ref_data.derived['img_separation_velocity'], dtype=float)
        ref_shape = ref_img.shape
        r_ref = np.asarray(ref_data.derived['r_position_centered'], dtype=float)
        v_ref = np.asarray(ref_data.derived['vel_array'], dtype=float)

        combined_region = f"{dataset_A.region}_{dataset_B.region}"

        # Short runs can be sequentially mapped to nights 1..4 to mimic old scripts.
        sequential_night_map = {}
        if len(valid_items) <= 4:
            order = sorted(valid_items, key=lambda d: (d.epoch_index if d.epoch_index is not None else 10**9, d.file_id))
            for i, item in enumerate(order):
                sequential_night_map[item.file_id] = f'n{i+1}'

        night_slit_maps = {}
        for data in valid_items:
            img = np.asarray(data.derived['img_separation_velocity'], dtype=float)
            if img.shape != ref_shape:
                img = self._step4_resize_map(img, ref_shape)

            slit = self._step4_slit_from_region(data.region)
            if data.file_id in sequential_night_map:
                night = sequential_night_map[data.file_id]
            else:
                night = self._step4_night_from_epoch(data.epoch_index)

            night_slit_maps.setdefault(night, {})
            night_slit_maps[night].setdefault(slit, np.zeros(ref_shape, dtype=float))
            night_slit_maps[night][slit] += np.nan_to_num(img, nan=0.0)
        night_keys = sorted(night_slit_maps.keys())

        # 1) Add slits for each night
        night_combined = {}
        for night in night_keys:
            total = np.zeros(ref_shape, dtype=float)
            for slit_img in night_slit_maps[night].values():
                total += slit_img
            night_combined[night] = total

        # 2) Add nights
        all_nights = np.zeros(ref_shape, dtype=float)
        for night in night_keys:
            all_nights += night_combined[night]

        step4_products = {
            'step4_night_slit_maps': night_slit_maps,
            'step4_night_combined': night_combined,
            'step4_all_nights_combined': all_nights,
            'step4_r_position': r_ref,
            'step4_vel_array': v_ref,
            'step4_region': combined_region,
        }
        dataset_A.products.update(step4_products)
        dataset_B.products.update(step4_products)
        return dataset_A, dataset_B