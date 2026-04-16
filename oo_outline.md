# OO Refactor Outline

This folder keeps the legacy scripts alongside a new object-oriented scaffold.

## Proposed modules

- `data_model.py`: raw observation table, image creation, table selection, persistence helpers.
- `spectrum.py`: 1D spectrum objects, interpolation, normalization, extracted stellar and clump spectra.
- `psf.py`: PSF fitting and evaluation.
- `ccf.py`: Doppler shift and cross-correlation.
- `plotting.py`: all figures and plotting helpers.
- `file_manager.py`: loading and saving files, path conventions.
- `pipeline.py`: orchestration for steps 1 to 5.

## Migration strategy

1. Keep the old scripts untouched for comparison.
2. Move one capability at a time into the new modules.
3. Make the step scripts thin wrappers around the pipeline.
4. Preserve the old function names as compatibility shims if needed.


Observation or DataFrame-like class: wraps one loaded .npy table, metadata, and basic selections.

- Responsibilities: load/save a file, expose x, y, r, w, f, e, build images, slice by wavelength or slit position, remove outliers.
- This would absorb things like make_image, clean_outliers, dw_selection, and the file I/O currently scattered in step1_analysis.py.

Spectrum1D class: represents a 1D spectrum with wavelength, flux, error, and optional spline fit.

- Responsibilities: rough extraction, interpolation, normalization, plotting-friendly access, tellur
- This would naturally own logic from rough_spectrum, fit_spec, fit_spec_clump, spectrum, and related functions.

PSFModel class: represents the spatial profile across r.

- Responsibilities: fit rough PSF, fit refined PSF, evaluate the spline, refit after clipping, keep knots/weights/configuration together.
- This is where fit_psf, fit_psf_rough, and PSF evaluation logic should live.

CCFAnalyzer class: handles cross-correlation and radial velocity estimation.

- Responsibilities: Doppler shift, compute CCF, scan velocity grids, derive best velocity, store CCF curves.
- This would gather doppler_shift, cross_correlation, cross_correlation_velocity, and the clump-related CCF variants.

ClumpModel or FeatureModel class: optional if clump analysis is central.

- Responsibilities: build clump spectrum, simulate clumps, shift templates, compare to stellar spectrum.
- This would hold the clump-specific pieces currently mixed into the spectroscopy utilities.

Plotter class: strictly visualization, no physics logic.

- Responsibilities: plot image, spectrum, PSF, CCF, grouped spectra, velocity maps, master spectrum, etc.
- This would replace plot_functions.py as a plotting service, ideally with methods like plot_spectrum, plot_psf, plot_ccf, plot_image, and so on.

AnalysisPipeline or StepRunner class: orchestrates steps 1 to 5.

- Responsibilities: define the sequence of operations, pass intermediate objects forward, manage outputs and checkpoints.
- The step scripts would become small methods or stage objects like run_step1, run_step2, etc.

DataRepository or FileManager class: handles filenames, folders, serialization, and loading all epochs/groups.

- Responsibilities: load .npy, save pickle results, manage input/output paths, keep naming conventions out of analysis code.
- I would avoid a vague File class unless it actually stores metadata and path rules; otherwise a repository/service is cleaner.