#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core data containers for the AU Mic spectroscopy pipeline.

This module is intentionally lightweight at first: it defines the domain objects
that will gradually absorb the current procedural helpers from
spectroscopy_functions.py.
"""

import numpy as np


class Data:
    def __init__(self, raw_values: np.ndarray, source_path: str = None, file_id: str = None, epoch_index: int = None, region: str = None, masks: dict = None, derived: dict = None):
        """ Data container for the AU Mic pipeline.

        Attributes:
            raw_values: np.ndarray -- the raw data values
            source_path: str   -- the path to the source file
            file_id: str       -- the unique identifier for the file (ex: AUMic000_A12.npy)
            epoch_index: int   -- the index of the epoch (0 to 39)
            region: str        -- the region of interest (ex: A12)
            masks: dict        -- the segmentation masks (example keys : valid, outlier, sigma_clip)
            derived: dict      -- the derived quantities
            stats: dict        -- the statistical measures
        """
        self.raw_values= raw_values
        self.clean_values = np.array([])
        self.clipped_values = np.array([])
        self.source_path = source_path
        self.file_id = file_id
        self.epoch_index = epoch_index
        self.region = region
        self.masks = {} if masks is None else dict(masks)
        self.derived = {} if derived is None else dict(derived)
        self.stats = {}
        self.stats = self.calculate_stats()

    def sort(self):
        order = np.argsort(self.raw_values['x'])
        return self.raw_values[order]

    def copy(self):
        return Data(
            raw_values=self.raw_values.copy(),
            source_path=self.source_path,
            file_id=self.file_id,
            epoch_index=self.epoch_index,
            region=self.region,
            masks=self.masks,
            derived=self.derived,
        )
    
    def calculate_stats(self):
        dict_stats = {}
        dict_stats['fmin'] = np.min(self.raw_values['f'])
        dict_stats['fmax'] = np.max(self.raw_values['f'])
        dict_stats['fmean'] = np.mean(self.raw_values['f'])
        dict_stats['xmin'] = np.min(self.raw_values['x'])
        dict_stats['xmax'] = np.max(self.raw_values['x'])
        dict_stats['xmean'] = np.mean(self.raw_values['x'])
        dict_stats['ymin'] = np.min(self.raw_values['y'])
        dict_stats['ymax'] = np.max(self.raw_values['y'])
        dict_stats['ymean'] = np.mean(self.raw_values['y'])
        return dict_stats
    
    def see_derived(self):
        for key, value in self.derived.items():
            print(f"{key}: {value}")
    


class Dataset:
    def __init__(self, items: list[Data], region: str, nb_files: int, products: dict):
        """ Dataset container for the AU Mic pipeline.

        Attributes:
            items: list[Data]    -- the list of Data objects
            region: str          -- the region of interest (ex: A12)
            nb_files: int        -- the number of files in the dataset
            products: dict       -- the dictionary of derived products (Shared outputs built from multiple files (stacked spectra, master spectrum, CCF maps).)
        """
        self.items = items
        self.region = region
        self.nb_files = nb_files
        self.products = products

    def build_step1_2_dataset(self):
        step1_2_items = []
        for src in self.items:
            corrected = src.derived.get('r_corrected_values')
            if corrected is None:
                continue
            new_data = Data(raw_values=corrected.copy(), source_path=src.source_path, 
                            file_id=src.file_id.replace('.npy', '_S1_2.npy'), epoch_index=src.epoch_index, 
                            region=src.region, masks={}, derived={})

            # Reuse the final step1 spectrum as the rough-spectrum reference for PSF.
            if src.derived.get('spec_spl') is not None:
                new_data.derived['rss_interp'] = src.derived['spec_spl']

            # Keep access to the original detector frame values for residual plots.
            new_data.derived['raw_values_original'] = src.raw_values.copy()
            new_data.derived['clipped_values'] = new_data.raw_values
            new_data.derived['parent_file_id'] = src.file_id
            step1_2_items.append(new_data)

        return Dataset(items=step1_2_items, region=self.region, nb_files=len(step1_2_items), products={})
