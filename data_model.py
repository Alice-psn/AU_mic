#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Core data containers for the AU Mic spectroscopy pipeline.

This module is intentionally lightweight at first: it defines the domain objects
that will gradually absorb the current procedural helpers from
spectroscopy_functions.py.
"""

import numpy as np


class Data:
    def __init__(self, values: np.ndarray, source_path: str=None, file_id:str=None, epoch_index:int=None, region:str=None, masks:dict={}):
        """ Data container for the AU Mic pipeline.

        Attributes:
            values: np.ndarray -- the raw data values, as loaded from the .npy files
            source_path: str   -- the path to the source file
            file_id: str       -- the unique identifier for the file (ex: AUMic000_A12.npy)
            epoch_index: int   -- the index of the epoch (0 to 39)
            region: str        -- the region of interest (ex: A12)
            masks: dict        -- the segmentation masks (example keys : valid, outlier, sigma_clip)
            derived: dict      -- the derived quantities
            stats: dict        -- the statistical measures
        """
        self.values= values
        self.source_path = source_path
        self.file_id = file_id
        self.epoch_index = epoch_index
        self.region = region
        self.masks = masks
        self.stats = {}
        self.stats = self.calculate_stats()

    def sort(self):
        pass
    
    def calculate_stats(self):
        dict_stats = {}
        dict_stats['fmin'] = np.min(self.values['f'])
        dict_stats['fmax'] = np.max(self.values['f'])
        dict_stats['fmean'] = np.mean(self.values['f'])
        dict_stats['xmin'] = np.min(self.values['x'])
        dict_stats['xmax'] = np.max(self.values['x'])
        dict_stats['xmean'] = np.mean(self.values['x'])
        dict_stats['ymin'] = np.min(self.values['y'])
        dict_stats['ymax'] = np.max(self.values['y'])
        dict_stats['ymean'] = np.mean(self.values['y'])
        return dict_stats



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


class Observation:
    pass


class AnalysisResult:
    pass
