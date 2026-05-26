#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""File and path helpers for the analysis pipeline."""

import os
import numpy as np
from data_model import Dataset, Data


class FileManager:
    """ Data extraction of the files

    Attributes:
        root: str             -- root of the data
        region: str           -- for example 'A12' corresponding to one spectral region
        nb_files: int         -- number of files to be analyzed from 000 to nb_files-1, must be > 0
        stellar_model_path : str  -- file path of the stellar model spectrum
        files: list[str]      -- list of file names to be loaded, for example ['AUMic000_A12.npy', 'AUMic001_A12.npy', ...]
    """
    def __init__(self, root:str, region:str, nb_files:int, stellar_model_path:str=None):
        self.root = root
        self.region = region
        self.nb_files = nb_files
        self.stellar_model_path = stellar_model_path
        self.files = ['AUMic000_' + self.region +'.npy']
        for i in range(1, self.nb_files):
            self.files.append('AUMic' + str(i).zfill(3) + '_' + self.region +'.npy')

    def load_data(self):
        dataset = Dataset(items=[], region=self.region, nb_files=self.nb_files, products={}, stellar_model_path=self.stellar_model_path)
        for file in self.files:
            file_path = os.path.join(self.root, file)
            values = np.load(file_path)
            data_object = Data(values, source_path=file_path, file_id=file, epoch_index=int(file[5:8]), region=self.region)
            dataset.items.append(data_object)
        return dataset