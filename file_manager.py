#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""File and path helpers for the analysis pipeline."""

import os
import numpy as np
from data_model import Dataset, Data


class FileManager:
    """ Data extraction of the files

    Attributes:
        root: str        -- root of the data
        region: str      -- for example 'A12' corresponding to one spectral region
        nb_files: int    -- number of files to be analyzed from 000 to nb_files-1, default value = 39
        files: list[str] -- list of file names to be loaded, for example ['AUMic000_A12.npy', 'AUMic001_A12.npy', ...]
    """
    def __init__(self, root:str, region:str, nb_files:int=39):
        self.root = root
        self.region = region #A12
        self.nb_files = nb_files #4
        self.files = ['AUMic000_' + self.region +'.npy'] 
        for i in range(1, self.nb_files):
            self.files.append('AUMic' + str(i).zfill(3) + '_' + self.region +'.npy')

    def load_data(self):
        dataset = Dataset(items=[], region=self.region, nb_files=self.nb_files, products={})
        for file in self.files:
            file_path = os.path.join(self.root, file)
            values = np.load(file_path)
            data_object = Data(values, source_path=file_path, file_id=file, epoch_index=int(file[5:8]), region=self.region)
            dataset.items.append(data_object)
        return dataset

    def save_results(self, results, filename):
        pass
