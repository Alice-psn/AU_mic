#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""High-level orchestration layer for steps 1 to 5.

This is only a scaffold. The goal is to move the legacy scripts toward a
small number of explicit workflow stages that consume and produce domain
objects.
"""
import numpy as np

from file_manager import FileManager
from plotting import Plotter

def main(): #plot all spectra related to a region
    root = 'C:/Users/alice/Documents/Stage Suède/data/'
    region = 'A12'
    file_manager = FileManager(root,region,nb_files=4)
    dataset = file_manager.load_data() #A12 Dataset from 000 to 004=nb_files
    plotter = Plotter(plot_bool=True)
    plotter.plot_initial_data(dataset)


class AnalysisPipeline:
    pass

if __name__ == "__main__":
    main()