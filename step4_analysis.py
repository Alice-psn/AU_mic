#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:18:53 2024

@author: beatricecaccherano
"""

import pickle
import os
import numpy as np
import scipy as sp 
import time
import matplotlib.pyplot as plt 

import spectroscopy_functions as spec_fun
import plot_functions as plot_func

st = time.time()

SESSION = 1 #old analysis
#SESSION = 2 #new analysis

ngroup = "11"

if(SESSION==1):

    root = 'C:/Users/alice/Documents/Stage Suède/data/'
    root1 = 'C:/Users/alice/Documents/Stage Suède/data1/'
    root2 = 'C:/Users/alice/Documents/Stage Suède/data2/'
    root3 = 'C:/Users/alice/Documents/Stage Suède/data3/'
    root4 = 'C:/Users/alice/Documents/Stage Suède/data4/'
    
if(SESSION==2):
    root = '/home/beatricecaccherano/Master_Thesis_Analysis/data/'
    root1 = '/home/beatricecaccherano/Master_Thesis_Analysis/dataS1/'
    root2 = '/home/beatricecaccherano/Master_Thesis_Analysis/dataS2/'
    root3 = '/home/beatricecaccherano/Master_Thesis_Analysis/dataS3/'
    root4 = '/home/beatricecaccherano/Master_Thesis_Analysis/dataS4/'
    #root4 = '/home/beatricecaccherano/Master_Thesis_Analysis/dataS4_redshift/' #vmin=-10.0
    
    
file_name_A ="A" + ngroup
file_name_B ="B" + ngroup
"""
Creation of the list of files whcih you want analyse: 
    Write the name of the first file of the group, the group number (=group_num)
    and the group (=group).
"""

file_path_A = root4+'S4_'+file_name_A+ '.npy' 
print(file_path_A)
file_path_B = root4+'S4_'+file_name_B+ '.npy' 
print(file_path_B)

data_file_A = open(file_path_A, 'rb')
img_ccf_ERF_n1A, img_ccf_ERF_n2A, img_ccf_ERF_n3A, img_ccf_ERF_n4A  = pickle.load(data_file_A)
data_file_B = open(file_path_B, 'rb')
img_ccf_ERF_n1B, img_ccf_ERF_n2B, img_ccf_ERF_n3B, img_ccf_ERF_n4B  = pickle.load(data_file_B)


aumic_name_A = root + 'AUMic000_' + file_name_A + '.npy'
aumic_A = np.load(aumic_name_A)
rmin_A, rmax_A = -42.0, 42.0#np.min(aumic_A['r']) , np.max(aumic_A['r'])
r_position_A = np.linspace(rmin_A, rmax_A, 90)
lenr = len(r_position_A)
aumic_name_B = root + 'AUMic000_' + file_name_B + '.npy'
aumic_B = np.load(aumic_name_B)
rmin_B, rmax_B = -42.0, 42.0 #np.min(aumic_B['r']) , np.max(aumic_B['r'])
r_position_B = np.linspace(rmin_B, rmax_B, 90)
lenr = len(r_position_B)
vmax = 100.0
vmin = -100.0
lenv = int(vmax- vmin +1)
vel_array = np.linspace(vmin,vmax,lenv)


def collapse_epochs(stack, label):
    arr = np.asarray(stack)
    if arr.size == 0:
        return np.zeros((lenr, lenv))

    # Common case: numeric stack already stored as (n_epoch, nr, nv) or a single 2D map.
    if arr.dtype != object:
        if arr.ndim == 3:
            img = np.nansum(arr, axis=0)
        elif arr.ndim == 2:
            img = np.nan_to_num(arr, nan=0.0)
        else:
            return np.zeros((lenr, lenv))
    else:
        # Fallback for object stacks containing None/mixed elements.
        valid_planes = []
        for plane in arr.ravel():
            if plane is None:
                continue
            plane_arr = np.asarray(plane)
            if plane_arr.ndim != 2:
                continue
            valid_planes.append(np.nan_to_num(plane_arr, nan=0.0))

        if len(valid_planes) == 0:
            return np.zeros((lenr, lenv))

        img = np.sum(valid_planes, axis=0)
    return img


def resize_image(img, target_shape):
    """Resize a 2D image to target shape using separable linear interpolation."""
    img = np.nan_to_num(np.asarray(img, dtype=float), nan=0.0)
    nr_t, nv_t = target_shape
    nr, nv = img.shape
    if (nr, nv) == (nr_t, nv_t):
        return img

    r_src = np.linspace(0.0, 1.0, nr)
    r_dst = np.linspace(0.0, 1.0, nr_t)
    v_src = np.linspace(0.0, 1.0, nv)
    v_dst = np.linspace(0.0, 1.0, nv_t)

    tmp = np.empty((nr_t, nv))
    for j in range(nv):
        tmp[:, j] = np.interp(r_dst, r_src, img[:, j])

    out = np.empty((nr_t, nv_t))
    for i in range(nr_t):
        out[i, :] = np.interp(v_dst, v_src, tmp[i, :])

    return out


def add_images(img_ref, img_new, label):
    """Add two CCF maps; resample img_new to img_ref shape when needed."""
    img_ref = np.nan_to_num(np.asarray(img_ref, dtype=float), nan=0.0)
    img_new = np.nan_to_num(np.asarray(img_new, dtype=float), nan=0.0)
    if img_ref.shape != img_new.shape:
        img_new = resize_image(img_new, img_ref.shape)
    return img_ref + img_new

"""SLIT A"""
img_ccf_n1A = collapse_epochs(img_ccf_ERF_n1A, "SLIT A night 1")
# Quicklook of the CCF image in velocity/separation coordinates.
plot_func.make_image_v_r_epochs(img_ccf_n1A, r_position_A, vel_array,
                                np.max(img_ccf_n1A), np.min(img_ccf_n1A),
                                "CCF image of the night 15/08/2023 + SLIT A")
max_img_ccf_n1A = 10.0#np.max(img_ccf_n1A)
min_img_ccf_n1A = 0.0 #np.min(img_ccf_n1A)
title1 = "Radial Velocity - Separtion image || Night 15/08/2023 + SLIT A"
plot_func.make_image_v_r_epochs(img_ccf_n1A, r_position_A, vel_array, max_img_ccf_n1A, min_img_ccf_n1A, title1)
img_ccf_n2A = collapse_epochs(img_ccf_ERF_n2A, "SLIT A night 2")
max_img_ccf_n2A = 9.0#np.max(img_ccf_n2A)
min_img_ccf_n2A = 0.0#np.min(img_ccf_n2A)
title2 = "Radial Velocity - Separtion image || Night 23/08/2023 + SLIT A"
plot_func.make_image_v_r_epochs(img_ccf_n2A, r_position_A, vel_array, max_img_ccf_n2A, min_img_ccf_n2A,title2)
img_ccf_n3A = collapse_epochs(img_ccf_ERF_n3A, "SLIT A night 3")
max_img_ccf_n3A = 11.0#np.max(img_ccf_n3A)
min_img_ccf_n3A = 0.0#np.min(img_ccf_n3A)
title3 = "Radial Velocity - Separtion image || Night 26/08/2023 + SLIT A"
plot_func.make_image_v_r_epochs(img_ccf_n3A, r_position_A, vel_array, max_img_ccf_n3A, min_img_ccf_n3A, title3)
img_ccf_n4A = collapse_epochs(img_ccf_ERF_n4A, "SLIT A night 4")
max_img_ccf_n4A = 10.0#np.max(img_ccf_n4A)
min_img_ccf_n4A = 0.0#np.min(img_ccf_n4A)
title4 = "Radial Velocity - Separtion image || Night 27/08/2023 + SLIT A"
plot_func.make_image_v_r_epochs(img_ccf_n4A, r_position_A, vel_array, max_img_ccf_n4A, min_img_ccf_n4A,title4)

"""SLIT B"""
img_ccf_n1B = collapse_epochs(img_ccf_ERF_n1B, "SLIT B night 1")
max_img_ccf_n1B = 10.0#np.max(img_ccf_n1B)
min_img_ccf_n1B = 0.0 
title1 = "Radial Velocity - Separtion image || Night 15/08/2023 + SLIT B"
plot_func.make_image_v_r_epochs(img_ccf_n1B, r_position_B, vel_array,  max_img_ccf_n1B, min_img_ccf_n1B, title1)
img_ccf_n2B = collapse_epochs(img_ccf_ERF_n2B, "SLIT B night 2")
max_img_ccf_n2B = np.max(img_ccf_n2B)
min_img_ccf_n2B = 0.0
title2 = "Radial Velocity - Separtion image || Night 23/08/2023 + SLIT B"
plot_func.make_image_v_r_epochs(img_ccf_n2B, r_position_B, vel_array,  max_img_ccf_n2B, min_img_ccf_n2B, title2)
img_ccf_n3B = collapse_epochs(img_ccf_ERF_n3B, "SLIT B night 3")
max_img_ccf_n3B = 11.0#np.max(img_ccf_n3B)
min_img_ccf_n3B = 0.0
title3 = "Radial Velocity - Separtion image || Night 26/08/2023 + SLIT B"
plot_func.make_image_v_r_epochs(img_ccf_n3B, r_position_B, vel_array, max_img_ccf_n3B, min_img_ccf_n3B, title3)
img_ccf_n4B = collapse_epochs(img_ccf_ERF_n4B, "SLIT B night 4")
max_img_ccf_n4B = 10.0#np.max(img_ccf_n4B)
min_img_ccf_n4B = 0.0
title4 = "Radial Velocity - Separtion image || Night 27/08/2023 + SLIT B"
plot_func.make_image_v_r_epochs(img_ccf_n4B, r_position_B, vel_array, max_img_ccf_n4B, min_img_ccf_n4B,title4)

"""SLIT A+B"""
img_ccf_n1AB = add_images(collapse_epochs(img_ccf_ERF_n1A, "SLIT A night 1"),
                          collapse_epochs(img_ccf_ERF_n1B, "SLIT B night 1"),
                          "SLIT A+B night 1")
max_img_ccf_n1AB = 20.0#np.max(img_ccf_n1AB)
min_img_ccf_n1AB = 0.0#np.min(img_ccf_n1AB)
title1 = "Radial Velocity - Separtion image || Night 15/08/2023 + SLIT A+B"
plot_func.make_image_v_r_epochs(img_ccf_n1AB, r_position_B, vel_array,  max_img_ccf_n1AB, min_img_ccf_n1AB, title1)
img_ccf_n2AB = add_images(collapse_epochs(img_ccf_ERF_n2A, "SLIT A night 2"),
                          collapse_epochs(img_ccf_ERF_n2B, "SLIT B night 2"),
                          "SLIT A+B night 2")
max_img_ccf_n2AB = 20.0#np.max(img_ccf_n2AB)
min_img_ccf_n2AB = 0.0#np.min(img_ccf_n2AB)
title2 = "Radial Velocity - Separtion image || Night 23/08/2023 + SLIT A+B"
plot_func.make_image_v_r_epochs(img_ccf_n2AB, r_position_B, vel_array,  max_img_ccf_n2AB, min_img_ccf_n2AB,title2)
img_ccf_n3AB = add_images(collapse_epochs(img_ccf_ERF_n3A, "SLIT A night 3"),
                          collapse_epochs(img_ccf_ERF_n3B, "SLIT B night 3"),
                          "SLIT A+B night 3")
max_img_ccf_n3AB = 22.0#np.max(img_ccf_n3AB)
min_img_ccf_n3AB = 0.0#np.min(img_ccf_n3AB)
title3 = "Radial Velocity - Separtion image || Night 26/08/2023 + SLIT A+B"
plot_func.make_image_v_r_epochs(img_ccf_n3AB, r_position_B, vel_array,  max_img_ccf_n3AB, min_img_ccf_n3AB, title3)
img_ccf_n4AB = add_images(collapse_epochs(img_ccf_ERF_n4A, "SLIT A night 4"),
                          collapse_epochs(img_ccf_ERF_n4B, "SLIT B night 4"),
                          "SLIT A+B night 4")
max_img_ccf_n4AB = 20.0#np.max(img_ccf_n4AB)
min_img_ccf_n4AB = 0.0#np.min(img_ccf_n4AB)
title4 = "Radial Velocity - Separtion image || Night 27/08/2023 + SLIT A+B"
plot_func.make_image_v_r_epochs(img_ccf_n4AB, r_position_B, vel_array,  max_img_ccf_n4AB, min_img_ccf_n4AB, title4)


"""
#SLIT A-B
img_ccf_n1A_B = img_ccf_ERF_n1A.sum(axis = 0) - img_ccf_ERF_n1B.sum(axis = 0)
max_img_ccf_n1A_B = 20.0
min_img_ccf_n1A_B = 0.0 
title1 = "Radial Velocity - Separtion image || Night 15/08/2023 + SLIT A-B"
plot_func.make_image_v_r_epochs(img_ccf_n1A_B, r_position_B, vel_array,  max_img_ccf_n1A_B, min_img_ccf_n1A_B, title1)
img_ccf_n2A_B = img_ccf_ERF_n2A.sum(axis = 0) - img_ccf_ERF_n2B.sum(axis = 0)
max_img_ccf_n2A_B = 20.0
min_img_ccf_n2A_B = 0.0
title2 = "Radial Velocity - Separtion image || Night 23/08/2023 + SLIT A-B"
plot_func.make_image_v_r_epochs(img_ccf_n2A_B, r_position_B, vel_array,max_img_ccf_n2A_B, min_img_ccf_n2A_B,title2)
img_ccf_n3A_B = img_ccf_ERF_n3A.sum(axis = 0) - img_ccf_ERF_n3B.sum(axis = 0)
max_img_ccf_n3A_B = 22.0
min_img_ccf_n3A_B = 0.0
title3 = "Radial Velocity - Separtion image || Night 26/08/2023 + SLIT A-B"
plot_func.make_image_v_r_epochs(img_ccf_n3A_B, r_position_B, vel_array,max_img_ccf_n3A_B, min_img_ccf_n3A_B, title3)
img_ccf_n4A_B = img_ccf_ERF_n4A.sum(axis = 0) - img_ccf_ERF_n4B.sum(axis = 0)
max_img_ccf_n4A_B = 20.0
min_img_ccf_n4A_B = 0.0
title4 = "Radial Velocity - Separtion image || Night 27/08/2023 + SLIT A-B"
plot_func.make_image_v_r_epochs(img_ccf_n4A_B, r_position_B, vel_array,max_img_ccf_n4A_B, min_img_ccf_n4A_B, title4)
"""


"""Good epochs"""
img_ccf_good = collapse_epochs(img_ccf_ERF_n1A, "SLIT A night 1")
img_ccf_good = add_images(img_ccf_good, collapse_epochs(img_ccf_ERF_n1B, "SLIT B night 1"), "GOOD epochs + B1")
img_ccf_good = add_images(img_ccf_good, collapse_epochs(img_ccf_ERF_n2A, "SLIT A night 2"), "GOOD epochs + A2")
img_ccf_good = add_images(img_ccf_good, collapse_epochs(img_ccf_ERF_n2B, "SLIT B night 2"), "GOOD epochs + B2")

max_img_ccf_good = 38.0#np.max(img_ccf_good)
min_img_ccf_good = 0.0#np.min(img_ccf_good)
title = "Radial Velocity - Separtion image || GOOD epochs"
plot_func.make_image_v_r_epochs(img_ccf_good, r_position_B, vel_array,  max_img_ccf_good, min_img_ccf_good, title)

"""all the epochs"""
img_ccf = collapse_epochs(img_ccf_ERF_n1A, "SLIT A night 1")
img_ccf = add_images(img_ccf, collapse_epochs(img_ccf_ERF_n1B, "SLIT B night 1"), "ALL epochs + B1")
img_ccf = add_images(img_ccf, collapse_epochs(img_ccf_ERF_n2A, "SLIT A night 2"), "ALL epochs + A2")
img_ccf = add_images(img_ccf, collapse_epochs(img_ccf_ERF_n2B, "SLIT B night 2"), "ALL epochs + B2")
img_ccf = add_images(img_ccf, collapse_epochs(img_ccf_ERF_n3A, "SLIT A night 3"), "ALL epochs + A3")
img_ccf = add_images(img_ccf, collapse_epochs(img_ccf_ERF_n3B, "SLIT B night 3"), "ALL epochs + B3")
img_ccf = add_images(img_ccf, collapse_epochs(img_ccf_ERF_n4A, "SLIT A night 4"), "ALL epochs + A4")
img_ccf = add_images(img_ccf, collapse_epochs(img_ccf_ERF_n4B, "SLIT B night 4"), "ALL epochs + B4")

max_img_ccf = 38.0#np.max(img_ccf)
min_img_ccf = 0.0#np.min(img_ccf)
title = "Radial Velocity - Separtion image || all epochs"
plot_func.make_image_v_r_epochs(img_ccf, r_position_B, vel_array,  max_img_ccf, min_img_ccf, title)

data_file_A.close()