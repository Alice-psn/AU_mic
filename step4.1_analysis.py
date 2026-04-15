#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 15:02:37 2024

@author: beatricecaccherano
"""


import pickle
import os
import numpy as np
import scipy as sp 
import time
import matplotlib.pyplot as plt 
from astropy import constants as const
from astropy import units as u


import spectroscopy_functions as spec_fun
import plot_functions as plot_func

st = time.time()

SESSION = 1 #old analysis
#SESSION = 2 #new analysis


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

#ngroups = ["11", "12", "13", "14", "15", "21", "22", "23", "25", "31", "32", "33", "34", "35"]
#ngroups = ["11", "12", "13", "23", "25", "31", "35"]
ngroups = ["27"]

d_star = 9.79 #pc -- Distance of AU Mic star
M_star = 0.4*const.M_sun #AU Mic Mass
print(M_star)
print(const.G)

final_img = 0.0
for j in range(len(ngroups)):
    file_name_A ="A" + ngroups[j]
    file_name_B ="B" + ngroups[j]
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
    
    
    img_ccf = img_ccf_ERF_n1A.sum(axis = 0) + img_ccf_ERF_n1B.sum(axis = 0) +  img_ccf_ERF_n2A.sum(axis = 0) + img_ccf_ERF_n2B.sum(axis = 0) 
    + img_ccf_ERF_n3A.sum(axis = 0) + img_ccf_ERF_n3B.sum(axis = 0) +  img_ccf_ERF_n4A.sum(axis = 0) + img_ccf_ERF_n4B.sum(axis = 0)

    max_img_ccf = 40.0#np.max(img_ccf)
    min_img_ccf = 0.0#np.min(img_ccf)
    title = "Radial Velocity - Separtion image || all epochs"
    plot_func.make_image_v_r_epochs(img_ccf, r_position_B, vel_array,  max_img_ccf, min_img_ccf, title)
    
    final_img += img_ccf
    
    
max_final_img = 150#400/200/160
min_final_img = 40.0#np.min(img_ccf)    
title = "Radial Velocity - Separtion image "
plot_func.make_image_v_r_epochs(final_img, r_position_B, vel_array,  max_final_img, min_final_img, title)
    















