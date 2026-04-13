#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 09:55:24 2024

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

d_star = 9.79 #pc -- Distance of AU Mic star
M_star = 0.4*const.M_sun #AU Mic Mass



"""Definition of the Maximum Keplerian velocity LIMIT:"""
d_clumps_AU = np.linspace(0.0, 60.0, 90)#AU
d_clumps = d_clumps_AU*1.496e+11*u.m
R0 = 10.0*1.496e+11*u.m
kepl_vel = np.sqrt((M_star*const.G)/(np.abs(d_clumps)))*0.001
kepl_vel_0 = np.sqrt((M_star*const.G)/(R0))*0.001 #vel at the point of realesing 
b_1 = 0.7
vel_limit_1 = kepl_vel_0*np.sqrt((2.0*b_1)-1.0)
vel_1 = np.sqrt((M_star*const.G)*((2.0-2.0*b_1)/d_clumps + (2*b_1-1)/R0))*0.001 
b_2 = 1.5
vel_limit_2 = kepl_vel_0*np.sqrt((2.0*b_2)-1.0)
vel_2 = np.sqrt((M_star*const.G)*((2.0-2.0*b_2)/d_clumps + (2*b_2-1)/R0))*0.001 
b_3 = 5.6
vel_limit_3 = kepl_vel_0*np.sqrt((2.0*b_3)-1.0)
vel_3= np.sqrt((M_star*const.G)*((2.0-2.0*b_3)/d_clumps + (2*b_3-1)/R0))*0.001    
plot_func.plot_Keplerian_Limit(d_clumps_AU, kepl_vel, vel_limit_1.value, vel_1, vel_limit_2.value, vel_2, vel_limit_3.value, vel_3)

"""Maximum Limit of Keplerian velocity of a CIRCULAR EDGE-ON RING orbiting the star:""" 
#Assuming clockwise of the ring
title = "Maximum Limit of Keplerian velocity of a CIRCULAR EDGE-ON RING"
d_clumps_arcsec = np.linspace(0.0, 1.5, 90)#arcsec
d_clumps = d_clumps_arcsec*4.84814e-6*d_star*30856776000000000*u.m
kepl_vel_R = - np.sqrt((M_star*const.G)/(np.abs(d_clumps)))*0.001    
kepl_vel_L = np.sqrt((M_star*const.G)/(np.abs(d_clumps)))*0.001    
plot_func.plot_circular_ring(d_clumps_arcsec, kepl_vel_R, kepl_vel_L, title )   

"""ORBITING EDGE-ON RING :""" 
#assuming counterclockwise rotation
title = 'Circular ring of dust orbiting a star, seen edge on'
d_clumps_arcsec = np.linspace(0.0, 1.5, 300)#arcsec
angle = np.linspace(0.0, 2.0*np.pi, 300)
vr = 25.0*np.sin(angle)
dx = 1.5*np.sin(angle)
plot_func.plot_ring(dx, vr, title)

"""ORBITING EDGE-ON RING + DOPPLER SHIFT:""" 
#assuming counterclockwise rotation
title = 'Circular ring of dust orbiting a star, seen edge on + Doppler shift'
d_clumps_arcsec = np.linspace(0.0, 1.5, 300)#arcsec
angle = np.linspace(0.0, 2.0*np.pi, 300)
vr = 25.0*(1.0+np.sin(angle))
dx = 1.5*np.sin(angle)
plot_func.plot_ring(dx, vr, title)

"""EXPANDING OUTWARD EDGE-ON RING :""" 
title = 'EXPANDING OUTWARD EDGE-ON RING'
d_clumps_arcsec = np.linspace(0.0, 1.5, 300)#arcsec
angle = np.linspace(0.0, 2.0*np.pi, 300)
vr = -25.0*np.cos(angle)
dx = 1.5*np.sin(angle)
plot_func.plot_ring(dx, vr, title)
 
"""EXPANDING OUTWARD EDGE-ON RING + DOPPLER SHIFT :""" 
title = 'EXPANDING OUTWARD EDGE-ON RING + DOPPLER SHIFT '
d_clumps_arcsec = np.linspace(-1.5, 1.5, 300)#arcsec
phi = np.linspace(0.0, 2.0*np.pi, 300)
vr = 25.0*(1.0+np.cos(phi))
dx = 1.5*np.sin(phi)
plot_func.plot_ring(dx, vr, title)  
 

"""ORBITING * EXPANDING OUTWARD EDGE-ON RING:"""
title = 'ORBITING + EXPANDING OUTWARD EDGE-ON RING'
d_clumps_arcsec = np.linspace(0.0, 1.5, 300)#arcsec
angle = np.linspace(0.0, 2.0*np.pi, 300)
vr = 25.0*np.cos(angle) + 25.0*np.sin(angle)
dx = 1.5*np.sin(angle)
plot_func.plot_ring(dx, vr, title)

"""ORBITING * EXPANDING OUTWARD EDGE-ON RING + DOPPLER SHIFT:"""
title = 'ORBITING + EXPANDING OUTWARD EDGE-ON RING'
d_clumps_arcsec = np.linspace(0.0, 1.5, 300)#arcsec
angle = np.linspace(0.0, 2.0*np.pi, 300)
vr = 25.0*(1.0+np.cos(angle)) + 25.0*np.sin(angle)
dx = 1.5*np.sin(angle)
plot_func.plot_ring(dx, vr, title)

"""ORBITING * EXPANDING OUTWARD EDGE-ON RING + DOPPLER SHIFT:"""
title = 'ORBITING + EXPANDING OUTWARD EDGE-ON RING'
d_clumps_arcsec = np.linspace(0.0, 1.5, 300)#arcsec
angle = np.linspace(0.0, 2.0*np.pi, 300)
vr1 = 25.0*(1.0+np.cos(angle)) + 5.0*np.sin(angle)
vr2 = 25.0*(1.0+np.cos(angle)) + 25.0*np.sin(angle)
vr3 = 5.0*(1.0+np.cos(angle)) + 25.0*np.sin(angle)
dx = 1.5*np.sin(angle)
plot_func.plot_ring_multiple(dx, vr1, vr2, vr3, title)

#EXPANDING+KEPLERIAN OUTWARD EDGE-ON RING :
#Assuming counterclockwise of the ring
title = "EXPANDING+KEPLERIAN OUTWARD EDGE-ON RING"
R0 = 0.2*4.84814e-6*d_star*30856776000000000*u.m # distance at which the star push eceeds the gravitational force
bb = 2.5 #Beta value of the model. It quantifies the effect of others forces which actin against the gravitational one
d_clumps_arcsec = np.linspace(0.0, 1.5, 90)#arcsec
d_clumps = d_clumps_arcsec*4.84814e-6*d_star*30856776000000000*u.m
exp_vel_R_ccw = np.sqrt((M_star*const.G)*((2.0-2.0*bb)/d_clumps + (2*bb - 1.0)/R0))*0.001  
print(exp_vel_R_ccw)  
exp_vel_B_ccw = - np.sqrt((M_star*const.G)*((2.0-2.0*bb)/d_clumps + (2*bb - 1.0)/R0))*0.001    
print(exp_vel_B_ccw)
plot_func.plot_expanding_klep_ring(d_clumps_arcsec, exp_vel_R_ccw, exp_vel_B_ccw, title)   
print(exp_vel_R_ccw)     
    


