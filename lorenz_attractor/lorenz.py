#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 15:16:57 2021

@author: pietro
"""

from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

def lorenz(v,t):

    a1=10;
    a2=28;
    a3=2.667;
    
    x=v[0];
    y=v[1];
    z=v[2];
    
    dxdt= a1*(y-x)
    dydt= a2*x-y-x*z
    dzdt= x*y-a3*z
    return[dxdt,dydt,dzdt]

def solvelorenz(x0,t):
    # Set initial values
    x0= x0
    t = np.linspace(0,t,200*t)
    
    #solve system
    u= odeint(lorenz,x0,t)
    
    x=u[:,0]
    y=u[:,1]
    z=u[:,2]
    return x,y,z

t=30;
x0=[0., 2., 3.]
x,y,z=solvelorenz(x0,t)

# Plot
ax = plt.figure().add_subplot(projection='3d')

ax.plot(x, y, z, lw=0.5)
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("Lorenz Attractor")

plt.show()
