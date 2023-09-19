#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 13:01:42 2022

@author: pietro
"""
import lorenz
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

fig = plt.figure(figsize=(7, 8), dpi=120)
ax = fig.add_subplot(projection='3d')
t=1

#Initial conditions for lorenz's systems
n=3
box_xyz=[-40.,40.,-40,40.,-10.,50.]
x=np.arange(box_xyz[0],box_xyz[1],(box_xyz[1]-box_xyz[0])/n)
y=np.arange(box_xyz[2],box_xyz[3],(box_xyz[3]-box_xyz[2])/n)
z=np.arange(box_xyz[4],box_xyz[5],(box_xyz[5]-box_xyz[4])/n)

#Solve systems
print("Solving systems")
for i in range(n**3):
    stx = "x{0},y{0},z{0}".format(i)
    x_0 = [x[i//n**2],y[i%n**2//n],z[i%n**2%n]]
    exec(stx + "= lorenz.solvelorenz(x_0,t)")
    std = "data{0}".format(i)
    exec(std+"= np.array(["+stx+"])")
    ii="{0}".format(i)
    exec("line" + ii+", =ax.plot(data" + ii+"[0, 0:1], data"+ii+"[1, 0:1], data"+ii+"[2, 0:1], 'r', linewidth=0.4)")

#Plot image
print("Plot image")
def update(num, data, line):
    for i in range(n**3):
        ii="{0}".format(i)
        exec("line" + ii+ ".set_data(data" +ii+"[:2, (num-10):num])")
        exec("line" + ii+ ".set_3d_properties(data" +ii+"[2, (num-10):num])")
    print(num)
    
N = 200*t

#Define 3d plot axis and label
ax.set_xlim3d([-25, 25])
ax.set_xlabel('X')
ax.set_ylim3d([-25, 25])
ax.set_ylabel('Y')
ax.set_zlim3d([0.0, 40.0])
ax.set_zlabel('Z')
ax.set_title("Lorenz Attractor")

#Make animation .mp4
ani = animation.FuncAnimation(fig, update, N, fargs=(data1, line1), interval=10000/N, blit=False)
writervideo = animation.FFMpegWriter(fps=17) 
ani.save("anima.mp4", writer=writervideo)
#ani.save('matplot003.gif', writer='imagemagick')
plt.show()