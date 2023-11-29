# -*- coding: utf-8 -*-
"""
Finds the displacement angles of a series of connected objects, such that
the potential energy of the system is at a minimum, and overlays the result
on the original image of the objects.

Created on Fri Nov 17 11:43:43 2023
@author: Juke
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

plt.style.use(['dark_background'])

# %% Input Parameters

# Image 
pic = 'C:\\Users\\Juke\\Desktop\\Zelda\\Bend\\2.jpg'

# gravity (normal is 29, low grav is 7.25)
g = 29

# Number of objects not including stake
N = 20

# Length of objects (from shapeParam in RawParam spreadsheet)
L=3.712

# Mass of objects (from RigidBodyEntityParam in Interactable Objects Sheet)
m = 400

# Angle from horizontal of 1st object attached to stake, must be measured or 
# guess until it fits
th0 = 0.93 # Radians

# Pixel coordinates of the connecton to the stake
start = [1065, 587]

# Adjust this until the segments are the same length as the objects in the image
scale = 14.7

#%%

# initial guess for optimizer
angles = np.zeros(N-1)

# Calculates potential energy of a configuration, given the displacement angles
# and spring constant k
def V(A, k):
    
    # Initialize y coordinate list
    Y = [0]
    
    # Append 1st angle, will not be adjusted during optimization
    A = np.append(th0, A)
    
    # The angle from horizontal of each object is the cumulative sum of the 
    # displacement angles of the previous objects and its own 
    As = np.cumsum(A)
    
    for a in As:
       #Calculate y length of each object
        Y.append(L * np.sin(a))

    # Convert to numpy array
    Y = np.array(Y)
    # The y coordinate of each object is the sum of the y lengths of the 
    # previous objects, and it's own
    Y = np.cumsum(Y)
    
    # This turns Y into the y coordinates of the centers of mass of all but the
    # first object, which is not free to move from the optimizer's POV
    Y = Y[:-1] + np.diff(Y) / 2
    
    # V = sum of( mgy + (1/2)ka^2 )
    return m*g*Y.sum() + 0.5* k * (A[1:]** 2).sum()

# Returns the x and y coordinates of the connection points for plotting
def bendXY(k):
    
    A = minimize(lambda a: V(a, k), angles).x
    A = np.append(th0, A)
    A = np.cumsum(A)
    
    x = L * np.cos(A)
    y = L * np.sin(A)
    
    # Append the origin 
    x = np.append(0, x)
    y = np.append(0, y)
    
    # May need to delete minus sign if construct faces the other way
    x = -np.cumsum(x) * scale + start[0]
    
    y = -np.cumsum(y) * scale + start[1]
    
    return[x, y]
    
#%%

# Set spring constant and get solution
k = 82550000
x, y = bendXY(k)

# Overlay solution onto the image

fig, ax = plt.subplots(figsize=(12,6))

plt.imshow(plt.imread(pic))

ax.plot(x, y, '--', color = [0, 1, 0], linewidth = 1)

ax.scatter(x, y, color = [0, 1, 0], s=10)
