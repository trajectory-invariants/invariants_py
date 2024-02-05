# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 2023

@author: Riccardo

goal:
    insert a 3D mesh into an existing 3D plot by reading an stl file

    Parameters
    ----------
    stl_file_location: location of stl file conatining the desired mesh
    R: desired orientation of the 3D mesh
    pos: desired position of the 3D mesh
    colour: desired colour of the mesh
    ax: plot to which the mesh is added

    Returns
    -------
        a 3D plot in which the 3D mesh is inserted

"""
import numpy as np
import math as math
from mpl_toolkits import mplot3d
from stl import mesh

def plot_stl(stl_file_location,pos,R,colour,alpha,ax):
    stl_mesh = mesh.Mesh.from_file(stl_file_location)
    Tr  = np.vstack((np.hstack((R,np.array([pos]).T)), [0,0,0,1]))
    stl_mesh.transform(Tr)
    collection = mplot3d.art3d.Poly3DCollection(stl_mesh.vectors)
    collection.set_facecolor(colour); collection.set_alpha(alpha)
    ax.add_collection3d(collection)