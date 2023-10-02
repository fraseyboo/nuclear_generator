#!/usr/bin/env python
import math as mth
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib import cm, colors

import vtk_utils
import mpl_utils
import physics_utils

import matplotlib

def main(frontend='vtk'):

	# Make mesh of thetas and phis
	phi, theta = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]

	# initialise deformations for a spherical nucleus with mass A=100
	beta2 = 0
	beta3 = 0
	beta4 = 0
	A = 100

	m2 = 0
	m3 = 0
	m4 = 0

	if frontend == 'mpl':
		mpl_utils.matplotlib_render(A, beta2, beta3, beta4, m2, m3, m4, theta, phi)
	elif frontend == 'vtk':
		render_window = vtk_utils.vtk_render(A, beta2, beta3, beta4, m2, m3, m4, theta, phi)
		vtk_utils.write_gltf(render_window)
		



if __name__ == '__main__':
	main(frontend='vtk')
