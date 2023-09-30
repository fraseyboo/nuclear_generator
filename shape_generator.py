#!/usr/bin/env python
import math as mth
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib import cm, colors

import vtk_utils


import matplotlib
if matplotlib.pyplot.get_backend() == 'Qt5Agg':
    try:
        matplotlib.use('GTK3Agg')
        print('backend set to GTK3Agg to avoid conflict with VTK plot')
    except:
        print('Tried to switch to GTK3, failed')
        print('Currently using:', matplotlib.get_backend())


# Define functions for calculating radial position
def _SH(m,l,x):
	return ((-1)**m) * (np.sqrt(((2*l+1)*mth.factorial(l-m))/(4*np.pi*mth.factorial(l+m)))) * sp.lpmv(m,l,x)

def _r0(A):
	return 1.2*A**(1./3)
def _r2(beta2, m2, theta):
	return beta2 * _SH(m2,2,np.cos(theta))
#	return beta2*(0.5 * np.sqrt(5/(4*np.pi)) * ( 3*(np.cos(theta))**2 - 1 ))
def _r3(beta3, m3, theta):
	return beta3 * _SH(m3,3,np.cos(theta))
#	return beta3 * ((0.25 * np.sqrt(7/np.pi)) * ( 5*(np.cos(theta))**3 - (3 * np.cos(theta))))
def _r4(beta4, m4, theta):
	return beta4 * _SH(m4,4,np.cos(theta))
#	return 0
def _rt(r0, r2, r3, r4):
	return r0 + r2 + r3 + r4

# Functions for transformation to Cartesian coordinates
def _x(r, theta, phi):
	return r * np.sin(theta) * np.cos(phi)
def _y(r, theta, phi):
	return r * np.sin(theta) * np.sin(phi)
def _z(r, theta):
	return r * np.cos(theta)

# Calculate full distance from
def _R(x, y, z):
	return (x**2 + y**2 + z**2)**(1./2)

def main():

	# Make mesh of thetas and phis
	phi, theta = np.mgrid[0:2*np.pi:50j, 0:np.pi:50j]

	# Define some limits for axes and deformations
	limAxis = 15
	limDef  = 10

	# initialise deformations for a spherical nucleus with mass A=100
	beta2 = 0
	beta3 = 0
	beta4 = 0
	A = 100

	m2 = 0
	m3 = 0
	m4 = 0

	# Calculate the stuff
	r0 = _r0(A)
	r2 = _r2(beta2, m2, theta)
	r3 = _r3(beta3, m3, theta)
	r4 = _r4(beta4, m4, theta)
	r  = _rt(r0,r2,r3,r4)

	x = _x(r, theta, phi)
	y = _y(r, theta, phi)
	z = _z(r, theta)

	R = _R(x, y, z)
	N = R/R.max()

	# Make the plot 
	fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), figsize=(6,5))
	plt.subplots_adjust(bottom=0.30, top=0.99)
	im = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=cm.jet(N), shade=True)

	# Set axis limits
	ax.axes.set_xlim3d(  left=-1*limAxis, right=limAxis) 
	ax.axes.set_ylim3d(bottom=-1*limAxis,   top=limAxis) 
	ax.axes.set_zlim3d(bottom=-1*limAxis,   top=limAxis)

	# Make a colorbar
	m = cm.ScalarMappable(cmap=cm.jet)
	m.set_array(R)
	m.set_clim(vmin=4, vmax=20)
	fig.colorbar(m)

	# Make some sliders for changing variables
	axcolor = 'lightgoldenrodyellow'
	aA  = plt.axes([0.35,0.23,0.5,0.03], facecolor=axcolor)
	aB2 = plt.axes([0.35,0.18,0.5,0.03], facecolor=axcolor)
	aB3 = plt.axes([0.35,0.13,0.5,0.03], facecolor=axcolor)
	aB4 = plt.axes([0.35,0.08,0.5,0.03], facecolor=axcolor)

	aM2 = plt.axes([0.07,0.18,0.15,0.03], facecolor=axcolor)
	aM3 = plt.axes([0.07,0.13,0.15,0.03], facecolor=axcolor)
	aM4 = plt.axes([0.07,0.08,0.15,0.03], facecolor=axcolor)

	sA  = Slider(aA,'A',0,250, valinit=A, valstep=1)
	sB2 = Slider(aB2,r'$\beta_2$',-1*limDef,limDef, valinit=beta2)
	sB3 = Slider(aB3,r'$\beta_3$',-1*limDef,limDef, valinit=beta3)
	sB4 = Slider(aB4,r'$\beta_4$',-1*limDef,limDef, valinit=beta4)
	sM2 = Slider(aM2,r'$m_2$',-2, 2, valinit=m2, valstep=1)
	sM3 = Slider(aM3,r'$m_3$',-3, 3, valinit=m3, valstep=1)
	sM4 = Slider(aM4,r'$m_4$',-4, 4, valinit=m4, valstep=1)

	resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
	reset_button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

	exportax = plt.axes([0.9, 0.025, 0.1, 0.04])
	export_button = Button(exportax, 'Export', color=axcolor, hovercolor='0.975')

	# Define function for updating plot when changing sliders
	def update(val):
		A = sA.val
		beta2 = sB2.val
		beta3 = sB3.val
		beta4 = sB4.val
		m2 = sM2.val
		m3 = sM3.val
		m4 = sM4.val
		ax.clear()

		r0 = _r0(A)
		r2 = _r2(beta2, m2, theta)
		r3 = _r3(beta3, m3, theta)
		r4 = _r4(beta4, m4, theta)
		r  = _rt(r0,r2,r3,r4)
		
		x = _x(r, theta, phi)
		y = _y(r, theta, phi)
		z = _z(r, theta)
		
		R = _R(x, y, z)
		N = R/R.max()

		im = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=cm.jet(N), shade=True)

		ax.axes.set_xlim3d(  left=-1*limAxis, right=limAxis) 
		ax.axes.set_ylim3d(bottom=-1*limAxis,   top=limAxis) 
		ax.axes.set_zlim3d(bottom=-1*limAxis,   top=limAxis)

		fig.canvas.draw_idle()

	def reset(event):
		sA.reset()
		sB2.reset()
		sB3.reset()
		sB4.reset()


	def vtk_export(event):

		A = sA.val
		beta2 = sB2.val
		beta3 = sB3.val
		beta4 = sB4.val
		m2 = sM2.val
		m3 = sM3.val
		m4 = sM4.val
		# ax.clear()

		r0 = _r0(A)
		r2 = _r2(beta2, m2, theta)
		r3 = _r3(beta3, m3, theta)
		r4 = _r4(beta4, m4, theta)
		r  = _rt(r0,r2,r3,r4).T

		nuclear_shape = vtk_utils.add_spherical_function(r, add_gridlines=False)

		actor_dict = dict()
		actor_dict.update({'shape': nuclear_shape})

		a,b,c = vtk_utils.render(actors=actor_dict)

		vtk_utils.write_gltf(b)



	# Update plot when slider is changed
	sA.on_changed(update)
	sB2.on_changed(update)
	sB3.on_changed(update)
	sB4.on_changed(update)
	sM2.on_changed(update)
	sM3.on_changed(update)
	sM4.on_changed(update)

	reset_button.on_clicked(reset)
	export_button.on_clicked(vtk_export)

	plt.show()


if __name__ == '__main__':
	main()
