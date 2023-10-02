import math as mth
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib import cm, colors

import physics_utils
import vtk_utils


# Define some limits for axes and deformations
limAxis = 15
limDef  = 5

def matplotlib_render(A, beta2, beta3, beta4, m2, m3, m4, theta, phi): 
	
	r = physics_utils.calculate_r(A, beta2, m2, beta3, m3, beta4, m4, theta)

	x = physics_utils._x(r, theta, phi)
	y = physics_utils._y(r, theta, phi)
	z = physics_utils._z(r, theta)

	R = physics_utils._R(x, y, z)
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

		r = physics_utils.calculate_r(A, beta2, m2, beta3, m3, beta4, m4, theta)

		x = physics_utils._x(r, theta, phi)
		y = physics_utils._y(r, theta, phi)
		z = physics_utils._z(r, theta)
		
		R = physics_utils._R(x, y, z)
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

		r = physics_utils.calculate_r(A, beta2, m2, beta3, m3, beta4, m4, theta)

		nuclear_shape = vtk_utils.add_spherical_function(r, add_gridlines=False)

		actor_dict = dict()
		actor_dict.update({'shape': nuclear_shape})
		
		render_window = vtk_utils.render(actors=actor_dict)

		vtk_utils.write_gltf(render_window)

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