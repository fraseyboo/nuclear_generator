#!/usr/bin/env python
import numpy as np
import sys
import argparse

import vtk_utils
import mpl_utils
import physics_utils

import config

from matplotlib import pyplot as plt

def main():


    parser = argparse.ArgumentParser(description="Nuclear Shape Generator", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-A", "-nucleons", default=120, type=int, help="Number of nucleons")
    parser.add_argument("-B2", "-beta2", default=0, help="Beta_2 value")
    parser.add_argument("-B3", "-beta3", default=0, help="Beta_3 value")
    parser.add_argument("-B4", "-beta4", default=0, help="Beta_4 value")
    parser.add_argument("-M2", "-mode2", default=0, type=int, help="mode_2 value")
    parser.add_argument("-M3", "-mode3", default=0, type=int, help="mode_3 value")
    parser.add_argument("-M4", "-mode4", default=0, type=int, help="mode_4 value")
    parser.add_argument("-F", "-frontend", default=config.frontend, type=str, help="Visualisation frontend (vtk or mpl)")
    parser.add_argument("-S", "-savename", default=config.savename, type=str, help="Save name")
    parser.add_argument("-C", "-colormap", default=config.colormap, type=str, help="Colormap")
    parser.add_argument("-G", "-gran", default=50, type=str, help="Mesh Granularity")
    parser.add_argument("-H1", "-harm1", default=0, type=int, help="Harmonic n")
    parser.add_argument("-H2", "-harm2", default=0, type=int, help="Harmonic m")
    parser.add_argument("-H3", "-harm3", default=0, help="Harmonic Magnitude")
    parser.add_argument("-D", "-dark", default=False, action='store_true', help="Dark Mode")
    
    args = parser.parse_args()
    arguments = vars(args)
    print('Current arguments', arguments)

    mesh_granularity = int(arguments['G'])
    h1 = int(arguments['H1'])
    h2 = int(arguments['H2'])
    h3 = float(arguments['H3'])

    config.granularity = mesh_granularity
    config.h1 = h1
    config.h2 = h2 
    config.h3 = h3

    config.dark_mode = arguments['D']
    if config.dark_mode:
        print('Dark mode enabled')



    if (h2 != 0):
        secondary_scalar = physics_utils.generate_spherical_harmonic(n=h1,m=h2, granularity=mesh_granularity, verbose=True)
    else:
        secondary_scalar = None
    # Make mesh of thetas and phi
    phi, theta = np.mgrid[0:2*np.pi:mesh_granularity*1j, 0:np.pi:mesh_granularity*1j]

    A = int(arguments['A'])
    beta2 = float(arguments['B2'])
    beta3 = float(arguments['B3'])
    beta4 = float(arguments['B4'])
    m2 = int(arguments['M2'])
    m3 = int(arguments['M3'])
    m4 = int(arguments['M4'])


    config.A = int(arguments['A'])
    config.B2 = float(arguments['B2'])
    config.B3 = float(arguments['B3'])
    config.B4 = float(arguments['B4'])
    config.M2 = float(arguments['M2'])
    config.M3 = float(arguments['M3'])
    config.M4 = float(arguments['M4'])

    frontend = arguments['F']
    config.savename = arguments['S']
    config.colormap = arguments['C']
    config.secondary_scalar = secondary_scalar

    if frontend == 'mpl':
        mpl_utils.matplotlib_render(A, beta2, beta3, beta4, m2, m3, m4, theta, phi)
    elif frontend == 'vtk':
        render_window = vtk_utils.vtk_render(A, beta2, beta3, beta4, m2, m3, m4, theta, phi)
        # vtk_utils.write_gltf(render_window)
        
if __name__ == '__main__':
    main()
