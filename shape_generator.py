#!/usr/bin/env python
import numpy as np
import sys
import argparse

import vtk_utils
import mpl_utils
import physics_utils

import config

mesh_granularity = 50

# secondary_scalar = physics_utils.generate_spherical_harmonic(n=3,m=8, granularity=mesh_granularity)
secondary_scalar = None

def main():


    parser = argparse.ArgumentParser(description="Nuclear Shape Generator", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-A", "-nucleons", default=120, type=int, help="Number of nucleons")
    parser.add_argument("-B2", "-beta2", default=0, help="Beta_2 value")
    parser.add_argument("-B3", "-beta3", default=0, help="Beta_3 value")
    parser.add_argument("-B4", "-beta4", default=0, help="Beta_4 value")
    parser.add_argument("-M2", "-mode2", default=0, type=int, help="mode_2 value")
    parser.add_argument("-M3", "-mode3", default=0, type=int, help="mode_3 value")
    parser.add_argument("-M4", "-mode4", default=0, type=int, help="mode_4 value")
    parser.add_argument("-F", "-frontend", default=config.frontend, type=str, help="Visualisation frontend")
    parser.add_argument("-S", "-savename", default=config.savename, type=str, help="Save name")
    args = parser.parse_args()
    arguments = vars(args)
    print('Current arguments', arguments)

    # Make mesh of thetas and phis
    phi, theta = np.mgrid[0:2*np.pi:mesh_granularity*1j, 0:np.pi:mesh_granularity*1j]

    # initialise deformations for an example nucleus
    # beta2 = 0.154
    # beta3 = 0.097
    # beta4 = 0.080
    # A = 224

    # beta2 = 0
    # beta3 = 0
    # beta4 = 2
    # A = 224

    # m2 = 0
    # m3 = 0
    # m4 = 0

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

    if frontend == 'mpl':
        mpl_utils.matplotlib_render(A, beta2, beta3, beta4, m2, m3, m4, theta, phi)
    elif frontend == 'vtk':
        render_window = vtk_utils.vtk_render(A, beta2, beta3, beta4, m2, m3, m4, theta, phi, secondary_scalar=secondary_scalar)
        # vtk_utils.write_gltf(render_window)
        
if __name__ == '__main__':
    main()
