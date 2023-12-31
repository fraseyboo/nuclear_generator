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
    parser.add_argument("-A", "-nucleons", default=config.A, type=int, help="Number of nucleons")
    parser.add_argument("-B2", "-beta2", default=config.B2, help="Beta_2 value")
    parser.add_argument("-B3", "-beta3", default=config.B3, help="Beta_3 value")
    parser.add_argument("-B4", "-beta4", default=config.B4, help="Beta_4 value")
    parser.add_argument("-M2", "-mode2", default=config.M2, type=int, help="mode_2 value")
    parser.add_argument("-M3", "-mode3", default=config.M3, type=int, help="mode_3 value")
    parser.add_argument("-M4", "-mode4", default=config.M4, type=int, help="mode_4 value")
    parser.add_argument("-F", "-frontend", default=config.frontend, type=str, help="Visualisation frontend (vtk or mpl)")
    parser.add_argument("-S", "-savename", default=config.savename, type=str, help="Save name")
    parser.add_argument("-C", "-colormap", default=config.colormap, type=str, help="Colormap")
    parser.add_argument("-G", "-gran", default=config.granularity, type=str, help="Mesh Granularity")
    parser.add_argument("-H1", "-harm1", default=config.h1, type=int, help="Harmonic n")
    parser.add_argument("-H2", "-harm2", default=config.h2, type=int, help="Harmonic m")
    parser.add_argument("-H3", "-harm3", default=config.h3, help="Harmonic Magnitude")
    parser.add_argument("-D", "-dark", default=config.dark_mode, action='store_true', help="Dark Mode")
    parser.add_argument("-R", "-rendering", default=config.rendering_style, type=int, help="Rendering mode")
    
    args = parser.parse_args()
    arguments = vars(args)
    print('Current arguments', arguments)



    config.granularity = int(arguments['G'])
    config.h1 = int(arguments['H1'])
    config.h2 = int(arguments['H2'])
    config.h3 = float(arguments['H3'])
    config.rendering_style = int(arguments['R'])

    config.dark_mode = arguments['D']
    if config.dark_mode:
        print('Dark mode enabled')



    if (config.h2 != 0):
        secondary_scalar = physics_utils.generate_spherical_harmonic(n=config.h1,m=config.h2, granularity=config.granularity, verbose=True)
    else:
        secondary_scalar = None

    config.A = int(arguments['A'])
    config.B2 = float(arguments['B2'])
    config.B3 = float(arguments['B3'])
    config.B4 = float(arguments['B4'])
    config.M2 = int(arguments['M2'])
    config.M3 = int(arguments['M3'])
    config.M4 = int(arguments['M4'])

    frontend = arguments['F']
    config.savename = arguments['S']
    config.colormap = arguments['C']
    config.secondary_scalar = secondary_scalar

    if frontend == 'mpl':
        mpl_utils.matplotlib_render()
    elif frontend == 'vtk':
        render_window = vtk_utils.vtk_render()
        # vtk_utils.write_gltf(render_window)
        
if __name__ == '__main__':
    main()
