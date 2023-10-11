#!/usr/bin/env python
import numpy as np

import vtk_utils
import mpl_utils
import physics_utils


mesh_granularity = 50

# secondary_scalar = physics_utils.generate_spherical_harmonic(n=3,m=8, granularity=mesh_granularity)
secondary_scalar = None

def main(frontend='vtk'):

    # Make mesh of thetas and phis
    phi, theta = np.mgrid[0:2*np.pi:mesh_granularity*1j, 0:np.pi:mesh_granularity*1j]

    # initialise deformations for an example nucleus
    beta2 = 0.154
    beta3 = 0.097
    beta4 = 0.080
    A = 224

    m2 = 0
    m3 = 0
    m4 = 0

    if frontend == 'mpl':
        mpl_utils.matplotlib_render(A, beta2, beta3, beta4, m2, m3, m4, theta, phi)
    elif frontend == 'vtk':
        render_window = vtk_utils.vtk_render(A, beta2, beta3, beta4, m2, m3, m4, theta, phi, secondary_scalar=secondary_scalar)
        # vtk_utils.write_gltf(render_window)
        
if __name__ == '__main__':
    main(frontend='vtk')
