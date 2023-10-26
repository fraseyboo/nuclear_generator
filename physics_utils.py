import math as mth
import numpy as np
import scipy.special as sp
from scipy.special import sph_harm


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


def calculate_r(A, beta2, m2, beta3, m3, beta4, m4, theta=None, mesh_granularity=50):
    
    if theta is None:
        phi, theta = np.mgrid[0:2*np.pi:(mesh_granularity)*1j, 0:np.pi:(mesh_granularity)*1j]
    
    r0 = _r0(A)
    r2 = _r2(beta2, m2, theta)
    r3 = _r3(beta3, m3, theta)
    r4 = _r4(beta4, m4, theta)
    r  = _rt(r0,r2,r3,r4)
        
    return r


def generate_spherical_harmonic(n, m, granularity=50, radius=1, absolute=False, verbose=False):

    x_granularity = granularity 
    y_granularity = granularity 

    r = 1

    phi, theta = np.mgrid[0:np.pi:(x_granularity)*1j, 0:2 * np.pi:(y_granularity)*1j]

    common_names = ['???', 'Dipole', 'Quadrupole', 'Octupole', 'Hexadecapole']

    if verbose:
        print('n:%f, m:%f' %(n, m))
    if n == 0:
        if np.abs(m) <= 3:
            print(common_names[np.abs(m)])
        else:
            print('???')


    harmonic = sph_harm(n, m, theta, phi).real * radius

    if absolute:
        harmonic = np.abs(harmonic)

    return harmonic