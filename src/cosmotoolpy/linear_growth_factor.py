import numpy as np
from scipy import integrate

def linear_growth_factor(Omega_m: float, Omega_lambda: float, z: float, H0: float = 67.36) -> float:
    '''
    Compute the linear growth factor of a cosmology at redshift z

    Parameters
    ----------
    Omega_m: float
        Proportion of matter component today
    Omega_lambda: float
        Proportion of dark energy component today
    z: float
        Redshift
    H0: float, optional
        Today's Hubble parameter

    Returns
    -------
    linear growth factor: float
        Normalized linear growth factor
    '''
    Omega_k = 1-Omega_m-Omega_lambda

    def integral(z):
        integrand = -(1+z)/(H0**3*(Omega_m*(1+z)**3+Omega_lambda+Omega_k*(1+z)**2)**(3/2))
        return integrand

    delta0  = H0*integrate.quad(integral, np.inf, 0)[0]
    delta_z = H0*(Omega_m*(1+z)**3+Omega_lambda+Omega_k*(1+z)**2)**(1/2)*integrate.quad(integral, np.inf, z)[0]

    return delta_z/delta0