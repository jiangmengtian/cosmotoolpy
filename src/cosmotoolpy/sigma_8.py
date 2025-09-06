import numpy as np
from scipy import integrate

def sigma_R(P_interpolate: interp1d, R: int = 8) -> float:
    '''
    Compute sigma8 from the interpolated power spectrum

    Parameters
    ----------
    P_interpolate: interp1d
        An interp1d instance generated with k and P(k)
    R: int
        Radius of the spherical top-hat window function

    Returns
    -------
    sigma: float
        Sigma8 value
    '''
    pi = np.pi
    
    def integral(k):
        p = P_interpolate(k)
        return k*k*p*(3*(np.sin(k*R)-k*R*np.cos(k*R))/k/R/k/R/k/R)**2
        
    sigma, err = integrate.quad(integral, 0, np.inf, limit=100, epsrel=1e-4)
    sigma = (sigma/2/pi/pi)**(1/2)
    
    return sigma