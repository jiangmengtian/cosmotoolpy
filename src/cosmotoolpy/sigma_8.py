import numpy as np
from scipy import integrate

def sigma_R(P_interpolate, R=8):
    pi = np.pi
    
    def integral(k):
        p = P_interpolate(k).item()
        return k*k*p*(3*(np.sin(k*R)-k*R*np.cos(k*R))/k/R/k/R/k/R)**2
        
    sigma, err = integrate.quad(integral, 0, np.inf, limit=100, epsrel=1e-4)
    sigma = (sigma/2/pi/pi)**(1/2)
    
    return sigma