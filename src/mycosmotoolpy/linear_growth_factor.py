import numpy as np
from scipy import integrate

def lgf(Omega_m, Omega_lambda, z, H0=67.36):
    
    Omega_k = 1-Omega_m-Omega_lambda

    def integral(z):
        integrand = -(1+z)/(H0**3*(Omega_m*(1+z)**3+Omega_lambda+Omega_k*(1+z)**2)**(3/2))
        return integrand

    delta0  = H0*integrate.quad(integral, np.inf, 0)[0]
    delta_z = H0*(Omega_m*(1+z)**3+Omega_lambda+Omega_k*(1+z)**2)**(1/2)*integrate.quad(integral, np.inf, z)[0]

    return delta_z/delta0