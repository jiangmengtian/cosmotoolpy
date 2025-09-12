from .power_spectrum import PowerSpectrum
from typing import Tuple
import numpy as np
from .linear_growth_factor import linear_growth_factor

def zeldovich_approximation(pow_spec: PowerSpectrum, Ngrid: int, z_init: int, Omega_m: float = 0.23, Omega_lambda: float = 0.77, L: float = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Generate initial condition using Zeldovich approximation

    Parameters
    ----------
    pow_spec: PowerSpectrum
    Ngrid: int
        Number of grids
    z_init: int
        Initial redshift
    Omega_m: float, optional
        Proportion of matter component today
    Omega_lambda: float, optional
        Proportion of dark energy component today
    L: float, optional
        Box size

    Returns
    -------
    psix: 3d-array
        Displacement of particles in x-direction
    psiy: 3d-array
        Displacement of particles in y-direction
    psiz: 3d-array
        Displacement of particles in z-direction
    '''
    pi = np.pi
    V  = L*L*L
    kF = 2*pi/L

    delta_k = pow_spec.get_initial_condition(Ngrid)*linear_growth_factor(Omega_m, Omega_lambda, z_init)
    nx_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    ny_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    nz_axis = np.fft.rfftfreq(Ngrid, 1/Ngrid)
    nx, ny, nz = np.meshgrid(nx_axis, ny_axis, nz_axis, indexing='ij', sparse=True)
    k_modsq = (nx*nx+ny*ny+nz*nz)*kF*kF
    inv_k_modsq = np.divide(1.0, k_modsq, out=np.zeros_like(k_modsq, dtype=float), where=k_modsq!=0)
    factor = 1j*delta_k*kF
    psix = np.fft.irfftn(factor*nx*inv_k_modsq)/V*(Ngrid*Ngrid*Ngrid)
    psiy = np.fft.irfftn(factor*ny*inv_k_modsq)/V*(Ngrid*Ngrid*Ngrid)
    psiz = np.fft.irfftn(factor*nz*inv_k_modsq)/V*(Ngrid*Ngrid*Ngrid)
    
    return psix, psiy, psiz