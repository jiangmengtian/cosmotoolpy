from typing import Tuple
import numpy as np
from cosmotoolpy import cpseh

def binning_correction(P_interpolate: interp1d, Ngrid: int, L: float = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Compute the binning correction for the power spectrum

    Parameters
    ----------
    P_interpolate: interp1d
        An interp1d instance generated from arrays k and P(k)
    Ngrid: int
        Number of grids
    L: float, optional
        Box size

    Returns
    -------
    k: 1d-array
        Wavenumber of each bin
    Pk: 1d-array
        Averaged power spectrum of each bin
    Nk: 1d-array
        Number of independent modes in each bin
    '''
    pi = np.pi
    V  = L*L*L
    
    nx_axis = np.fft.fftfreq(Ngrid, L/Ngrid)
    ny_axis = np.fft.fftfreq(Ngrid, L/Ngrid)
    nz_axis = np.fft.rfftfreq(Ngrid, L/Ngrid)
    nx, ny, nz = np.meshgrid(nx_axis, ny_axis, nz_axis, indexing='ij', sparse=True)
    k_modulus = 2*pi*np.sqrt(nx*nx+ny*ny+nz*nz)
    delta_k = np.sqrt(V*P_interpolate(k_modulus))
    
    return cpseh.power_spectrum_estimator(delta_k.astype(np.complex128), Ngrid)