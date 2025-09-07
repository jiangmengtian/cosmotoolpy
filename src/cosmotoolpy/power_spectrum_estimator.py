from typing import Tuple
import numpy as np

def power_spectrum_estimator_naive(delta_k: np.ndarray, Ngrid: int, L: float = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Compute the power spectrum of a density contrast field in Fourier space (memory saving but time consuming version)

    Parameters
    ----------
    delta_k: 3d-array
        Density contrast field in Fourier space
    Ngrid: int
        Number of grids
    L: float, optional
        Box size

    Returns
    -------
    k: 1d-array
        Wavenumber of each bin
    P_k: 1d-array
        Averaged power spectrum of each bin
    N_k: 1d-array
        Number of independent modes in each bin
    '''
    pi = np.pi
    kF = 2*pi/L
    V  = L*L*L
    
    kx_size, ky_size, kz_size = delta_k.shape
    nx_max = kx_size//2
    ny_max = ky_size//2
    nz_max = kz_size
    n_max = round(np.sqrt(nx_max*nx_max+ny_max*ny_max+nz_max*nz_max))+1
    P_k = np.zeros(n_max, dtype=float)
    N_k = np.zeros(n_max, dtype=int)
    k = kF * np.arange(n_max)
    kx_mid = (kx_size-1)//2
    ky_mid = (ky_size-1)//2
    
    for i in range(kx_size):
        nx = i-kx_size if i>kx_mid else i
        for j in range(ky_size):
            ny = j-ky_size if j>ky_mid else j
            for m in range(kz_size):
                if (m==0 and (i>kx_mid or (i==0 and j>ky_mid))): 
                continue
                index = round(np.sqrt(nx*nx+ny*ny+m*m))
                N_k[index] += 1
                P_k[index] += abs(delta_k[i, j, m])**2
    for i in range(n_max):
        if not N_k[i]==0:
            P_k[i] = P_k[i]/V/N_k[i]

    return k[1:], P_k[1:], N_k[1:]

def power_spectrum_estimator_fast(delta_k: np.ndarray, Ngrid: int, L: float = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Compute the power spectrum of a density contrast field in Fourier space (time saving but memory consuming version)
    '''
    pi = np.pi
    kF = 2*pi/L
    V  = L*L*L

    nx_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    ny_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    nz_axis = np.fft.rfftfreq(Ngrid, 1/Ngrid)
    nx, ny, nz = np.meshgrid(nx_axis, ny_axis, nz_axis, indexing='ij', sparse=True)
    k_reduced_modulus = np.sqrt(nx*nx+ny*ny+nz*nz)
    k_index = np.round(k_reduced_modulus).astype(int)
    delta_square = np.abs(delta_k)**2
    mask = ~(((nz==0) & (nx<0)) | ((nz==0) & (nx==0) & (ny<0)))
    k_index_1d = k_index[mask].ravel()
    delta_square_1d = delta_square[mask].ravel()
    N_k = np.bincount(k_index_1d)
    P_k = np.bincount(k_index_1d, weights=delta_square_1d)
    n_max = k_index_1d.max() + 1
    k = kF * np.arange(n_max)
    mask = N_k > 0
    P_k[mask] /= (V*N_k[mask])

    return k[1:], P_k[1:], N_k[1:]