from typing import Tuple
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, round, pi

cnp.import_array()

def _power_spectrum_estimator_complex128(cnp.ndarray[cnp.complex128_t, ndim=3] delta_k, int Ngrid, double L) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Compute the power spectrum of a density contrast field in Fourier space for type(delta_k) == np.complex128
    '''
    cdef double kF = 2*pi/L
    cdef double V  = L*L*L

    cdef int kx_size = delta_k.shape[0]
    cdef int ky_size = delta_k.shape[1]
    cdef int kz_size = delta_k.shape[2]
    cdef int nx_max = kx_size // 2
    cdef int ny_max = ky_size // 2
    cdef int nz_max = kz_size
    cdef int n_max = <int>(round(sqrt(<double>(nx_max*nx_max+ny_max*ny_max+nz_max*nz_max)))+1)

    cdef cnp.ndarray[cnp.float64_t, ndim=1] k = np.arange(n_max, dtype=np.float64) * kF
    cdef cnp.ndarray[cnp.float64_t, ndim=1] P_k = np.zeros(n_max, dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] N_k = np.zeros(n_max, dtype=np.int64)
    
    cdef int i, j, m, index, nx, ny
    cdef int kx_mid = (kx_size-1)//2
    cdef int ky_mid = (ky_size-1)//2
    cdef cnp.complex128_t[:, :, :] delta_k_view = delta_k
    for i in range(kx_size):
        nx = i-kx_size if i>kx_mid else i
        for j in range(ky_size):
            ny = j-ky_size if j>ky_mid else j
            for m in range(kz_size):
                if (m==0 and (i>kx_mid or (i==0 and j>ky_mid))): 
                    continue
                index = <int>round(sqrt(<double>(nx*nx+ny*ny+m*m)))
                N_k[index] += 1
                P_k[index] += delta_k_view[i,j,m].real**2+delta_k_view[i,j,m].imag**2
    for i in range(n_max):
        if N_k[i]>0:
            P_k[i] = P_k[i]/(V*N_k[i])

    return k[1:], P_k[1:], N_k[1:]

def _power_spectrum_estimator_float64(cnp.ndarray[cnp.float64_t, ndim=3] delta_k, int Ngrid, double L) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Compute the power spectrum of a density contrast field in Fourier space for type(delta_k) == np.float64
    '''
    cdef double kF = 2*pi/L
    cdef double V  = L*L*L

    cdef int kx_size = delta_k.shape[0]
    cdef int ky_size = delta_k.shape[1]
    cdef int kz_size = delta_k.shape[2]
    cdef int nx_max = kx_size // 2
    cdef int ny_max = ky_size // 2
    cdef int nz_max = kz_size
    cdef int n_max = <int>(round(sqrt(<double>(nx_max*nx_max+ny_max*ny_max+nz_max*nz_max)))+1)

    cdef cnp.ndarray[cnp.float64_t, ndim=1] k = np.arange(n_max, dtype=np.float64) * kF
    cdef cnp.ndarray[cnp.float64_t, ndim=1] P_k = np.zeros(n_max, dtype=np.float64)
    cdef cnp.ndarray[cnp.int64_t, ndim=1] N_k = np.zeros(n_max, dtype=np.int64)
    
    cdef int i, j, m, index, nx, ny
    cdef int kx_mid = (kx_size-1)//2
    cdef int ky_mid = (ky_size-1)//2
    cdef double[:, :, :] delta_k_view = delta_k
    for i in range(kx_size):
        nx = i-kx_size if i>kx_mid else i
        for j in range(ky_size):
            ny = j-ky_size if j>ky_mid else j
            for m in range(kz_size):
                if (m==0 and (i>kx_mid or (i==0 and j>ky_mid))): 
                    continue
                index = <int>round(sqrt(<double>(nx*nx+ny*ny+m*m)))
                N_k[index] += 1
                P_k[index] += delta_k_view[i,j,m]**2
    for i in range(n_max):
        if N_k[i]>0:
            P_k[i] = P_k[i]/(V*N_k[i])

    return k[1:], P_k[1:], N_k[1:]

def power_spectrum_estimator(delta_k: np.ndarray, Ngrid: int, L: float = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Compute the power spectrum of a density contrast field in Fourier space

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
    if delta_k.dtype == np.complex128:
        return _power_spectrum_estimator_complex128(delta_k, Ngrid, L)
    elif delta_k.dtype == np.float64:
        return _power_spectrum_estimator_float64(delta_k, Ngrid, L)
    else:
        raise TypeError(f"Unsupported input type: {delta_k.dtype}")