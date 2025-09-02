import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, abs, round, pi

def power_spectrum_estimator(cnp.ndarray[complex, ndim=3] delta_k, int Ngrid, double L=1000):
    cdef double kF = 2*pi/L
    cdef double V  = L*L*L
    
    cdef cnp.ndarray[dtype=cnp.float64_t] nx_axis = np.fft.fftfreq(Ngrid, 1.0/Ngrid)
    cdef cnp.ndarray[dtype=cnp.float64_t] ny_axis = np.fft.fftfreq(Ngrid, 1.0/Ngrid)
    cdef cnp.ndarray[dtype=cnp.float64_t] nz_axis = np.fft.rfftfreq(Ngrid, 1.0/Ngrid)
    cdef double nx_max = np.max(np.abs(nx_axis))
    cdef double ny_max = np.max(np.abs(ny_axis))
    cdef double nz_max = np.max(np.abs(nz_axis))
    cdef int n_max = <int>round(sqrt(nx_max*nx_max+ny_max*ny_max+nz_max*nz_max))+1

    cdef cnp.ndarray[dtype=cnp.float64_t] k = np.array([kF * i for i in range(n_max)], dtype=np.float64)
    cdef cnp.ndarray[dtype=cnp.float64_t] P_k = np.zeros(n_max, dtype=np.float64)
    cdef cnp.ndarray[dtype=cnp.int32_t] N_k = np.zeros(n_max, dtype=np.int32)

    cdef int i, j, m, index
    cdef int kx_size = nx_axis.size
    cdef int ky_size = ny_axis.size
    cdef int kz_size = nz_axis.size
    
    for i in range(kx_size):
        for j in range(ky_size):
            for m in range(kz_size):
                if nz_axis[m]==0 and nx_axis[i]<0:
                    continue
                if nz_axis[m]==0 and nx_axis[i]==0 and ny_axis[j]<0:
                    continue
                index = <int>round(sqrt(nx_axis[i]*nx_axis[i]+ny_axis[j]*ny_axis[j]+nz_axis[m]*nz_axis[m]))
                N_k[index] += 1
                P_k[index] += abs(delta_k[i, j, m])**2
    for i in range(n_max):
        if N_k[i]>0:
            P_k[i] = P_k[i]/V/N_k[i]

    return k[1:], P_k[1:], N_k[1:]