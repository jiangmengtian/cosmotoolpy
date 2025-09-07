import numpy as np
cimport numpy as cnp
from libc.math cimport floor
# cimport cython

cnp.import_array()

# @cython.boundscheck(False)  # Disable bounds checking
# @cython.wraparound(False)   # Disable negative indexing

def deconvolution(delta_k: np.ndarray, Ngrid: int, p: int) -> np.ndarray:
    '''
    Deconvolve window function

    Parameters
    ----------
    delta_k: 3d-array
        Density contrast field
    Ngrid: int
        Number of grids
    p: int
        Order of the window function

    Returns
    -------
    delta_k: 3d-array
        Deconvolved density contrast
    '''
    nx_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    ny_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    nz_axis = np.fft.rfftfreq(Ngrid, 1/Ngrid)
    nx, ny, nz = np.meshgrid(nx_axis, ny_axis, nz_axis, indexing='ij', sparse=True)
    window_function = (np.sinc(nx/Ngrid)*np.sinc(ny/Ngrid)*np.sinc(nz/Ngrid))**p
    delta_k = delta_k/window_function
    return delta_k

def _ngp_float32(cnp.ndarray[cnp.float32_t, ndim=2] reduced_pos, int Ngrid) -> np.ndarray:
    cdef cnp.ndarray[cnp.float64_t, ndim=3] delta_x = np.zeros((Ngrid, Ngrid, Ngrid), dtype=np.float64)
    cdef double[:, :, :] delta_x_view = delta_x
    cdef float[:, :] reduced_pos_view = reduced_pos
    cdef int particle_num = reduced_pos.shape[0]
    cdef int i, x_index, y_index, z_index
    for i in range(particle_num):
        x_index = <int>floor(reduced_pos_view[i, 0])
        y_index = <int>floor(reduced_pos_view[i, 1])
        z_index = <int>floor(reduced_pos_view[i, 2])
        delta_x_view[x_index, y_index, z_index] += 1
    return delta_x

def _ngp_float64(cnp.ndarray[cnp.float64_t, ndim=2] reduced_pos, int Ngrid) -> np.ndarray:
    cdef cnp.ndarray[cnp.float64_t, ndim=3] delta_x = np.zeros((Ngrid, Ngrid, Ngrid), dtype=np.float64)
    cdef double[:, :, :] delta_x_view = delta_x
    cdef double[:, :] reduced_pos_view = reduced_pos
    cdef int particle_num = reduced_pos.shape[0]
    cdef int i, x_index, y_index, z_index
    for i in range(particle_num):
        x_index = <int>floor(reduced_pos_view[i, 0])
        y_index = <int>floor(reduced_pos_view[i, 1])
        z_index = <int>floor(reduced_pos_view[i, 2])
        delta_x_view[x_index, y_index, z_index] += 1
    return delta_x

def ngp_interlace(pos: np.ndarray, Ngrid: int, L: float = 1000) -> np.ndarray:
    '''
    Compute the corresponding density contrast in Fourier space of particles under ngp scheme

    Parameters
    ----------
    pos: 2d-array
        Position of particles
    Ngrid: int
        Number of grids
    L: float, optional
        Box size

    Returns
    -------
    delta_k: 3d-array
        Density contrast field in Fourier space
    '''
    pi = np.pi
    kF = 2*pi/L
    V  = L*L*L
    particle_num = pos.shape[0]
    rhobar = particle_num/V
    grid_length = L/Ngrid
    reduced_pos = pos/grid_length

    if reduced_pos.dtype == np.float32:
        delta_x1 = _ngp_float32(reduced_pos, Ngrid)/(grid_length**3*rhobar)
    elif reduced_pos.dtype == np.float64:
        delta_x1 = _ngp_float64(reduced_pos, Ngrid)/(grid_length**3*rhobar)
    else:
        raise TypeError(f"Unsupported input type: {reduced_pos.dtype}")
    delta_k1 = np.fft.rfftn(delta_x1)*V/(Ngrid*Ngrid*Ngrid)

    reduced_pos += 0.5
    reduced_pos %= Ngrid
    if reduced_pos.dtype == np.float32:
        delta_x2 = _ngp_float32(reduced_pos, Ngrid)/(grid_length**3*rhobar)
    elif reduced_pos.dtype == np.float64:
        delta_x2 = _ngp_float64(reduced_pos, Ngrid)/(grid_length**3*rhobar)
    else:
        raise TypeError(f"Unsupported input type: {reduced_pos.dtype}")
    delta_k2 = np.fft.rfftn(delta_x2)*V/(Ngrid*Ngrid*Ngrid)
    nx_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    ny_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    nz_axis = np.fft.rfftfreq(Ngrid, 1/Ngrid)
    nx, ny, nz = np.meshgrid(nx_axis, ny_axis, nz_axis, indexing='ij', sparse=True)
    phase = np.exp(1j*pi*(nx+ny+nz)/Ngrid)
    delta_k2 = delta_k2*phase
    
    delta_k = (delta_k1+delta_k2)/2
    return deconvolution(delta_k, Ngrid, 1)

def _cic_float32(cnp.ndarray[cnp.float32_t, ndim=2] reduced_pos, int Ngrid) -> np.ndarray:
    cdef cnp.ndarray[cnp.float64_t, ndim=3] delta_x = np.zeros((Ngrid, Ngrid, Ngrid), dtype=np.float64)
    cdef double[:, :, :] delta_x_view = delta_x
    cdef float[:, :] reduced_pos_view = reduced_pos
    cdef int particle_num = reduced_pos.shape[0]
    cdef int i, x_index, y_index, z_index
    cdef int N = Ngrid-1
    cdef double wx1, wy1, wz1, wx0, wy0, wz0
    for i in range(particle_num):
        x_index = <int>floor(reduced_pos_view[i, 0])
        y_index = <int>floor(reduced_pos_view[i, 1])
        z_index = <int>floor(reduced_pos_view[i, 2])
        wx1 = reduced_pos_view[i,0]-x_index
        wy1 = reduced_pos_view[i,1]-y_index
        wz1 = reduced_pos_view[i,2]-z_index
        wx0 = 1-wx1
        wy0 = 1-wy1
        wz0 = 1-wz1
        delta_x_view[x_index, y_index, z_index] += wx0*wy0*wz0
        delta_x_view[(x_index+1)&N, y_index, z_index] += wx1*wy0*wz0
        delta_x_view[x_index, (y_index+1)&N, z_index] += wx0*wy1*wz0
        delta_x_view[x_index, y_index, (z_index+1)&N] += wx0*wy0*wz1
        delta_x_view[(x_index+1)&N, (y_index+1)&N, z_index] += wx1*wy1*wz0
        delta_x_view[(x_index+1)&N, y_index, (z_index+1)&N] += wx1*wy0*wz1
        delta_x_view[x_index, (y_index+1)&N, (z_index+1)&N] += wx0*wy1*wz1
        delta_x_view[(x_index+1)&N, (y_index+1)&N, (z_index+1)&N] += wx1*wy1*wz1

    return delta_x

def _cic_float64(cnp.ndarray[cnp.float64_t, ndim=2] reduced_pos, int Ngrid) -> np.ndarray:
    cdef cnp.ndarray[cnp.float64_t, ndim=3] delta_x = np.zeros((Ngrid, Ngrid, Ngrid), dtype=np.float64)
    cdef double[:, :, :] delta_x_view = delta_x
    cdef double[:, :] reduced_pos_view = reduced_pos
    cdef int particle_num = reduced_pos.shape[0]
    cdef int i, x_index, y_index, z_index
    cdef int N = Ngrid-1
    cdef double wx1, wy1, wz1, wx0, wy0, wz0
    for i in range(particle_num):
        x_index = <int>floor(reduced_pos_view[i, 0])
        y_index = <int>floor(reduced_pos_view[i, 1])
        z_index = <int>floor(reduced_pos_view[i, 2])
        wx1 = reduced_pos_view[i,0]-x_index
        wy1 = reduced_pos_view[i,1]-y_index
        wz1 = reduced_pos_view[i,2]-z_index
        wx0 = 1-wx1
        wy0 = 1-wy1
        wz0 = 1-wz1
        delta_x_view[x_index, y_index, z_index] += wx0*wy0*wz0
        delta_x_view[(x_index+1)&N, y_index, z_index] += wx1*wy0*wz0
        delta_x_view[x_index, (y_index+1)&N, z_index] += wx0*wy1*wz0
        delta_x_view[x_index, y_index, (z_index+1)&N] += wx0*wy0*wz1
        delta_x_view[(x_index+1)&N, (y_index+1)&N, z_index] += wx1*wy1*wz0
        delta_x_view[(x_index+1)&N, y_index, (z_index+1)&N] += wx1*wy0*wz1
        delta_x_view[x_index, (y_index+1)&N, (z_index+1)&N] += wx0*wy1*wz1
        delta_x_view[(x_index+1)&N, (y_index+1)&N, (z_index+1)&N] += wx1*wy1*wz1

    return delta_x

def cic_interlace(pos: np.ndarray, Ngrid: int, L: float = 1000) -> np.ndarray:
    '''
    Compute the corresponding density contrast in Fourier space of particles under cic scheme

    Parameters
    ----------
    pos: 2d-array
        Position of particles
    Ngrid: int
        Number of grids
    L: float, optional
        Box size

    Returns
    -------
    delta_k: 3d-array
        Density contrast field in Fourier space
    '''
    pi = np.pi
    kF = 2*pi/L
    V  = L*L*L
    particle_num = pos.shape[0]
    rhobar = particle_num/V
    grid_length = L/Ngrid
    reduced_pos = pos/grid_length

    if reduced_pos.dtype == np.float32:
        delta_x1 = _cic_float32(reduced_pos, Ngrid)/(grid_length**3*rhobar)
    elif reduced_pos.dtype == np.float64:
        delta_x1 = _cic_float64(reduced_pos, Ngrid)/(grid_length**3*rhobar)
    else:
        raise TypeError(f"Unsupported input type: {reduced_pos.dtype}")
    delta_k1 = np.fft.rfftn(delta_x1)*V/(Ngrid*Ngrid*Ngrid)

    reduced_pos += 0.5
    reduced_pos %= Ngrid
    if reduced_pos.dtype == np.float32:
        delta_x2 = _cic_float32(reduced_pos, Ngrid)/(grid_length**3*rhobar)
    elif reduced_pos.dtype == np.float64:
        delta_x2 = _cic_float64(reduced_pos, Ngrid)/(grid_length**3*rhobar)
    else:
        raise TypeError(f"Unsupported input type: {reduced_pos.dtype}")
    delta_k2 = np.fft.rfftn(delta_x2)*V/(Ngrid*Ngrid*Ngrid)
    nx_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    ny_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    nz_axis = np.fft.rfftfreq(Ngrid, 1/Ngrid)
    nx, ny, nz = np.meshgrid(nx_axis, ny_axis, nz_axis, indexing='ij', sparse=True)
    phase = np.exp(1j*pi*(nx+ny+nz)/Ngrid)
    delta_k2 = delta_k2*phase
    
    delta_k = (delta_k1+delta_k2)/2
    return deconvolution(delta_k, L, Ngrid, 2)