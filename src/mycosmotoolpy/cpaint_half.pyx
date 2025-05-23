import numpy as np
cimport numpy as cnp
import scipy.constants as C
from libc.math cimport floor
# cimport cython

cnp.import_array()

# @cython.boundscheck(False)  # Disable bounds checking
# @cython.wraparound(False)   # Disable negative indexing

def deconvolution(delta_k, L, Ngrid, p):
    nx_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    ny_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    nz_axis = np.fft.rfftfreq(Ngrid, 1/Ngrid)
    ny, nx, nz = np.meshgrid(nx_axis, ny_axis, nz_axis)
    window_function = (np.sinc(nx/Ngrid)*np.sinc(ny/Ngrid)*np.sinc(nz/Ngrid))**p
    
    delta_k = delta_k/window_function
    return delta_k

def ngp(reduced_pos, Ngrid):
    cdef cnp.ndarray[cnp.float32_t, ndim=2] red_pos = reduced_pos
    cdef cnp.ndarray[cnp.float64_t, ndim=3] delta_x = np.zeros((Ngrid, Ngrid, Ngrid), dtype=np.float64)
    cdef double[:, :, :] delta_x_view = delta_x
    cdef float[:, :] red_pos_view = red_pos
    cdef int particle_num = red_pos.shape[0]
    cdef int i, x_index, y_index, z_index
    for i in range(particle_num):
        x_index = <int>floor(red_pos_view[i, 0])
        y_index = <int>floor(red_pos_view[i, 1])
        z_index = <int>floor(red_pos_view[i, 2])
        delta_x_view[x_index, y_index, z_index] += 1
    return delta_x

def ngp_interlace(pos, Ngrid, L=1000):
    pi = C.pi
    kF = 2*pi/L
    V  = L*L*L
    
    particle_num = np.size(pos, axis=0)
    rhobar = particle_num/V
    grid_length = L/Ngrid
    reduced_pos = pos/grid_length
    
    delta_x1 = ngp(reduced_pos, Ngrid)/grid_length**3/rhobar
    delta_k1 = np.fft.rfftn(delta_x1)*V/Ngrid/Ngrid/Ngrid

    reduced_pos += 0.5
    reduced_pos %= Ngrid

    delta_x2 = ngp(reduced_pos, Ngrid)/grid_length**3/rhobar
    delta_k2 = np.fft.rfftn(delta_x2)*V/Ngrid/Ngrid/Ngrid
    nx_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    ny_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    nz_axis = np.fft.rfftfreq(Ngrid, 1/Ngrid)
    ny, nx, nz = np.meshgrid(nx_axis, ny_axis, nz_axis)
    phase = np.exp(1j*pi*(nx+ny+nz)/Ngrid)
    delta_k2 = delta_k2*phase
    
    delta_k = (delta_k1+delta_k2)/2
    return deconvolution(delta_k, L, Ngrid, 1)

def cic(reduced_pos, Ngrid):
    cdef cnp.ndarray[cnp.float32_t, ndim=2] red_pos = reduced_pos
    cdef cnp.ndarray[cnp.float64_t, ndim=3] delta_x = np.zeros((Ngrid, Ngrid, Ngrid), dtype=np.float64)
    cdef double[:, :, :] delta_x_view = delta_x
    cdef float[:, :] red_pos_view = red_pos
    cdef int particle_num = red_pos.shape[0]
    cdef int i, x_index, y_index, z_index
    cdef int N = Ngrid-1
    for i in range(particle_num):
        x_index = <int>floor(red_pos_view[i, 0])
        y_index = <int>floor(red_pos_view[i, 1])
        z_index = <int>floor(red_pos_view[i, 2])
        delta_x_view[x_index, y_index, z_index] += (x_index+1-red_pos_view[i, 0])*(y_index+1-red_pos_view[i, 1])*(z_index+1-red_pos_view[i, 2])
        delta_x_view[(x_index+1)&N, y_index, z_index] += (red_pos_view[i, 0]-x_index)*(y_index+1-red_pos_view[i, 1])*(z_index+1-red_pos_view[i, 2])
        delta_x_view[(x_index+1)&N, (y_index+1)&N, z_index] += (red_pos_view[i, 0]-x_index)*(red_pos_view[i, 1]-y_index)*(z_index+1-red_pos_view[i, 2])
        delta_x_view[(x_index+1)&N, (y_index+1)&N, (z_index+1)&N] += (red_pos_view[i, 0]-x_index)*(red_pos_view[i, 1]-y_index)*(red_pos_view[i, 2]-z_index)
        delta_x_view[(x_index+1)&N, y_index, (z_index+1)&N] += (red_pos_view[i, 0]-x_index)*(y_index+1-red_pos_view[i, 1])*(red_pos_view[i, 2]-z_index)
        delta_x_view[x_index, (y_index+1)&N, z_index] += (x_index+1-red_pos_view[i, 0])*(red_pos_view[i, 1]-y_index)*(z_index+1-red_pos_view[i, 2])
        delta_x_view[x_index, (y_index+1)&N, (z_index+1)&N] += (x_index+1-red_pos_view[i, 0])*(red_pos_view[i, 1]-y_index)*(red_pos_view[i, 2]-z_index)
        delta_x_view[x_index, y_index, (z_index+1)&N] += (x_index+1-red_pos_view[i, 0])*(y_index+1-red_pos_view[i, 1])*(red_pos_view[i, 2]-z_index)
    return delta_x


def cic_interlace(pos, Ngrid, L=1000):
    pi = C.pi
    kF = 2*pi/L
    V  = L*L*L

    particle_num = np.size(pos, axis=0)
    rhobar = particle_num/V
    grid_length = L/Ngrid
    reduced_pos = pos/grid_length

    delta_x1 = cic(reduced_pos, Ngrid)/grid_length**3/rhobar
    delta_k1 = np.fft.rfftn(delta_x1)*V/Ngrid/Ngrid/Ngrid

    reduced_pos+=0.5
    reduced_pos%=Ngrid
    
    delta_x2 = cic(reduced_pos, Ngrid)/grid_length**3/rhobar
    delta_k2 = np.fft.rfftn(delta_x2)*V/Ngrid/Ngrid/Ngrid
    nx_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    ny_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    nz_axis = np.fft.rfftfreq(Ngrid, 1/Ngrid)
    ny, nx, nz = np.meshgrid(nx_axis, ny_axis, nz_axis)
    phase = np.exp(1j*pi*(nx+ny+nz)/Ngrid)
    delta_k2 = delta_k2*phase
    
    delta_k = (delta_k1+delta_k2)/2
    return deconvolution(delta_k, L, Ngrid, 2)