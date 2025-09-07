import numpy as np
import math

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

def _ngp(reduced_pos: np.ndarray, Ngrid: int) -> np.ndarray:
    '''
    Particle assignment of ngp scheme
    '''
    delta_x = np.zeros((Ngrid, Ngrid, Ngrid))
    particle_num = reduced_pos.shape[0]
    for i in range(particle_num):
    	x_index = math.floor(reduced_pos[i,0])
    	y_index = math.floor(reduced_pos[i,1])
    	z_index = math.floor(reduced_pos[i,2])
    	delta_x[x_index, y_index, z_index] += 1
    
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
    
    delta_x1 = _ngp(reduced_pos, Ngrid)/(grid_length**3*rhobar)
    delta_k1 = np.fft.rfftn(delta_x1)*V/(Ngrid*Ngrid*Ngrid)

    reduced_pos += 0.5
    reduced_pos %= Ngrid
    delta_x2 = _ngp(reduced_pos, Ngrid)/(grid_length**3*rhobar)
    delta_k2 = np.fft.rfftn(delta_x2)*V/(Ngrid*Ngrid*Ngrid)
    
    nx_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    ny_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    nz_axis = np.fft.rfftfreq(Ngrid, 1/Ngrid)
    nx, ny, nz = np.meshgrid(nx_axis, ny_axis, nz_axis, indexing='ij', sparse=True)
    phase = np.exp(1j*pi*(nx+ny+nz)/Ngrid)
    delta_k2 = delta_k2*phase
    
    delta_k = (delta_k1+delta_k2)/2
    return deconvolution(delta_k, Ngrid, 1)

def _cic(reduced_pos: np.ndarray, Ngrid: int) -> np.ndarray:
    '''
    Particle assignment of cic scheme
    '''
    delta_x = np.zeros((Ngrid, Ngrid, Ngrid))
    particle_num = reduced_pos.shape[0]
    for i in range(particle_num):
        x_index = math.floor(reduced_pos[i,0])
        y_index = math.floor(reduced_pos[i,1])
        z_index = math.floor(reduced_pos[i,2])
        wx1 = reduced_pos[i,0]-x_index
        wy1 = reduced_pos[i,1]-y_index
        wz1 = reduced_pos[i,2]-z_index
        wx0 = 1-wx1
        wy0 = 1-wy1
        wz0 = 1-wz1
        
        delta_x[x_index, y_index, z_index] += wx0*wy0*wz0
        delta_x[(x_index+1)%Ngrid, y_index, z_index] += wx1*wy0*wz0
        delta_x[x_index, (y_index+1)%Ngrid, z_index] += wx0*wy1*wz0
        delta_x[x_index, y_index, (z_index+1)%Ngrid] += wx0*wy0*wz1
        delta_x[(x_index+1)%Ngrid, (y_index+1)%Ngrid, z_index] += wx1*wy1*wz0
        delta_x[(x_index+1)%Ngrid, y_index, (z_index+1)%Ngrid] += wx1*wy0*wz1
        delta_x[x_index, (y_index+1)%Ngrid, (z_index+1)%Ngrid] += wx0*wy1*wz1
        delta_x[(x_index+1)%Ngrid, (y_index+1)%Ngrid, (z_index+1)%Ngrid] += wx1*wy1*wz1
        
    return delta_x        

def cic_interlace(pos, Ngrid, L=1000):
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
    
    delta_x1 = _cic(reduced_pos, Ngrid)/(grid_length**3*rhobar)
    delta_k1 = np.fft.rfftn(delta_x1)*V/(Ngrid*Ngrid*Ngrid)

    reduced_pos += 0.5
    reduced_pos %= Ngrid
    delta_x2 = _cic(reduced_pos, Ngrid)/(grid_length**3*rhobar)
    delta_k2 = np.fft.rfftn(delta_x2)*V/(Ngrid*Ngrid*Ngrid)
    nx_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    ny_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    nz_axis = np.fft.rfftfreq(Ngrid, 1/Ngrid)
    nx, ny, nz = np.meshgrid(nx_axis, ny_axis, nz_axis, indexing='ij', sparse=True)
    phase = np.exp(1j*pi*(nx+ny+nz)/Ngrid)
    delta_k2 = delta_k2*phase
    
    delta_k = (delta_k1+delta_k2)/2
    return deconvolution(delta_k, Ngrid, 2)