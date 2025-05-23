import numpy as np
import scipy.constants as C
from mycosmotoolpy import cpseh

def binned_power_spectrum(P_interpolate, Ngrid, L=1000):
    pi = C.pi
    V  = L*L*L
    kF = 2*pi/L
    
    nx_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    ny_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    nz_axis = np.fft.rfftfreq(Ngrid, 1/Ngrid)
    ny, nx, nz = np.meshgrid(nx_axis, ny_axis, nz_axis)
    k_modulus = (nx*nx+ny*ny+nz*nz)**(1/2)*kF
    delta_k = (V*P_interpolate(k_modulus))**(1/2)
    
    return cpseh.power_spectrum_estimator(delta_k.astype(np.complex128), Ngrid)