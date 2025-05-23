from random import gauss
import numpy as np
import scipy.constants as C

def ic_gaussian_random_field(P_interpolate, Ngrid, L=1000, mu=0, sigma=1):
    pi = C.pi
    V  = L*L*L
    kF = 2*pi/L
    normal_factor = 1/(sigma*sigma*V*V/Ngrid**3)**(1/2)
    
    GRFx = np.random.normal(mu, sigma, Ngrid*Ngrid*Ngrid)
    GRFx = np.reshape(GRFx, (Ngrid, Ngrid, Ngrid))
    GRFk = np.fft.rfftn(GRFx)*V/Ngrid/Ngrid/Ngrid
    nx_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    ny_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    nz_axis = np.fft.rfftfreq(Ngrid, 1/Ngrid)
    ny, nx, nz = np.meshgrid(nx_axis, ny_axis, nz_axis)
    k_modulus = (nx*nx+ny*ny+nz*nz)**(1/2)*kF
    P_fit = P_interpolate(k_modulus)
    GRFk = GRFk*normal_factor*(V*P_fit)**(1/2)

    return GRFk