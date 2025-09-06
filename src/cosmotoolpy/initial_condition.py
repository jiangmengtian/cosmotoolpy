from random import gauss
import numpy as np

def gaussian_random_field(P_interpolate: interp1d, Ngrid: int, L: float = 1000, mu: float = 0, sigma: float = 1) -> np.ndarray:
    '''
    Generate a gaussian random field from a given power spectrum

    Parameters
    ----------
    P_interpolate: interp1d
        An interp1d instance generated from arrays k and P(k)
    Ngrid: int
        Number of grids
    L: float, optional
        Box size
    mu: float, optional
        Mean of the gaussian distribution
    sigma: float, optional
        Standard deviation of the gaussian distribution

    Returns
    -------
    GRFk: 3d-array
        Gaussian random field in Fourier space
    '''
    pi = np.pi
    V  = L*L*L
    normal_factor = 1/np.sqrt(sigma*sigma*V*V/Ngrid**3)
    
    GRFx = np.random.normal(mu, sigma, size=(Ngrid, Ngrid, Ngrid))
    GRFk = np.fft.rfftn(GRFx)*V/Ngrid/Ngrid/Ngrid
    nx_axis = np.fft.fftfreq(Ngrid, L/Ngrid)
    ny_axis = np.fft.fftfreq(Ngrid, L/Ngrid)
    nz_axis = np.fft.rfftfreq(Ngrid, L/Ngrid)
    nx, ny, nz = np.meshgrid(nx_axis, ny_axis, nz_axis, indexing='ij', sparse=True)
    k_modulus = 2*pi*np.sqrt(nx*nx+ny*ny+nz*nz)
    P_fit = P_interpolate(k_modulus)
    GRFk = GRFk*normal_factor*np.sqrt(V*P_fit)

    return GRFk