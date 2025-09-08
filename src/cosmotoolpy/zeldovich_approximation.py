from .power_spectrum import PowerSpectrum
import numpy as np

def zeldovich_approximation(pow_spec: PowerSpectrum, Ngrid: int, L: float = 1000) -> np.ndarray:
    '''
    Generate initial condition using Zeldovich approximation

    Parameters
    ----------
    pow_spec: PowerSpectrum
    Ngrid: int
        Number of grids
    L: float, optional
        Box size

    Returns
    -------
    position: 3d-array
        Position of particles
    '''
    pi = np.pi
    V  = L*L*L
    kF = 2*pi/L

    delta_k = pow_spec.get_initial_condition(Ngrid)
    nx_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    ny_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    nz_axis = np.fft.rfftfreq(Ngrid, 1/Ngrid)
    nx, ny, nz = np.meshgrid(nx_axis, ny_axis, nz_axis, indexing='ij', sparse=True)
    k_modsq = (nx*nx+ny*ny+nz*nz)*kF*kF
    inv_k_modsq = np.divide(1.0, k_modsq, out=np.zeros_like(k_modsq, dtype=float), where=k_modsq!=0)
    factor = 1j*delta_k*kF
    psi_x = np.fft.irfftn(factor*nx*inv_k_modsq) / V
    psi_y = np.fft.irfftn(factor*ny*inv_k_modsq) / V
    psi_z = np.fft.irfftn(factor*nz*inv_k_modsq) / V

    nx_axis = np.arange(Ngrid)
    ny_axis = np.arange(Ngrid)
    nz_axis = np.arange(Ngrid)
    grid = np.stack(np.meshgrid(nx_axis, ny_axis, nz_axis, indexing="ij"), axis=-1)*(L/Ngrid)
    position = grid.reshape(-1, 3) + np.stack((psi_x, psi_y, psi_z), axis=-1).reshape(-1, 3)
    
    return position