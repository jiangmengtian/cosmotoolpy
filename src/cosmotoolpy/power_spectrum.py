import numpy as np
from scipy import interpolate
from .sigma_8 import sigma_R
from .binning_correction import binning_correction
from .initial_condition import gaussian_random_field

class PowerSpectrum:
    def __init__(self, wn, ps):
        '''
        PowerSpectrum constructor

        Parameters
        ----------
        wn: array_like
            Wavenumber, 1-dimensional
        ps: array_like
            Power spectrum, 1-dimensional
        '''
        self.wn = np.asarray(wn, dtype=float)
        self.ps = np.asarray(ps, dtype=float)

        if self.wn.ndim != 1 or self.ps.ndim !=1:
            raise ValueError("wavenumber and power spectrum must be 1-dimensional arrays")

        if self.wn.shape != self.ps.shape:
            raise ValueError("wavenumber and power spectrum must have the same shape")


    def get_interpolate(self):
        self.pfit = interpolate.interp1d(self.wn, self.ps, bounds_error=False, fill_value=0)

    def get_sigma8(self):
        if not hasattr(self, 'pfit'):
            self.get_interpolate()
        self.sigma8 = sigma_R(self.pfit)

    def get_binning_correction(self, Ngrid: int):
        if not hasattr(self, 'pfit'):
            self.get_interpolate()
        k, P_k, _ = binning_correction(self.pfit, Ngrid)
        self.bc = PowerSpectrum(k, P_k)

    def get_initial_condition(self, Ngrid: int) -> np.ndarray:
        if not hasattr(self, 'pfit'):
            self.get_interpolate()
        return gaussian_random_field(self.pfit, Ngrid)