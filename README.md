# cosmotoolpy
A tool package for cosmology research.

<!-- ## Summary of functions
- linear_growth_factor.py
    - ```lgf(Omega_m, Omega_lambda, z, H0=67.36) # return normalized linear growth factor D(z)```
- sigma_8.py
    - ```sigma_R(P_interpolate, R=8) # return sigma_8```
- initial_condition.py
    - ```ic_gaussian_random_field(P_interpolate, Ngrid, L=1000, mu=0, sigma=1) # return gaussian initial density contrast field in Fourier space```
- power_spectrum_estimator.py
    - ```power_spectrum_estimator(delta_k, Ngrid, L=1000) # return k, P_k, N_k```
- binning_correction.py
    - ```binning_correction(P_interpolate, Ngrid, L=1000) # return k, P_k, N_k```
- paint.py
    - ```ngp_interlace(pos, Ngrid, L=1000) # return density contrast field in Fourier space```
    - ```cic_interlace(pos, Ngrid, L=1000) # return density contrast field in Fourier space```
- cpower_spectrum_estimator.pyx
    - ```power_spectrum_estimator(delta_k, int Ngrid, double L=1000) # return k, P_k, N_k```
- cpaint.pyx
    - ```deconvolution(delta_k, L, Ngrid, p) # return density contrast field in Fourier space```
    - ```ngp(reduced_pos, Ngrid) # return mass field in configuration space```
    - ```ngp_interlace(pos, Ngrid, L=1000) # return density contrast field in Fourier space```
    - ```cic(reduced_pos, Ngrid) # return mass field in configuration space```
    - ```cic_interlace(pos, Ngrid, L=1000) # return density contrast field in Fourier space``` -->

## Installation
First, clone the repository and enter the project directory:
```bash
git clone https://github.com/jiangmengtian/cosmotoolpy.git
cd mycosmotoolpy
```
Then install the package in editable mode:
```bash
pip install -e .
```
This will automatically build Cython extensions. If you're modifying `.pyx` files and want to rebuild manually:
```bash
python setup.py build_ext --inplace
```