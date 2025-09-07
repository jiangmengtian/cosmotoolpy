from cosmotoolpy import lgf, PowerSpectrum, power_spectrum_estimator, ngp_interlace, cic_interlace
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def test_lgf():
    z=np.linspace(0,10,1000)
    Dcdm=[]
    Dlambdacdm=[]
    Dopencdm=[]
    for i in z:
        Dcdm.append(lgf(1,0,i))
        Dlambdacdm.append(lgf(0.3, 0.7, i))
        Dopencdm.append(lgf(0.3, 0, i))

    plt.plot(z+1, Dcdm, label='CDM')
    plt.plot(z+1, Dlambdacdm, label=r'$\Lambda$CDM')
    plt.plot(z+1, Dopencdm, label='Open CDM')
    plt.xlabel("$1+z$", fontsize=15)
    plt.ylabel("$D$", fontsize=15)
    plt.tick_params(
        top='on',
        right='on',

        direction='in',
    
        labeltop='on',
        labelright='on',
    )
    plt.tick_params(top='on', which='minor',direction='in')
    plt.tick_params(top='left', which='minor',direction='in')
    plt.xlim(1, 11)
    plt.ylim(0.08, 1.1)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title("Linear Growth Factor", fontsize=20)
    plt.savefig('test_lgf.pdf')

def test_power_spectrum():
    path = "z0.dat"
    z0 = np.array(pd.read_csv(path, header=None, sep='\\s+'))
    Ngrid = 128
    kNyquist = np.pi*Ngrid/1000
    
    camb = PowerSpectrum(z0[:,0], z0[:,1])
    camb.get_interpolate()
    camb.get_sigma8()
    print(camb.sigma8)
    camb.get_binning_correction(Ngrid)
    repeat = 100 # number of realization
    realization = []
    for i in range(repeat):
        deltak = camb.get_initial_condition(Ngrid)
        realization.append(power_spectrum_estimator(deltak, Ngrid)[1])
    realization = np.array(realization)
    ensemble_mean = np.mean(realization, axis=0)
    ensemble_std = np.std(realization, axis=0, ddof=1)/np.sqrt(repeat)

    plt.errorbar(camb.bc.wn, ensemble_mean, yerr=ensemble_std, fmt='.', color='black', ecolor='red', capsize=3, label='Measurement')
    plt.plot(camb.wn, camb.ps, label='CAMB')
    plt.plot(camb.bc.wn, camb.bc.ps, label='Binning correction')
    plt.axvline(kNyquist, linewidth=1, linestyle='--', color='k', label=r'$k_\mathrm{Nyquist}$')
    plt.xlabel("$k$($h$/Mpc)", fontsize=15)
    plt.ylabel('$P$(Mpc$^3/h^3$)', fontsize=15)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(0.005,np.max(camb.bc.wn))
    plt.ylim([100,50000])
    plt.tick_params(
        top='on',
        right='on',

        direction='in',
    
        labeltop='on',
        labelright='on',
    )
    plt.tick_params(top='on', which='minor',direction='in')
    plt.legend()
    plt.title('Linear Power Spectrum (100 realizations)', fontsize=20)
    plt.savefig('test_power_spectrum.pdf')

def test_cpaint():
    position_file = 'fakedata_1.0000/fakedata_1.0000/1/Position/000000'
    position = np.fromfile(position_file, dtype=np.float32, count=-1)*0.6732117
    position = np.reshape(position,(32**3,3))
    Ngrid = 256
    kNyquist = np.pi*Ngrid/1000

    k_temp, p_temp, _ =power_spectrum_estimator(ngp_interlace(position, Ngrid), Ngrid)
    ngp = PowerSpectrum(k_temp, p_temp)
    k_temp, p_temp, _ =power_spectrum_estimator(cic_interlace(position, Ngrid), Ngrid)
    cic = PowerSpectrum(k_temp, p_temp)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].scatter(ngp.wn, ngp.ps, color='black', s=[10]*np.size(ngp.wn))
    axes[0].plot(ngp.wn, ngp.ps)
    axes[0].set_xlabel("$k$($h$/Mpc)", fontsize=15)
    axes[0].set_ylabel('$P$(Mpc$^3/h^3$)', fontsize=15)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlim(np.min(ngp.wn), kNyquist)
    axes[0].tick_params(
        top='on',
        right='on',

        direction='in',
    
        labeltop='on',
        labelright='on',
    )
    axes[0].tick_params(top='on', which='minor',direction='in')
    axes[0].set_title('Power Spectrum with Interlaced NGP', fontsize=20)

    axes[1].scatter(cic.wn, cic.ps, color='black', s=[10]*np.size(cic.wn))
    axes[1].plot(cic.wn, cic.ps)
    axes[1].set_xlabel("$k$($h$/Mpc)", fontsize=15)
    axes[1].set_ylabel('$P$(Mpc$^3/h^3$)', fontsize=15)
    axes[1].set_xscale('log')
    axes[1].set_yscale('log')
    axes[1].set_xlim(np.min(cic.wn), kNyquist)
    axes[1].tick_params(
        top='on',
        right='on',

        direction='in',
        
        labeltop='on',
        labelright='on',
    )
    axes[1].tick_params(top='on', which='minor',direction='in')
    axes[1].set_title('Power Spectrum with Interlaced CIC', fontsize=20)
    plt.tight_layout()
    plt.savefig('test_cpaint.pdf')