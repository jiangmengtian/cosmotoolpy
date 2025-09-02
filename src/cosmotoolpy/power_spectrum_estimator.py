import numpy as np
import scipy.constants as C

def power_spectrum_estimator(delta_k, Ngrid, L=1000):
    pi = C.pi
    kF = 2*pi/L
    V  = L*L*L
    
    kx_size = np.size(delta_k, 0)
    ky_size = np.size(delta_k, 1)
    kz_size = np.size(delta_k, 2)
    nx_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    ny_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    nz_axis = np.fft.rfftfreq(Ngrid, 1/Ngrid)
    nx_max = np.max(np.abs(nx_axis))
    ny_max = np.max(np.abs(ny_axis))
    nz_max = np.max(np.abs(nz_axis))
    n_max = round((nx_max*nx_max+ny_max*ny_max+nz_max*nz_max)**(1/2))+1
    P_k = [0]*n_max
    N_k = [0]*n_max
    k = []
    for i in range(n_max):
        k.append(kF*i)
    
    for i in range(kx_size):
        for j in range(ky_size):
            for m in range(kz_size):
                if nz_axis[m]==0 and nx_axis[i]<0:
                    continue
                if nz_axis[m]==0 and nx_axis[i]==0 and ny_axis[j]<0:
                    continue
                index = round((nx_axis[i]*nx_axis[i]+ny_axis[j]*ny_axis[j]+nz_axis[m]*nz_axis[m])**(1/2))
                N_k[index] += 1
                P_k[index] += abs(delta_k[i, j, m])**2
    for i in range(n_max):
        if not N_k[i]==0:
            P_k[i] = P_k[i]/V/N_k[i]

    return k[1:], P_k[1:], N_k[1:]