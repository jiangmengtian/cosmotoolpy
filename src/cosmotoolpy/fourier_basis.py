import numpy as np

def fourier_basis(Ngrid: int, L: float = 1000, *, option: str = None) -> np.ndarray:
    '''
    Generate normalized Fourier spherical vector/helical vector/spherical tensor/helical tensor basis

    Parameters
    ----------
    *args
    Ngrid: int
        Number of grids
    L: float, optional
        Box size in real space
    **kwargs
    option: str
        Four option to choose the catagory of basis, 'vector spherical', 'vector helical', 'tensor spherical', 'tensor helical'

    Returns
    -------
    basis: 5d-array
        Component of independent basis in Cartesian coordinates 
    '''
    kx_axis = np.fft.fftfreq(Ngrid, L/Ngrid)
    ky_axis = np.fft.fftfreq(Ngrid, L/Ngrid)
    kz_axis = np.fft.rfftfreq(Ngrid, L/Ngrid)
    kx, ky, kz = np.meshgrid(kx_axis, ky_axis, kz_axis, indexing='ij')
    # k = np.stack((kx, ky, kz), axis=-1)
    k_mod_3d = np.sqrt(kx*kx + ky*ky + kz*kz)
    k_mod_2d = np.sqrt(kx*kx + ky*ky)
    kx_normalized = np.divide(kx, k_mod_3d, out=np.zeros_like(kx, dtype=float), where=k_mod_3d!=0)
    ky_normalized = np.divide(ky, k_mod_3d, out=np.zeros_like(ky, dtype=float), where=k_mod_3d!=0)
    kz_normalized = np.divide(kz, k_mod_3d, out=np.zeros_like(kz, dtype=float), where=k_mod_3d!=0)
    k_normalized = np.stack((kx_normalized, ky_normalized, kz_normalized), axis=-1)
    cos_theta = np.divide(kz, k_mod_3d, out=np.zeros_like(kz, dtype=float), where=k_mod_3d!=0)
    sin_theta = np.divide(k_mod_2d, k_mod_3d, out=np.zeros_like(k_mod_2d, dtype=float), where=k_mod_3d!=0)
    cos_phi = np.divide(kx, k_mod_2d, out=np.ones_like(kx, dtype=float), where=k_mod_2d!=0)
    sin_phi = np.divide(ky, k_mod_2d, out=np.zeros_like(kx, dtype=float), where=k_mod_2d!=0)
    etheta = np.stack((cos_theta*cos_phi, cos_theta*sin_phi, -sin_theta), axis=-1)
    ephi = np.stack((-sin_phi, cos_phi, np.zeros_like(kz, dtype=float)), axis=-1)
    ephi[0, 0, 0, 1] = 0

    match option:
        case 'vector spherical':
            return np.array([k_normalized, etheta, ephi])
        case 'vector helical':
            return np.array([k_normalized, (etheta+1j*ephi)/np.sqrt(2), (etheta-1j*ephi)/np.sqrt(2)])
        case 'tensor spherical':
            vec_basis = np.array([k_normalized, etheta, ephi])
            ten_basis = []
            for i in range(3):
                for j in range(i, 3):
                    ten_basis.append(np.einsum('ijkl,ijkm->ijklm', vec_basis[i], vec_basis[j]))
            decomp_basis_idx = [
                [[0, 3, 5], [np.sqrt(1/3), np.sqrt(1/3), np.sqrt(1/3)]],
                [[0, 3, 5], [np.sqrt(2/3), -np.sqrt(1/6), -np.sqrt(1/6)]],
                [[1, -1], [np.sqrt(1/2), np.sqrt(1/2)]],
                [[2, -2], [np.sqrt(1/2), np.sqrt(1/2)]],
                [[4, -4], [np.sqrt(1/2), np.sqrt(1/2)]],
                [[3, 5], [np.sqrt(1/2), -np.sqrt(1/2)]]
            ]
            ten_basis = np.stack(ten_basis, axis=0)
            decomp_basis=[]
            for i in range(6):
                decomp_basis_temp = np.zeros_like(ten_basis[0], dtype=float)
                for idx, coef in zip(decomp_basis_idx[i][0], decomp_basis_idx[i][1]):
                    if idx < 0:   
                        decomp_basis_temp += np.swapaxes(ten_basis[abs(idx)], -1, -2)*coef
                    else:
                        decomp_basis_temp += ten_basis[idx]*coef
                decomp_basis.append(decomp_basis_temp)
            return np.stack(decomp_basis, axis=0)
        case 'tensor helical':
            vec_basis = np.array([k_normalized, (etheta+1j*ephi)/np.sqrt(2), (etheta-1j*ephi)/np.sqrt(2)])
            ten_basis = []
            for i in range(3):
                for j in range(i, 3):
                    ten_basis.append(np.einsum('ijkl,ijkm->ijklm', vec_basis[i], vec_basis[j]))
            decomp_basis_idx = [
                [[0, 4, -4], [np.sqrt(1/3), np.sqrt(1/3), np.sqrt(1/3)]],
                [[0, 4, -4], [np.sqrt(2/3), -np.sqrt(1/6), -np.sqrt(1/6)]],
                [[1, -1], [np.sqrt(1/2), np.sqrt(1/2)]],
                [[2, -2], [np.sqrt(1/2), np.sqrt(1/2)]],
                [[3], [1]],
                [[5], [1]]
            ]
            ten_basis = np.stack(ten_basis, axis=0)
            decomp_basis=[]
            for i in range(6):
                decomp_basis_temp = np.zeros_like(ten_basis[0], dtype=complex)
                for idx, coef in zip(decomp_basis_idx[i][0], decomp_basis_idx[i][1]):
                    if idx < 0:   
                        decomp_basis_temp += np.swapaxes(ten_basis[abs(idx)], -1, -2)*coef
                    else:
                        decomp_basis_temp += ten_basis[idx]*coef
                decomp_basis.append(decomp_basis_temp)
            return np.stack(decomp_basis, axis=0)
        case _:
            raise TypeError("Invalid option")