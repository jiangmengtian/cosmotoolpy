import numpy as np
import scipy.constants as C
import math

def deconvolution(delta_k, L, Ngrid, p):
    nx_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    ny_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    nz_axis = np.fft.rfftfreq(Ngrid, 1/Ngrid)
    ny, nx, nz = np.meshgrid(nx_axis, ny_axis, nz_axis)
    window_function = (np.sinc(nx/Ngrid)*np.sinc(ny/Ngrid)*np.sinc(nz/Ngrid))**p
    
    delta_k = delta_k/window_function
    return delta_k

def ngp(pos, Ngrid, L=1000):
    pi = C.pi
    kF = 2*pi/L
    V  = L*L*L

    particle_num = np.size(pos, axis=0)
    rhobar = particle_num/V
    grid_length = L/Ngrid
    reduced_pos = pos/grid_length
    
    delta_x1 = np.zeros((Ngrid, Ngrid, Ngrid))
    for i in range(particle_num):
    	x_index = math.floor(reduced_pos[i,0])
    	y_index = math.floor(reduced_pos[i,1])
    	z_index = math.floor(reduced_pos[i,2])
    	delta_x1[x_index, y_index, z_index] += 1
    delta_x1 = delta_x1/grid_length**3/rhobar
    delta_k1 = np.fft.rfftn(delta_x1)*V/Ngrid/Ngrid/Ngrid

    reduced_pos += 0.5
    reduced_pos %= Ngrid
    delta_x2 = np.zeros((Ngrid, Ngrid, Ngrid))
    for i in range(particle_num):
    	x_index = math.floor(reduced_pos[i,0])
    	y_index = math.floor(reduced_pos[i,1])
    	z_index = math.floor(reduced_pos[i,2])
    	delta_x2[x_index, y_index, z_index] += 1
    delta_x2 = delta_x2/grid_length**3/rhobar
    delta_k2 = np.fft.rfftn(delta_x2)*V/Ngrid/Ngrid/Ngrid
    nx_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    ny_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    nz_axis = np.fft.rfftfreq(Ngrid, 1/Ngrid)
    ny, nx, nz=np.meshgrid(nx_axis, ny_axis, nz_axis)
    phase = np.exp(1j*pi*(nx+ny+nz)/Ngrid)
    delta_k2 = delta_k2*phase
    
    delta_k = (delta_k1+delta_k2)/2
    return deconvolution(delta_k, L, Ngrid, 1)

def cic(pos, Ngrid, L=1000):
    pi = C.pi
    kF = 2*pi/L
    V  = L*L*L

    particle_num = np.size(pos, axis=0)
    rhobar = particle_num/V
    grid_length = L/Ngrid
    reduced_pos = pos/grid_length
    
    delta_x1 = np.zeros((Ngrid, Ngrid, Ngrid))
    for i in range(particle_num):
        x_index = math.floor(reduced_pos[i,0])
        y_index = math.floor(reduced_pos[i,1])
        z_index = math.floor(reduced_pos[i,2])
        delta_x1[x_index, y_index, z_index] += (x_index+1-reduced_pos[i,0])*(y_index+1-reduced_pos[i,1])*(z_index+1-reduced_pos[i,2])
        delta_x1[(x_index+1)%Ngrid, y_index, z_index] += (reduced_pos[i,0]-x_index)*(y_index+1-reduced_pos[i,1])*(z_index+1-reduced_pos[i,2])
        delta_x1[(x_index+1)%Ngrid, (y_index+1)%Ngrid, z_index] += (reduced_pos[i,0]-x_index)*(reduced_pos[i,1]-y_index)*(z_index+1-reduced_pos[i,2])
        delta_x1[(x_index+1)%Ngrid, (y_index+1)%Ngrid, (z_index+1)%Ngrid] += (reduced_pos[i,0]-x_index)*(reduced_pos[i,1]-y_index)*(reduced_pos[i,2]-z_index)
        delta_x1[(x_index+1)%Ngrid, y_index, (z_index+1)%Ngrid] += (reduced_pos[i,0]-x_index)*(y_index+1-reduced_pos[i,1])*(reduced_pos[i,2]-z_index)
        delta_x1[x_index, (y_index+1)%Ngrid, z_index] += (x_index+1-reduced_pos[i,0])*(reduced_pos[i,1]-y_index)*(z_index+1-reduced_pos[i,2])
        delta_x1[x_index, (y_index+1)%Ngrid, (z_index+1)%Ngrid] += (x_index+1-reduced_pos[i,0])*(reduced_pos[i,1]-y_index)*(reduced_pos[i,2]-z_index)
        delta_x1[x_index, y_index, (z_index+1)%Ngrid] += (x_index+1-reduced_pos[i,0])*(y_index+1-reduced_pos[i,1])*(reduced_pos[i,2]-z_index)
    delta_x1 = delta_x1/grid_length**3/rhobar
    delta_k1 = np.fft.rfftn(delta_x1)*V/Ngrid/Ngrid/Ngrid

    reduced_pos+=0.5
    reduced_pos%=Ngrid
    delta_x2=np.zeros((Ngrid,Ngrid,Ngrid))
    for i in range(particle_num):
        x_index = math.floor(reduced_pos[i,0])
        y_index = math.floor(reduced_pos[i,1])
        z_index = math.floor(reduced_pos[i,2])
        delta_x2[x_index, y_index,z_index] += (x_index+1-reduced_pos[i,0])*(y_index+1-reduced_pos[i,1])*(z_index+1-reduced_pos[i,2])
        delta_x2[(x_index+1)%Ngrid, y_index,z_index] += (reduced_pos[i,0]-x_index)*(y_index+1-reduced_pos[i,1])*(z_index+1-reduced_pos[i,2])
        delta_x2[(x_index+1)%Ngrid, (y_index+1)%Ngrid, z_index] += (reduced_pos[i,0]-x_index)*(reduced_pos[i,1]-y_index)*(z_index+1-reduced_pos[i,2])
        delta_x2[(x_index+1)%Ngrid, (y_index+1)%Ngrid, (z_index+1)%Ngrid] += (reduced_pos[i,0]-x_index)*(reduced_pos[i,1]-y_index)*(reduced_pos[i,2]-z_index)
        delta_x2[(x_index+1)%Ngrid, y_index, (z_index+1)%Ngrid] += (reduced_pos[i,0]-x_index)*(y_index+1-reduced_pos[i,1])*(reduced_pos[i,2]-z_index)
        delta_x2[x_index, (y_index+1)%Ngrid, z_index] += (x_index+1-reduced_pos[i,0])*(reduced_pos[i,1]-y_index)*(z_index+1-reduced_pos[i,2])
        delta_x2[x_index, (y_index+1)%Ngrid, (z_index+1)%Ngrid] += (x_index+1-reduced_pos[i,0])*(reduced_pos[i,1]-y_index)*(reduced_pos[i,2]-z_index)
        delta_x2[x_index, y_index, (z_index+1)%Ngrid] += (x_index+1-reduced_pos[i,0])*(y_index+1-reduced_pos[i,1])*(reduced_pos[i,2]-z_index)
    delta_x2=delta_x2/grid_length**3/rhobar
    delta_k2=np.fft.rfftn(delta_x2)*V/Ngrid/Ngrid/Ngrid
    nx_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    ny_axis = np.fft.fftfreq(Ngrid, 1/Ngrid)
    nz_axis = np.fft.rfftfreq(Ngrid, 1/Ngrid)
    ny, nx, nz=np.meshgrid(nx_axis, ny_axis, nz_axis)
    phase = np.exp(1j*pi*(nx+ny+nz)/Ngrid)
    delta_k2 = delta_k2*phase
    
    delta_k = (delta_k1+delta_k2)/2
    return deconvolution(delta_k, L, Ngrid, 2)