# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 12:27:32 2025

@author: victi
"""

import numpy as np
import matplotlib.pyplot as plt


# DECLARO LOS PARÁMETROS :3 
beta2 = 1.0       # dispersión de segundo orden
gamma = 1         # coeficiente Kerr
alpha = 1.0       # pérdidas
A0 = 1.0          # amplitud inicial
T0 = 1.0          # acho temporal del pulso



# MALLADO TEMPORAL
Nt = 2048               # número de puntos temporales
Tmax = 20.0            
dt = Tmax / Nt
t = np.linspace(-Tmax/2, Tmax/2, Nt) # el vectorcillo

# Frecuencias angulares (FFT)
w = 2 * np.pi * np.fft.fftfreq(Nt, dt)




# MALLADO EN z (igual que en t,  vaya)
zmax = 5.0
Nz = 500
dz = zmax / Nz
z = np.linspace(0, zmax, Nz)



# CAMPO INICIAL (pulso gaussiano)
A = A0 * np.exp(-t**2 / (2*T0**2))




# Declaro el operador L (medio paso)
L_half = np.exp((1j * beta2 * w**2 / 2 - alpha / 2) * dz / 2)



# Abrimos A para ir guardandolo
A_z = np.zeros((Nz, Nt), dtype=complex)
A_z[0, :] = A


# Bucle ya del SSFM
for n in range(1, Nz):

    # Hago Fourier-- Multiplico-- Inversa
    A_w = np.fft.fft(A)
    A_w *= L_half
    A = np.fft.ifft(A_w)

    # Ya en tiempos, multplico por N
    A *= np.exp(1j * gamma * np.abs(A)**2 * dz)

    # Vuelvo a frecuencias, multiplico, e inversa
    A_w = np.fft.fft(A)
    A_w *= L_half
    A = np.fft.ifft(A_w)

    # Guardar campo
    A_z[n, :] = A


# GRÁFICAS

# 1) INTENSIDAD 

plt.figure(figsize=(8,5))
plt.imshow(
    np.abs(A_z)**2,
    extent=[t[0], t[-1], z[-1], z[0]],
    aspect='auto'
)
plt.colorbar(label='|A(z,t)|²')
plt.xlabel('Tiempo t')
plt.ylabel('Distancia z')
plt.title('Evolución temporal del pulso (SSFM)')
plt.show()


# 2) PULSO EN DISTINTAS z

plt.figure(figsize=(8,5))
for zi in [0, int(Nz/4), Nz-1]:
    plt.plot(t, np.abs(A_z[zi])**2, label=f'z = {z[zi]:.2f}')

plt.xlabel('Tiempo t')
plt.ylabel('|A(z,t)|²')
plt.title('Intensidad del pulso a distintas distancias')
plt.legend()
plt.grid(True)
plt.show()


# 3) ENVOLVENTE Y PARTE REAL (última z)

plt.figure(figsize=(8,5))
plt.plot(t, np.abs(A_z[-1]), 'k', linewidth=2, label='|A(z,t)|')
plt.plot(t, np.real(A_z[-1]), 'r', label='Re{A(z,t)}')
plt.xlabel('Tiempo t')
plt.ylabel('Amplitud')
plt.title('Envolvente y parte real del campo')
plt.legend()
plt.grid(True)
plt.show()


