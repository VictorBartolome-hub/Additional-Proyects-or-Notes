# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 11:21:06 2025

@author: victi
"""

import numpy as np
import matplotlib.pyplot as plt

#CAS0 1: SIN DISPERSIÓN NI NADA.
# Parámetros
vg = 2.0          # velocidad de grupo
z_values = [0, 2, 4]  # posiciones de propagación
t = np.linspace(-10, 10, 1000)

# Pulso de entrada A0(t)
def A0(t):
    return np.exp(-t**2)

# Gráfica
plt.figure(figsize=(8,5))

for z in z_values:
    A = A0(t - z/vg)
    plt.plot(t, A, label=f'z = {z}')

plt.xlabel('Tiempo t')
plt.ylabel('Amplitud A(z,t)')
plt.title('Propagación sin dispersión (traslación temporal)')
plt.legend()
plt.grid(True)
plt.show()





#CASO 3) KERR Y SFM

A0 = 1.0
t0 = 1.0
gamma = 1.0
z = 3.0

t = np.linspace(-6, 6, 3000)

# Pulso de entrada
Ain = A0 * np.exp(-t**2 / (2*t0**2))

# Fase no lineal
theta = gamma * np.abs(Ain)**2 * z

# Campo con Kerr
A = Ain * np.exp(1j * theta)

# Frecuencia instantánea
omega_inst = -np.gradient(theta, t)

# --------- Gráfica 1: envolvente + parte real ---------
plt.figure(figsize=(8,5))
plt.plot(t, np.abs(A), 'k', linewidth=2, label='|A(z,t)| (envolvente)')
plt.plot(t, np.real(A), 'r', label='Re{A(z,t)}')
plt.xlabel('Tiempo t')
plt.ylabel('Amplitud')
plt.title('Efecto Kerr: envolvente y parte real')
plt.legend()
plt.grid(True)
plt.show()

# --------- Gráfica 2: frecuencia instantánea ---------
plt.figure(figsize=(8,5))
plt.plot(t, omega_inst)
plt.xlabel('Tiempo t')
plt.ylabel('ω(t)')
plt.title('Frecuencia instantánea inducida por SPM')
plt.grid(True)
plt.show()







# CASO 2: DISPERSIÓN SEGUNDO ORDEN
# Parámetros
A0 = 1.0
t0 = 1.0
beta2 = 1.0
z_values = [0, 2, 5]      # distancias de propagación
t = np.linspace(-10, 10, 2000)

# Solución analítica
def A(z, t):
    denom = t0**2 + 1j*beta2*z
    return A0 * t0 / np.sqrt(denom) * np.exp(-t**2 / (2*denom))

# Gráfica
plt.figure(figsize=(8,5))

for z in z_values:
    plt.plot(t, np.abs(A(z, t)), label=f'z = {z}')

plt.xlabel('Tiempo t')
plt.ylabel('|A(z,t)|')
plt.title('Propagación con dispersión de segundo orden')
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(8,5))
for z in z_values:
    plt.plot(t, np.real(A(z,t)), label=f'z = {z}')

plt.xlabel('Tiempo t')
plt.ylabel('Re{A(z,t)}')
plt.title('Pulso gaussiano con dispersión (parte real)')
plt.legend()
plt.grid(True)
plt.show()








#SOLITON 1

# Variables
T = np.linspace(-8, 8, 3000)
Z_values = [0, np.pi/2, np.pi]

# Solitón fundamental
def q(Z, T):
    return 1/np.cosh(T) * np.exp(1j * Z/2)

# --------- Gráfica 1: intensidad ---------
plt.figure(figsize=(8,5))
for Z in Z_values:
    plt.plot(T, np.abs(q(Z, T))**2, label=f'Z = {Z:.2f}')

plt.xlabel('T')
plt.ylabel('|q(Z,T)|²')
plt.title('Solitón fundamental – Intensidad')
plt.legend()
plt.grid(True)
plt.show()

# --------- Gráfica 2: envolvente + parte real ---------
plt.figure(figsize=(8,5))
for Z in Z_values:
    plt.plot(T, np.abs(q(Z, T)), '--', linewidth=2,
             label=f'|q|, Z = {Z:.2f}')
    plt.plot(T, np.real(q(Z, T)),
             label=f'Re(q), Z = {Z:.2f}')

plt.xlabel('T')
plt.ylabel('Amplitud')
plt.title('Solitón fundamental – Envolvente y parte real')
plt.legend()
plt.grid(True)
plt.show()

