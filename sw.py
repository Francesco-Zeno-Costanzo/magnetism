"""
Simple code to simulate Stoner-Wohlfarth hysteresis loops for a uniaxial single-domain particle.
The model considers uniaxial anisotropy and an external magnetic field at an angle to the easy axis.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


def energy(theta, phi, H, Ku, mu0, Ms):
    '''
    Energy function for Stoner-Wohlfarth model.
    Anisotropy + Zeeman energy.

    Parameters
    ----------
    theta : float
        Angle of magnetization with respect to easy axis (radians).
    phi : float
        Angle of applied field with respect to easy axis (radians).
    H : float
        Applied magnetic field magnitude.
    Ku : float
        Uniaxial anisotropy constant.
    mu0 : float
        Permeability.
    Ms : float
        Saturation magnetization.
    
    Returns
    -------
    Total energy (float)
    '''
    return Ku * np.sin(theta)**2 - mu0 * H * Ms * np.cos(phi - theta)

def find_minimum(H, phi, args=()):
    '''
    Find the equilibrium angle theta that minimizes the energy for given H and phi.

    Parameters
    ----------
    H : float
        Applied magnetic field magnitude.
    phi : float
        Angle of applied field with respect to easy axis (radians).
    args : tuple, optional
        Additional arguments for the energy function (default is empty).

    Returns
    -------
    minimum of the energy (float)
    '''
    
    fun = lambda th: energy(th, phi, H, *args)
    res = minimize_scalar(fun, bounds=(0, 2*np.pi), method='bounded')
    return res.x

def compute_hysteresis(H_vals, phi, args=()):
    '''
    Simulates the hysteresis loop along a trajectory of the H_vals fields,
    for a given angle phi between the field and the easy axis.
    Returns the array of magnetization projected along H.

    Parameters
    ----------
    H_vals : array_like
        Array of applied magnetic field magnitudes.
    phi : float
        Angle of applied field with respect to easy axis (radians).
    args : tuple, optional
        Additional arguments for the energy function (default is empty).

    Return
    ------
    M_par : 1darray
        Magnetization
    thetas : 1darray
        positions of all minima at various H
    '''
    M_par  = []
    thetas = []
    
    for H in H_vals:
        
        theta_eq = find_minimum(H, phi, args=args)
        thetas.append(theta_eq)
        # Component along the H direction H = cos(theta - phi)
        M_par.append(np.cos(theta_eq - phi))

    return np.array(M_par), np.array(thetas)

if __name__ == "__main__":
    #==============================================
    # Computational parameters
    #==============================================

    Ku    = 1.0       # Anisotropy constant
    Ms    = 1.0       # Saturation magnetization 
    mu0   = 1.0       # Permeability
    n     = 200       # Number of points in half cycle
    H_max = 5.0      # Maximum applied field
    phi   = np.pi/4       # Angle between field and easy axis (radians) [0 rad = parallel]
 
    H_forward  = np.linspace(H_max, -H_max, n)
    H_backward = np.linspace(-H_max, H_max, n)
    H_cycle    = np.concatenate([H_forward, H_backward])
  
    #==============================================
    # Compute hysteresis loop
    #==============================================

    M_cycle, theta_cycle = compute_hysteresis(H_cycle, phi, args=(Ku, mu0, Ms))
    M_cycle *= Ms  # Scale by saturation magnetization
    
    #==============================================
    # Plot results
    #==============================================
    
    plt.figure(1)
    plt.plot(H_cycle, M_cycle, 'b-')
    plt.plot(-H_cycle, -M_cycle, 'b-')

    plt.xlabel("Applied Field H")
    plt.ylabel("Magnetization M")
    plt.title(f"Hysteresis loop (phi = {phi:.2f} rad)")
    plt.grid()
    plt.show()
    