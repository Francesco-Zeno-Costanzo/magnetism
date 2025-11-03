"""
Simple code to simulate Stoner-Wohlfarth hysteresis loops for an
ensemble of uniaxial single-domain non interacting particles.
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
        Magnetization of one domain
    '''
    M_par  = []
    thetas = []
    
    for H in H_vals:
        
        theta_eq = find_minimum(H, phi, args=args)
        thetas.append(theta_eq)
        # Component along the H direction H = Ms * cos(theta - phi)
        M_par.append(np.cos(theta_eq - phi))

    return np.array(M_par)


def compute_ensemble_hysteresis(H_vals, phis, Kus, Ms_vals, mu0):
    '''
    Compute hysteresis for an ensemble of non-interacting uniaxial domains.

    Parameters
    ----------
    H_vals : array
        External field values.
    phis : array
        Array of easy-axis orientations (radians).
    Kus : array
        Anisotropy constants for each domain.
    Ms_vals : array
        Saturation magnetizations for each domain.
    mu0 : float
        Permeability constant.

    Return
    ------
    M_total : 1darray
        Magnetization of all ensamble
    '''
    N = len(phis)
    M_total = np.zeros_like(H_vals, dtype=float)

    for i in range(N):
        M_i = compute_hysteresis(H_vals, phis[i], args=(Kus[i], mu0, Ms_vals[i]))
        M_total += M_i*Ms_vals[i]

    return M_total / N  # average magnetization

if __name__ == "__main__":

    np.random.seed(69420)  # for reproducibility

    #==============================================
    # Computational parameters
    #==============================================

    mu0   = 1.0  # Permeability
    n     = 200  # Number of points in half cycle
    H_max = 5.0  # Maximum applied field
    N_d   = 100  # Number of domains in ensemble
 
    H_forward  = np.linspace(H_max, -H_max, n)
    H_backward = np.linspace(-H_max, H_max, n)
    H_cycle    = np.concatenate([H_forward, H_backward])
   
    # Random distribution of anisotropy axes
    phis    = np.random.uniform(0, np.pi/2, N_d)  # uniformly in [0, π/2]
    Kus     = np.ones(N_d) * 1.0                  # identical Ku
    Ms_vals = np.ones(N_d) * 1.0                  # identical Ms
    #Kus     = np.random.normal(1.0, 0.1, N_d)
    #Ms_vals = np.random.normal(1.0, 0.1, N_d)


    #==============================================
    # Compute hysteresis loop
    #==============================================

    M_cycle = compute_ensemble_hysteresis(H_cycle, phis, Kus, Ms_vals, mu0)

    #==============================================
    # Plot results
    #==============================================

    plt.figure(1)
    plt.plot(H_cycle, M_cycle, 'b-')
    plt.plot(-H_cycle, -M_cycle, 'b-')

    plt.xlabel("Applied Field H")
    plt.ylabel("Magnetization M")
    plt.title("Stoner-Wohlfarth ensemble (uniaxial, non-interacting)")
    plt.grid()
    plt.show()
