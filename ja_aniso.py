"""
Simple implementation of the Jiles-Atherton model for magnetic hysteresis
in anisotropic implementation
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

def solve(t0, tf, h_t, derivs, init, tau, args=()):
    '''
    Integrator Cash-Karp-Runga-Kutta Method

    Parameters
    ----------
    t0 : float
        lower bound of integration
    tf : float
        upper bound of integration
    h_t : float
        guess for integration steps
    init : 1darray
        array of initial condition
    tau : float
        required accuracy
    derivs : callable
        function to integrate, must accept vectorial input
    args : tuple, optional
        extra arguments to pass to derivs

    Return
    ------
    Y : array, shape (iter, len(init))
        solution of equation
    t : 1darray
        time
    h : 1darray
        array of steps
    '''

    #Useful function

    #Adaptive stepsize
    def rkqs(y, dydx, x, htry, eps, yscal, derivs, args=()):
        '''
        function to controll Adaptive stepsize

        Parameters
        ----------
        y : 1darray
            solution at point x
        dydx : 1darray
            array for RHS of equation
        x : float
            actual point
        htry : float
            guess for integration steps
        eps : float
            required accuracy
        yscal : 1darray
            array to controll the accuracy
        derivs : callable
            function to integrate, must accept vectorial input
        args : tuple, optional
            extra arguments to pass to derivs

        Return
        ------
        x : float
            new point fo solution
        y : 1darray
            solution at point x
        hdid : float
            value of step used
        hnext : float
            prediction of next stepsize
        '''

        SAFETY  = 0.9
        PGROW   = -0.2
        PSHRINK = -0.25
        ERRCON  = 1.89e-4 #(5/SAFETY)**(1/PGROW)

        h = htry

        while True:
            ytemp, yerr = rkck(y, dydx, x, h, derivs, args)
            errarr = [abs(yerr[i]/yscal[i]) for i in range(len(yerr))]
            errmax = np.max(errarr)
            errmax /= eps
            if errmax <= 1.0: #Step was succesful, time for next step!
                break

            htemp = SAFETY * h * errmax**PSHRINK #reducing step size, try again
            if h >= 0.0:
                h = max(htemp, 0.1*h)
            else:
                h = min(htemp, 0.1*h)

            xnew = x+h
            if xnew == x:
                print("Stepsize underflow in rkqs")
                continue
        
        if errmax > ERRCON:
            hnext = SAFETY * h * errmax**PGROW
        else:
            hnext = h*5

        hdid = h
        x += hdid
        y = ytemp

        return x, y, hdid, hnext


    #Cash-Karp Runge-Kutta step
    def rkck(y, dydx, x, h, f, args=()):
        '''
        function for integration of quation

        Parameters
        ----------
        y : 1darray
            solution at point x
        dydx : 1darray
            array for RHS of equation
        x : float
            actual point
        h : float
            integration step
        f : callable
            function to integrate, must accept vectorial input
        args : tuple, optional
            extra arguments to pass to f

        Return
        ------
        yout : 1darray
            solution
        yerr : 1darray
            error
        '''
        # For temporal part of f
        A = np.array([0, 0.2, 0.3, 0.6, 1.0, 0.875])

        # For solution part of f
        B = np.array([
            [0,          0,       0,         0,            0,        0],
            [0.2,        0,       0,         0,            0,        0],
            [3/40,       9/40,    0,         0,            0,        0],
            [0.3,       -0.9,     1.2,       0,            0,        0],
            [-11/54,     2.5,    -70/27,     35/27,        0,        0],
            [1631/55296, 175/512, 575/13824, 44275/110592, 253/4096, 0]
        ])

        # For update the solutions ad truncation error
        C = np.array([37/378,     0, 250/621,     125/594,     0,         512/1771])
        D = np.array([2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 0.25])
        DC = C - D

        # Step of algorithm
        k    = np.zeros((6, len(y)))
        k[0] = dydx

        for j in range(1, 6):
            y_temp = y + h * sum(B[j, l] * k[l] for l in range(j))
            k[j] = f(x + A[j] * h, y_temp, *args)
        
        # Final solution, accumulate increments with proper weights
        yout = y + h * np.dot(C, k)

        # Estimate error as difference between fourth and fifth order methods
        yerr = h * np.dot(DC, k)

        return yout, yerr

    # Integration

    TINY = 1e-30                      # To avoid division by zero
    Y = []                            # To store solution
    t = []                            # To store time
    H = []                            # To store integration steps
    x = t0                            # Initial point of integration
    h = h_t*(tf - t0)/abs(tf - t0)    # Initial step (tf - t0)/abs(tf - t0) cab be +-1
    y = init                          # Initial condition

    Y.append(y)                       # Store the first point
    t.append(x)                       # Store the first time
    H.append(h)                       # Store the first step
    iter = 0                          # To count iterations


    while x < tf or x > tf:

        dydx = derivs(x, y, *args)

        yscal = abs(y) + abs(h*dydx) + TINY
        if (x+h-tf)*(x+h-t0) > 0 : h = tf-x # If stepsize can overshoot, decrease

        x, ytemp, hdid, hnext = rkqs(y, dydx, x, h, tau, yscal, derivs, args)
        H.append(hdid)     # Store the steps used
        y = ytemp          # Update the solution
        h = hnext          # Update step (sometimes hdid?? why??)

        Y.append(y)        # Store solution
        t.append(x)        # Store time
        iter += 1          # Update iteration

    print(f'Number of iterartions = {iter}')

    return np.array(Y), np.array(t), np.array(H)


#============================================================
#  Jiles-Atherton Model: Anhysteretic magnetization
#============================================================

def anhysteretic_magnetization_anisotropic(H_eff, Ms, a, Ku, phi, w):
    """
    Calcola la magnetizzazione anhisteretica con anisotropia uniaxiale.
    
    Parameters
    ----------
    Parameters
    ----------
    H_eff : float or array_like
        Effective magnetic field H + alpha*M.
    Ms : float
        Saturation magnetization.
    a : float
        Anhysteretic parameter controlling the curvature of M_an.
    Ku : float
        Uniaxial anisotropy constant.
    phi : float
        Angle of applied field with respect to easy axis (radians).
    w : float
        Weighting factor between isotropic and anisotropic contributions.

    Returns
    -------
    M_an : float or array_like
        Anhysteretic magnetization value.
    """
    #### Analytic solution for isotropic case ####
    H = np.asarray(H_eff, dtype=float)
    x = H / a
    M_an_iso = np.zeros_like(H, dtype=float)

    small = np.abs(x) < 1e-6
    if np.any(~small):
        xm = x[~small]
        M_an_iso[~small] = Ms * (1 / np.tanh(xm) - 1 / xm)
    if np.any(small):
        xs = x[small]
        M_an_iso[small] = Ms * (xs / 3 - xs**3 / 45 + 2 * xs**5 / 945)
   

    #### Numerical integration for anisotropic case ####
    E_1 = lambda theta: x * np.cos(theta) - (Ku / (Ms * a)) * np.sin(phi - theta)**2
    E_2 = lambda theta: x * np.cos(theta) - (Ku / (Ms * a)) * np.sin(phi + theta)**2
    F_1 = lambda theta: np.exp( (E_1(theta) + E_2(theta))/2) * np.sin(theta)
    F_2 = lambda theta: np.exp( (E_1(theta) + E_2(theta))/2) * np.sin(theta) * np.cos(theta)

    num = quad(F_2, 0, np.pi)[0]
    den = quad(F_1, 0, np.pi)[0]

    M_an = w * M_an_iso + (1 - w) * Ms * num / den

    return M_an


# ============================================================
#  Differential equation for Jiles-Atherton model
# ============================================================

def dM_dH(H, M, params, delta):
    """
    Compute dM/dH for the Jiles-Atherton model.

    Parameters
    ----------
    H : float
        Applied magnetic field.
    M : float
        Current magnetization.
    params : dict
        Dictionary containing model parameters:
            Ms, a, k, c, alpha.
    delta : int
        +1 when H is increasing, -1 when decreasing.

    Returns
    -------
    dM_dH : float
        Derivative of M with respect to H.
    """
    Ms    = params['Ms']
    a     = params['a']
    k     = params['k']
    c     = params['c']
    alpha = params['alpha']
    Ku    = params['Ku']
    phi   = params['phi']
    w     = params['w']

    H_eff = H + alpha * M
    M_an  = anhysteretic_magnetization_anisotropic(H_eff, Ms, a, Ku, phi, w)

    # Analytic derivative isotropic part dM_an/dH_eff = (Ms / a) * (1/x^2 - csch^2(x))
    # Use series for small x to avoid catastrophic cancellation.
    x = H_eff / a
    abs_x = abs(x)
    if abs_x < 1e-6:
        dM_an_dHeff_iso = (Ms / a) * (1.0/3.0 - x**2 / 15.0 + 2.0 * x**4 / 189.0)
    else:
        # compute csch^2(x) = 1 / sinh^2(x) safely
        csch2 = 1.0 / (np.sinh(x)**2)
        dM_an_dHeff_iso = (Ms / a) * (1.0 / x**2 - csch2)
    
    # Numerical derivative anisotropic part
    dH = 1e-6 * max(abs(H_eff), a)
    

    # M_an at H_eff + dH
    H_eff_plus  = H_eff + dH
    x_plus      = H_eff_plus / a
    E_1 = lambda theta: x_plus * np.cos(theta) - (Ku / (Ms * a)) * np.sin(phi - theta)**2
    E_2 = lambda theta: x_plus * np.cos(theta) - (Ku / (Ms * a)) * np.sin(phi + theta)**2
    F_1 = lambda theta: np.exp( (E_1(theta) + E_2(theta))/2) * np.sin(theta)
    F_2 = lambda theta: np.exp( (E_1(theta) + E_2(theta))/2) * np.sin(theta) * np.cos(theta)
    num_plus    = quad(F_2, 0, np.pi)[0]
    den_plus    = quad(F_1, 0, np.pi)[0]
    M_an_plus   = Ms * num_plus / den_plus 
    # M_an at H_eff - dH
    H_eff_minus = H_eff - dH
    x_minus     = H_eff_minus / a
    E_1 = lambda theta: x_minus * np.cos(theta) - (Ku / (Ms * a)) * np.sin(phi - theta)**2
    E_2 = lambda theta: x_minus * np.cos(theta) - (Ku / (Ms * a)) * np.sin(phi + theta)**2
    F_1 = lambda theta: np.exp( (E_1(theta) + E_2(theta))/2) * np.sin(theta)
    F_2 = lambda theta: np.exp( (E_1(theta) + E_2(theta))/2) * np.sin(theta) * np.cos(theta)
    num_minus   = quad(F_2, 0, np.pi)[0]
    den_minus   = quad(F_1, 0, np.pi)[0]
    M_an_minus  = Ms * num_minus / den_minus  
    
    dM_an_dHeff_aniso = (M_an_plus - M_an_minus) / (2 * dH)


    # Put derivative together
    dM_an_dHeff = w * dM_an_dHeff_iso + (1 - w) * dM_an_dHeff_aniso 

    denom = k * delta - alpha * (M_an - M)
    if abs(denom) < 1e-12:
        denom = np.sign(denom) * 1e-12 if denom != 0 else 1e-12

    dM_dH = c * (dM_an_dHeff) + (1 - c) * (M_an - M) / denom
    return dM_dH


# ============================================================
#  Integration using the provided adaptive RK Cash–Karp solver
# ============================================================

def jiles_atherton_integrate(H_in_up, H_up, H_down, params, M_init=0, h_t=1e-3, tau=1e-10):
    """
    Integrate the Jiles-Atherton differential equation using an adaptive

    Parameters
    ----------
    H_in_up : list
        Sequence of applied magnetic field values for initial increasing branch.
    H_up : list
        Sequence of applied magnetic field values for increasing branch.
    H_down : list
        Sequence of applied magnetic field values for decreasing branch.
    params : dict
        Model parameters (Ms, a, k, c, alpha).
    M_init : float, optional
        Initial magnetization value (default is 0).
    h_t : float, optional
        Initial guess for integration step size (default is 1e-3).
    tau : float, optional
        Required accuracy for the integrator (default is 1e-10). 
   
    Returns
    -------
    H_points : 1darray
        Magnetic field values used during integration.
    M_points : 1darray
        Corresponding magnetization values.
    """
    # Increasing branch
    Y_in_up, T_in_up, _ = solve(H_in_up[0], H_in_up[-1], h_t, dM_dH, np.array([M_init]), tau, args=(params, +1))
    M_in_up             = Y_in_up[:, 0]

    #plt.plot(_)

    # Use last magnetization as initial condition for decreasing branch
    M0_down           = np.array([M_in_up[-1]])
    Y_down, T_down, _ = solve(H_down[0], H_down[-1], h_t, dM_dH, M0_down, tau, args=(params, -1))
    M_down            = Y_down[:, 0]
    #plt.plot(_)

    # Use last magnetization of decreasing branch as initial condition for increasing branch
    M0_up         = np.array([M_down[-1]])
    Y_up, T_up, _ = solve(H_up[0], H_up[-1], h_t, dM_dH, M0_up, tau, args=(params, +1))
    M_up          = Y_up[:, 0]
    #plt.plot(_)
    #plt.show()

    # Concatenate all branches (avoid duplicating)
    H_full = np.concatenate((T_in_up, T_down[1:], T_up[1:]))
    M_full = np.concatenate((M_in_up, M_down[1:], M_up[1:]))

    return H_full, M_full


if __name__ == "__main__":
    #==============================================
    # Computational parameters
    #==============================================
    params = {
        'Ms': 1.0,       # Saturation magnetization
        'a': 0.8,        # Quantifies domain walls density in the magnetic material 
        'k': 2.9,        # Quantifies average energy required to break pinning site in the magnetic material
        'c': 0.4,        # Magnetization reversibility coefficient
        'alpha': 0.01,   # Quantifies interdomain coupling in the magnetic material 
        'Ku': 1.0,       # Uniaxial anisotropy constant
        'phi': np.pi/4,  # Angle between applied field and easy axis (radians)
        'w': 0.5         # Weighting factor between isotropic and anisotropic contributions
    }

    H_max   = 30         # Maximum applied field
    H_in_up = [0, H_max]
    H_up    = [-H_max, H_max]
    H_down  = [H_max, -H_max]

    #==============================================
    # Compute hysteresis loop
    #==============================================

    H, M = jiles_atherton_integrate(H_in_up, H_up, H_down, params, tau=1e-8)

    #==============================================
    # Plot results
    #==============================================

    plt.figure(figsize=(8, 6))
    plt.plot(H, M)
    plt.xlabel('Applied field H')
    plt.ylabel('Magnetization M')
    
    plt.grid()
    plt.tight_layout()
    plt.show()
