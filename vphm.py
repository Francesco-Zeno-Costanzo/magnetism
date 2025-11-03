"""
Simple code to simulate magnetic hysteresis using Vector Preisach hysteresis model.
Based on the article:
D’Aloia, A.G.; Francesco, A.D.; Santis, V.D. A Novel Computational Method to Identify/Analyze
Hysteresis Loops of Hard Magnetic Materials. Magnetochemistry 2021, 7, 10. 
https://doi.org/10.3390/magnetochemistry7010010
"""
import numpy as np
import matplotlib.pyplot as plt


def AMB4(f, num_steps, t0, tf, init, args=()):
    '''
    Integrator with Adams-Bashforth-Moulton
    predictor and corrector of order 4

    Parameters
    ----------
    f : callable
        function to integrate, must accept vectorial input
    num_steps : int
        number of point of solution
    t0 : float
        lower bound of integration
    tf : float
        upper bound of integration
    init : 1darray
        array of initial condition
    args : tuple, optional
        extra arguments to pass to f

    Return
    ------
    X : array, shape (num_steps + 1, len(init))
        solution of equation
    t : 1darray
        time
    '''
    #time steps
    dt = (tf-t0)/num_steps

    X = np.zeros((num_steps + 1, len(init))) # Solution array
    t = np.zeros(num_steps + 1)              # Time array

    X[0, :] = init                           # Initial condition
    t[0]    = t0

    # First 3 steps with Runge-Kutta 4
    for i in range(3):
        xk1 = f(t[i], X[i, :], *args)
        xk2 = f(t[i] + dt/2, X[i, :] + xk1*dt/2, *args)
        xk3 = f(t[i] + dt/2, X[i, :] + xk2*dt/2, *args)
        xk4 = f(t[i] + dt, X[i, :] + xk3*dt, *args)
        X[i + 1, :] = X[i, :] + (dt/6)*(xk1 + 2*xk2 + 2*xk3 + xk4)
        t[i + 1]    = t[i] + dt

    # Adams-Bashforth-Moulton
    i = 3
    AB0 = f(t[i  ], X[i,   :], *args)
    AB1 = f(t[i-1], X[i-1, :], *args)
    AB2 = f(t[i-2], X[i-2, :], *args)
    AB3 = f(t[i-3], X[i-3, :], *args)

    for i in range(3,num_steps):
        # Prediction
        X[i + 1, :] = X[i, :] + dt/24*(55*AB0 - 59*AB1 + 37*AB2 - 9*AB3)
        t[i + 1]    = t[i] + dt
        # Correction
        AB3 = AB2
        AB2 = AB1
        AB1 = AB0
        AB0 = f(t[i+1], X[i + 1, :], *args)

        X[i + 1, :] = X[i, :] + dt/24*(9*AB0 + 19*AB1 - 5*AB2 + AB3)

    return X, t

def mag(omega_t, Y, alpha, beta, gamma):
    '''
    ODE for magnetization dynamics 
    '''
    
    mag, = Y

    mag_dot = gamma*np.sin(omega_t + beta)*np.exp(-alpha*np.cos(omega_t + beta)**2)
    Y_dot = np.array([mag_dot])

    return Y_dot

if __name__ == "__main__":

    #==============================================
    # Computational parameters
    #==============================================
    
    num_steps = 1000             # Number of steps
    t0        = 0                # Lower bound of integration
    tf        = 2*np.pi          # Upper bound of integration
    init      = np.array([0.0])  # Initial condition

    alpha     = 25.0             # Parameters
    beta      = 0.3              # of the 
    gamma     = 1.0              # model

    #==============================================
    # Compute hysteresis loop
    #==============================================

    sol, ts = AMB4(mag, num_steps, t0, tf, init, args=(alpha, beta, gamma))
    xs      = sol.T[0]
    # Normalize hysteresis loop to [-1, 1]
    norm_xs = (xs - xs.min())/(xs.max() - xs.min())*2 - 1
    
    #==============================================
    # Plot results
    #==============================================

    plt.plot(-np.cos(ts), norm_xs, 'b-')

    plt.xlabel("Applied Field H")
    plt.ylabel("Magnetization M")
    plt.grid()
    plt.show()