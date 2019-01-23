import numpy as np

'''
Numpy convention.
Let's store vector as 1-d array and matrices as 2-d arrays. Multiplication is done by np.dot.
'''
def raiseIfNan(A,error=None):
    if error is None: error = scl.LinAlgError("NaN in array")
    if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(abs(np.asarray(A))>1e30):
        raise error


def m2a(m): return np.array(m.flat)


def a2m(a): return np.matrix(a).T


def absmax(A): return np.max(abs(A))


def absmin(A): return np.min(abs(A))


def plotDDPConvergence(J, muLM, muV, gamma, theta, alpha):
    import matplotlib.pyplot as plt

    plt.figure(1, figsize=(6.4, 8))
    # Plotting the total cost sequence
    plt.subplot(511)
    plt.ylabel('cost')
    plt.plot(J)

    # Ploting mu sequences
    plt.subplot(512)
    plt.ylabel('mu')
    plt.plot(muLM, label='LM')
    plt.plot(muV, label='V')
    plt.legend()

    # Plotting the gradient sequence (gamma and theta)
    plt.subplot(513)
    plt.ylabel('gamma')
    plt.plot(gamma)
    plt.subplot(514)
    plt.ylabel('theta')
    plt.plot(theta)

    # Plotting the alpha sequence
    plt.subplot(515)
    plt.ylabel('alpha')
    ind = np.arange(len(alpha))
    plt.bar(ind, alpha)
    plt.xlabel('iteration')
    plt.show()


def plotOCSolution(xs, us):
    import matplotlib.pyplot as plt
    # Getting the state and control trajectories
    nx = xs[0].shape[0]
    nu = us[0].shape[0]
    X = [0.] * nx
    U = [0.] * nu
    for i in range(nx):
        X[i] = [ x[i] for x in xs]
    for i in range(nu):
        U[i] = [ u[i] for u in us]

    plt.figure(1)

    # Plotting the state trajectories
    plt.subplot(211)
    [plt.plot(X[i], label='x'+str(i)) for i in range(nx)]
    plt.legend()

    # Plotting the control commands
    plt.subplot(212)
    [plt.plot(U[i], label='u'+str(i)) for i in range(nu)]
    plt.legend()
    plt.xlabel('knots')
    plt.show()
