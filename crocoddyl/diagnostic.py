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