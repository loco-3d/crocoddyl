# flake8: noqa: F821
def cartpole_analytical_derivatives(model, data, x, u=None):
    if u is None:
        u = model.unone

    # Getting the state and control variables
    y, th, ydot, thdot = x[0].item(), x[1].item(), x[2].item(), x[3].item()
    f = u[0].item()

    # Shortname for system parameters
    m1, m2, lcart, g = model.m1, model.m2, model.l, model.g
    s, c = np.sin(th), np.cos(th)
    m = m1 + m2
    mu = m1 + m2 * s**2
    w = model.costWeights

    # derivative of xddot by x, theta, xdot, thetadot
    # derivative of thddot by x, theta, xdot, thetadot
    data.Fx[:, :] = np.array(
        [
            [
                0.0,
                (m2 * g * c * c - m2 * g * s * s - m2 * lcart * c * thdot) / mu,
                0.0,
                -m2 * lcart * s / mu,
            ],
            [
                0.0,
                (
                    (-s * f / lcart)
                    + (m * g * c / lcart)
                    - (m2 * c * c * thdot**2)
                    + (m2 * s * s * thdot**2)
                )
                / mu,
                0.0,
                -2 * m2 * c * s * thdot,
            ],
        ]
    )
    # derivative of xddot and thddot by f
    data.Fu[:] = np.array([1 / mu, c / (lcart * mu)])
    # first derivative of data.cost by x, theta, xdot, thetadot
    data.Lx[:] = np.array(
        [
            y * w[2] ** 2,
            s * ((w[0] ** 2 - w[1] ** 2) * c + w[1] ** 2),
            ydot * w[3] ** 2,
            thdot * w[4] ** 2,
        ]
    )
    # first derivative of data.cost by f
    data.Lu[:] = np.array([f * w[5] ** 2])
    # second derivative of data.cost by x, theta, xdot, thetadot
    data.Lxx[:] = np.array(
        [
            w[2] ** 2,
            w[0] ** 2 * (c**2 - s**2) + w[1] ** 2 * (s**2 - c**2 + c),
            w[3] ** 2,
            w[4] ** 2,
        ]
    )
    # second derivative of data.cost by f
    data.Luu[:] = np.array([w[5] ** 2])
