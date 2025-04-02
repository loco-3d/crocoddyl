import matplotlib.pyplot as plt
import numpy as np


def plotUnicycle(x):
    sc, delta = 0.1, 0.1
    a, b, th = x[0], x[1], x[2]
    c, s = np.cos(th), np.sin(th)
    refs = [
        plt.arrow(
            a - sc / 2 * c - delta * s,
            b - sc / 2 * s + delta * c,
            c * sc,
            s * sc,
            head_width=0.05,
        ),
        plt.arrow(
            a - sc / 2 * c + delta * s,
            b - sc / 2 * s - delta * c,
            c * sc,
            s * sc,
            head_width=0.05,
        ),
    ]
    return refs


def plotUnicycleSolution(xs, figIndex=1, show=True):
    plt.figure(figIndex, figsize=(6.4, 6.4))
    for x in xs:
        plotUnicycle(x)
    plt.axis([-2, 2.0, -2.0, 2.0])
    if show:
        plt.show()
