import matplotlib.pyplot as plt
import numpy as np


def plotUnicycle(x):
    sc, delta = .1, .1
    a, b, th = x[0].item(), x[1].item(), x[2].item()
    c, s = np.cos(th), np.sin(th)
    refs = [
        plt.arrow(a - sc / 2 * c - delta * s, b - sc / 2 * s + delta * c, c * sc, s * sc, head_width=.05),
        plt.arrow(a - sc / 2 * c + delta * s, b - sc / 2 * s - delta * c, c * sc, s * sc, head_width=.05)
    ]
    return refs


def plotUnicycleSolution(xs, figIndex=1, show=True):
    plt.figure(figIndex, figsize=(6.4, 6.4))
    for x in xs:
        plotUnicycle(x)
    plt.axis([-2, 2., -2., 2.])
    if show:
        plt.show()
