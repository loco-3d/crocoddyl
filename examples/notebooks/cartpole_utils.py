import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from math import sin, cos


def plotCartpole(x,patch=None):
    x_cart = xs[i][0]
    y_cart = 0.
    theta = xs[i][1]
    if patch is not None: patch.center = (x_cart, y_cart)

    x_pole = np.cumsum([x_cart,
                   -10 * 0.5 * sin(theta)])
    y_pole = np.cumsum([y_cart,
                   10 * 0.5 * cos(theta)])
    line.set_data(x_pole,y_pole)

def animateCartpole(xs,sleep=30):
    print("processing the animation ... ")
    fig = plt.figure()
    fig.set_dpi(100)
    fig.set_size_inches(7, 6.5)
    ax = plt.axes(xlim=(-9, 9), ylim=(-6, 6))
    patch = plt.Circle((5, -5), 0.25, fc='b')
    line, = ax.plot([], [], 'o-', lw=2)
    def init():
        patch.center = (0, 0)
        line.set_data([], [])
        ax.add_patch(patch)
        return patch,
    def animate(i):
        x_cart = xs[i][0]
        y_cart = 0.
        theta = xs[i][1]
        patch.center = (x_cart, y_cart)
        x_pole = np.cumsum([x_cart,
                            -10 * 0.5 * sin(theta)])
        y_pole = np.cumsum([y_cart,
                            10 * 0.5 * cos(theta)])
        line.set_data(x_pole,y_pole)
        return patch,
    anim = animation.FuncAnimation(fig, animate, 
                                   init_func=init,
                                   frames=len(xs), 
                                   interval=sleep,
                                   blit=True)
    print("... processing done")
    return anim
