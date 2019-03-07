from math import cos, sin

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt


def animateCartpole(xs,sleep=50):
    print("processing the animation ... ")
    cart_size = 1.
    pole_length = 5.
    fig = plt.figure()
    ax = plt.axes(xlim=(-8, 8), ylim=(-6, 6))
    patch = plt.Rectangle((0., 0.), cart_size, cart_size, fc='b')
    line, = ax.plot([], [], 'k-', lw=2)
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    def init():
        ax.add_patch(patch)
        line.set_data([], [])
        time_text.set_text('')
        return patch, line, time_text
    def animate(i):
        x_cart = xs[i][0]
        y_cart = 0.
        theta = xs[i][1]
        patch.set_xy([x_cart-cart_size/2, y_cart-cart_size/2])
        x_pole = np.cumsum([x_cart,
                            -pole_length * sin(theta)])
        y_pole = np.cumsum([y_cart,
                            pole_length * cos(theta)])
        line.set_data(x_pole,y_pole)
        time = i*sleep/1000.
        time_text.set_text('time = %.1f sec' % time)
        return patch, line, time_text
    anim = animation.FuncAnimation(fig, animate, 
                                   init_func=init,
                                   frames=len(xs), 
                                   interval=sleep,
                                   blit=True)
    print("... processing done")
    plt.show()
    return anim
