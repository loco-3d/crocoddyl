import matplotlib.pyplot as plt
import numpy as np

def plotUnicycle(x):
    sc,delta = .1,.1
    a,b,th = x[:4]
    c,s = np.cos(th),np.sin(th)
    refs = [
        plt.arrow(a-sc/2*c-delta*s,b-sc/2*s+delta*c,c*sc,s*sc,head_width=.05),
        plt.arrow(a-sc/2*c+delta*s,b-sc/2*s-delta*c,c*sc,s*sc,head_width=.05)
        ]
    return refs

