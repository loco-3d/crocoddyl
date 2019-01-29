import numpy as np
from numpy.linalg import norm,inv

from crocoddyl import ActivationDataQuad, ActivationModelQuad
from crocoddyl import ActivationDataWeightedQuad, ActivationModelWeightedQuad
'''
c = sum( a(ri) )
c' = sum( [a(ri)]' ) = sum( ri' a'(ri) ) = R' [ a'(ri) ]_i
c'' = R' [a'(ri) ]_i' = R' [a''(ri) ] R

ex 
a(x) =  x**2/x
a'(x) = x
a''(x) = 1

sum(a(ri)) = sum(ri**2/2) = .5*r'r
sum(ri' a'(ri)) = sum(ri' ri) = R' r
sum(ri' a''(ri) ri') = R' r
c'' = R'R
'''

from crocoddyl import ActivationDataSmoothAbs, ActivationModelSmoothAbs

# - ------------------------------
# --- Dim 1 ----------------------

def df(am,ad,x):
    h=1e-9
    return (am.calc(ad,x+h)-am.calc(ad,x))/h
def ddf(am,ad,x):
    h=1e-9
    return (am.calcDiff(ad,x+h)[0]-am.calcDiff(ad,x)[0])/h
    
am = ActivationModelQuad()
ad = am.createData()
x = np.random.rand(1)

am.calc(ad,x)
assert( norm(df(am,ad,x)-am.calcDiff(ad,x)[0]) < 1e-6 )
assert( norm(ddf(am,ad,x)-am.calcDiff(ad,x)[1]) < 1e-6 )

am = ActivationModelWeightedQuad(np.random.rand(1))
ad = am.createData()
assert( norm(df(am,ad,x)-am.calcDiff(ad,x)[0]) < 1e-6 )
assert( norm(ddf(am,ad,x)-am.calcDiff(ad,x)[1]) < 1e-6 )

am = ActivationModelSmoothAbs()
ad = am.createData()

assert( norm(df(am,ad,x)-am.calcDiff(ad,x)[0]) < 1e-6 )
assert( norm(ddf(am,ad,x)-am.calcDiff(ad,x)[1]) < 1e-6 )

# - ------------------------------
# --- Dim N ----------------------

def df(am,ad,x):
    h=1e-9
    dx = x*0
    J = np.zeros([ len(x),len(x) ])
    for i,_ in enumerate(x):
        dx[i] = h
        J[:,i] = (am.calc(ad,x+dx)-am.calc(ad,x))/h
        dx[i] = 0
    return J
def ddf(am,ad,x):
    h=1e-9
    dx = x*0
    J = np.zeros([ len(x),len(x) ])
    for i,_ in enumerate(x):
        dx[i] = h
        J[:,i] = (am.calcDiff(ad,x+dx)[0]-am.calcDiff(ad,x)[0])/h
        dx[i] = 0
    return J
    h=1e-9
    return 
    
x = np.random.rand(3)

am = ActivationModelQuad()
ad = am.createData()
J = df(am,ad,x)
H = ddf(am,ad,x)
assert( norm(np.diag(J.diagonal())-J)<1e-9 )
assert( norm(np.diag(H.diagonal())-H)<1e-9 )
assert( norm(df(am,ad,x).diagonal()-am.calcDiff(ad,x)[0]) < 1e-6 )
assert( norm(ddf(am,ad,x).diagonal()-am.calcDiff(ad,x)[1][:,0]) < 1e-6 )

am = ActivationModelWeightedQuad(np.random.rand(len(x)))
ad = am.createData()
assert( norm(df(am,ad,x).diagonal()-am.calcDiff(ad,x)[0]) < 1e-6 )
assert( norm(ddf(am,ad,x).diagonal()-am.calcDiff(ad,x)[1][:,0]) < 1e-6 )

am = ActivationModelSmoothAbs()
ad = am.createData()
assert( norm(df(am,ad,x).diagonal()-am.calcDiff(ad,x)[0]) < 1e-6 )
assert( norm(ddf(am,ad,x).diagonal()-am.calcDiff(ad,x)[1][:,0]) < 1e-6 )

