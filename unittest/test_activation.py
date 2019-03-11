import numpy as np
from crocoddyl import ActivationModelQuad, ActivationModelSmoothAbs, ActivationModelWeightedQuad
from crocoddyl.utils import EPS
from numpy.linalg import norm

# Comment:
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

# - ------------------------------
# --- Dim 1 ----------------------
h = np.sqrt(EPS)


def df(am, ad, x):
    return (am.calc(ad, x + h) - am.calc(ad, x)) / h


def ddf(am, ad, x):
    return (am.calcDiff(ad, x + h)[0] - am.calcDiff(ad, x)[0]) / h


am = ActivationModelQuad()
ad = am.createData()
x = np.random.rand(1)

am.calc(ad, x)
assert (norm(df(am, ad, x) - am.calcDiff(ad, x)[0]) < 1e2 * h)
assert (norm(ddf(am, ad, x) - am.calcDiff(ad, x)[1]) < 1e2 * h)

am = ActivationModelWeightedQuad(np.random.rand(1))
ad = am.createData()
assert (norm(df(am, ad, x) - am.calcDiff(ad, x)[0]) < 1e2 * h)
assert (norm(ddf(am, ad, x) - am.calcDiff(ad, x)[1]) < 1e2 * h)

am = ActivationModelSmoothAbs()
ad = am.createData()

assert (norm(df(am, ad, x) - am.calcDiff(ad, x)[0]) < 1e2 * h)
assert (norm(ddf(am, ad, x) - am.calcDiff(ad, x)[1]) < 1e2 * h)

# - ------------------------------
# --- Dim N ----------------------


def df(am, ad, x):
    dx = x * 0
    J = np.zeros([len(x), len(x)])
    for i, _ in enumerate(x):
        dx[i] = h
        J[:, i] = (am.calc(ad, x + dx) - am.calc(ad, x)) / h
        dx[i] = 0
    return J


def ddf(am, ad, x):
    dx = x * 0
    J = np.zeros([len(x), len(x)])
    for i, _ in enumerate(x):
        dx[i] = h
        J[:, i] = (am.calcDiff(ad, x + dx)[0] - am.calcDiff(ad, x)[0]) / h
        dx[i] = 0
    return J
    return


x = np.random.rand(3)

am = ActivationModelQuad()
ad = am.createData()
J = df(am, ad, x)
H = ddf(am, ad, x)
assert (norm(np.diag(J.diagonal()) - J) < 1e2 * h)
assert (norm(np.diag(H.diagonal()) - H) < 1e2 * h)
assert (norm(df(am, ad, x).diagonal() - am.calcDiff(ad, x)[0]) < 1e2 * h)
assert (norm(ddf(am, ad, x).diagonal() - am.calcDiff(ad, x)[1][:, 0]) < 1e2 * h)

am = ActivationModelWeightedQuad(np.random.rand(len(x)))
ad = am.createData()
assert (norm(df(am, ad, x).diagonal() - am.calcDiff(ad, x)[0]) < 1e2 * h)
assert (norm(ddf(am, ad, x).diagonal() - am.calcDiff(ad, x)[1][:, 0]) < 1e2 * h)

am = ActivationModelSmoothAbs()
ad = am.createData()
assert (norm(df(am, ad, x).diagonal() - am.calcDiff(ad, x)[0]) < 1e2 * h)
assert (norm(ddf(am, ad, x).diagonal() - am.calcDiff(ad, x)[1][:, 0]) < 1e2 * h)
