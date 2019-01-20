import numpy as np
from numpy.linalg import norm,inv

class ActivationModelQuad:
    def __init__(self):
        pass
    def calc(model,data,r):
        '''Return [ a(r_1) ... a(r_n) ] '''
        return r**2/2
    def calcDiff(model,data,r,recalc=True):
        if recalc: model.calc(data,r)
        ''' 
        Return [ a'(r_1) ... a'(r_n) ], diag([ a''(r_1) ... a''(r_n) ])
        '''
        return r,np.array([1.]*len(r))[:,None]
    def createData(self): return ActivationDataQuad(self)
class ActivationDataQuad:
    def __init__(self,model):
        pass

class ActivationModelWeightedQuad:
    def __init__(self,weights):
        self.weights=weights
    def calc(model,data,r):
        return model.weights*r**2/2
    def calcDiff(model,data,r,recalc=True):
        if recalc: model.calc(data,r)
        assert(len(model.weights)==len(r))
        return model.weights*r,model.weights[:,None]
    def createData(self): return ActivationDataWeightedQuad(self)
class ActivationDataWeightedQuad:
    def __init__(self,model):
        pass

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
    
class ActivationModelSmoothAbs:
    '''
    f(x) = s(1+x**2) ; f' = x/s ; f'' = (u'v-uv')/vv = (s-xx/s)/ss = (ss-xx)/sss
    c = sqrt(1+r**2) 
    c' = r'r / s
    (1/s)' = -s'/s**2 = -r'r/s**2
    c'' =   r'' r / s + r'r' / s + r'r (1/s)' 

    a = sqrt(1+x**2)
    a' = x /  sqrt(1+x**2) = x/a
    a'' = (u'v-uv')/v**2 = (a-xa')/a**2 = (a-x**2/a)/a**2 = (a**2-x**2)/a**3
        = (1+x**2-x**2) / a**3 = 1/a**3 

    c = sum ( sqrt(1 + ri**2) )= sum( a(ri) )
    c' = sum ( ri' a'(ri) ) = R' [ x/a ]
    c'' = R' [ a''(ri) ] R
        = R' [ 1/a**3 ] R

    '''
    def __init__(self):
        pass
    def calc(model,data,r):
        data.a = np.sqrt(1+r**2)
        return data.a
    def calcDiff(model,data,r,recalc=True):
        if recalc: model.calc(data,r)
        return r/data.a,(1/data.a**3)[:,None]
    def createData(self): return ActivationDataWeightedQuad(self)
class ActivationDataWeightedQuad:
    def __init__(self,model):
        pass

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

