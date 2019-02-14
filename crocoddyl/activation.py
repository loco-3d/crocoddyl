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

class ActivationModelInequality:
    """
    The activation starts from zero when r approaches b_l=lower or b_u=upper.
    The activation is zero when r is between b_l and b_u
    beta determines how much of the total acceptable range is not activated (default 90%)
    
    a(r) = (r**2)/2 for r>b_u.
    a(r) = (r**2)/2 for r<b_l.
    a(r) = 0. for b_l<=r<=b_u
    """
    def __init__(self, lowerLimit, upperLimit, beta=None):
        assert((lowerLimit<=upperLimit).all())
        assert(not np.any(np.isinf(lowerLimit)) and not np.any(np.isinf(upperLimit)) or beta==None)
        if beta is None:
            self.lower = lowerLimit;   self.upper = upperLimit
        else:
            assert(beta>0 and beta<1)
            self.beta=beta
            m = (lowerLimit+upperLimit)/2
            d = (upperLimit-lowerLimit)/2
            self.lower = m-beta*d
            self.upper = m+beta*d
      
    def calc(model,data,r):
        '''Return [ a(r_1) ... a(r_n) ] '''
        return np.minimum(r-model.lower, 0)**2/2 + np.maximum(r-model.upper, 0)**2/2
    def calcDiff(model,data,r,recalc=True):
        if recalc: model.calc(data,r)
        ''' 
        Return [ a'(r_1) ... a'(r_n) ], diag([ a''(r_1) ... a''(r_n) ])
        '''
        return np.minimum(r-model.lower, 0.) + np.maximum(r-model.upper, 0),\
          ((r-model.upper>=0.) + (r-model.lower<=0.)).astype(float)[:,None]

    def createData(self): return ActivationDataInequality(self)

class ActivationDataInequality:
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
    def createData(self): return ActivationDataSmoothAbs(self)

class ActivationDataSmoothAbs:
    def __init__(self,model):
        pass
