import numpy as np


class ActivationModelAbstract:
    """ Abstract class for activation models.

    In crocoddyl, an activation model takes the residual vector and computes
    the activation value and its derivatives from it. Activation value and
    its derivatives are computed by calc() and calcDiff(), respectively.
    """

    def __init__(self):
        self.ActivationDataType = ActivationDataAbstract

    def createData(self):
        """ Create the activation data

        :param model: activation model
        :param data: activation data
        :param r: residual vector
        """
        return self.ActivationDataType(self)

    def calc(model, data, r):
        """ Compute and return the activation value.

        :param model: activation model
        :param data: activation data
        :param r: residual vector
        :return the activation value
        """
        raise NotImplementedError("Not implemented yet.")

    def calcDiff(model, data, r):
        """ Compute the Jacobian of the activation model.

        :param model: activation model
        :param data: activation data
        :param r: residual vector
        :return residual vector and the Jacobian of the activation model
        """
        raise NotImplementedError("Not implemented yet.")


class ActivationDataAbstract:
    def __init(self, model):
        pass


class ActivationModelQuad(ActivationModelAbstract):
    def __init__(self):
        self.ActivationDataType = ActivationDataQuad

    def calc(self, data, r):
        '''Return [ a(r_1) ... a(r_n) ] '''
        return r**2 / 2

    def calcDiff(self, data, r, recalc=True):
        if recalc:
            self.calc(data, r)
        '''
        Return [ a'(r_1) ... a'(r_n) ], diag([ a''(r_1) ... a''(r_n) ])
        '''
        return r, np.array([1.] * len(r))[:, None]


class ActivationDataQuad(ActivationDataAbstract):
    def __init__(self, model):
        pass


class ActivationModelInequality(ActivationModelAbstract):
    """
    The activation starts from zero when r approaches b_l=lower or b_u=upper.
    The activation is zero when r is between b_l and b_u
    beta determines how much of the total acceptable range is not activated (default 90%)

    a(r) = (r**2)/2 for r>b_u.
    a(r) = (r**2)/2 for r<b_l.
    a(r) = 0. for b_l<=r<=b_u
    """

    def __init__(self, lowerLimit, upperLimit, beta=None):
        assert ((lowerLimit <= upperLimit).all())
        assert (not np.any(np.isinf(lowerLimit)) and not np.any(np.isinf(upperLimit)) or beta is None)
        self.ActivationDataType = ActivationDataInequality
        if beta is None:
            self.lower = lowerLimit
            self.upper = upperLimit
        else:
            assert (beta > 0 and beta <= 1)
            self.beta = beta
            m = (lowerLimit + upperLimit) / 2
            d = (upperLimit - lowerLimit) / 2
            self.lower = m - beta * d
            self.upper = m + beta * d

    def calc(self, data, r):
        '''Return [ a(r_1) ... a(r_n) ] '''
        # return np.minimum(r - self.lower, 0)**2 / 2 + np.maximum(r - self.upper, 0)**2 / 2
        return np.minimum(r - self.lower, 0)**2 / 2 + np.maximum(r - self.upper, 0)**2 / 2

    def calcDiff(self, data, r, recalc=True):
        if recalc:
            self.calc(data, r)
        '''
        Return [ a'(r_1) ... a'(r_n) ], diag([ a''(r_1) ... a''(r_n) ])
        '''
        return np.minimum(r - self.lower, 0.) + np.maximum(r - self.upper, 0), (
            (r - self.upper >= 0.) + (r - self.lower <= 0.)).astype(float)[:, None]

    def createData(self):
        return ActivationDataInequality(self)


class ActivationDataInequality(ActivationDataAbstract):
    def __init__(self, model):
        pass


class ActivationModelInequalityCont(ActivationModelAbstract):

    def __init__(self, lowerLimit, upperLimit, beta=None):
        assert ((lowerLimit <= upperLimit).all())
        assert (not np.any(np.isinf(lowerLimit)) and not np.any(np.isinf(upperLimit)) or beta is None)
        self.ActivationDataType = ActivationDataInequalityCont

        if beta is None:
            self.beta = 0
            self.lower = lowerLimit
            self.upper = upperLimit
        else:
            assert (beta > 0 and beta <= 1)
            self.beta = beta

        self.m = (lowerLimit + upperLimit) / 2
        self.d = (upperLimit - lowerLimit) / 2
        self.lower = self.m - self.beta * self.d
        self.upper = self.m + self.beta * self.d

    def calc(self, data, r):
        '''Return [ a(r_1) ... a(r_n) ] '''
        return ((r - self.m) / self.d)**6

    def calcDiff(self, data, r, recalc=True):
        if recalc:
            self.calc(data, r)

        return 6 * ((r - self.m) / self.d)**5, (30 * ((r - self.m) / self.d)**4)[:, None]

    def createData(self):
        return ActivationDataInequalityCont(self)


class ActivationDataInequalityCont(ActivationDataAbstract):
    def __init__(self, model):
        pass


class ActivationModelWeightedQuad(ActivationModelAbstract):
    def __init__(self, weights):
        self.weights = weights
        self.ActivationDataType = ActivationDataWeightedQuad

    def calc(self, data, r):
        return self.weights * r**2 / 2

    def calcDiff(self, data, r, recalc=True):
        if recalc:
            self.calc(data, r)
        assert (len(self.weights) == len(r))
        return self.weights * r, self.weights[:, None]


class ActivationDataWeightedQuad(ActivationDataAbstract):
    def __init__(self, model):
        pass


class ActivationModelSmoothAbs(ActivationModelAbstract):
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
        self.ActivationDataType = ActivationDataSmoothAbs
        pass

    def calc(self, data, r):
        data.a = np.sqrt(1 + r**2)
        return data.a

    def calcDiff(self, data, r, recalc=True):
        if recalc:
            self.calc(data, r)
        return r / data.a, (1 / data.a**3)[:, None]


class ActivationDataSmoothAbs(ActivationDataAbstract):
    def __init__(self, model):
        pass
