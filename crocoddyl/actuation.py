import warnings

import numpy as np

class ActuationModelUAM:
    '''
    This model transforms an actuation u into a joint torque tau.
    We implement here the simplest model: tau = S.T*u, where S is constant.
    '''

    def __init__(self, pinocchioModel, rotorDistance, coefM, coefF):
        self.pinocchio = pinocchioModel
        if (pinocchioModel.joints[1].shortname() != 'JointModelFreeFlyer'):
            warnings.warn('Strange that the first joint is not a freeflyer')
        self.nq = pinocchioModel.nq
        self.nv = pinocchioModel.nv
        self.nx = self.nq + self.nv
        self.ndx = self.nv * 2
        self.nu = self.nv - 2
        self.d = rotorDistance
        self.cm = coefM
        self.cf = coefF

    def calc(self, data, x, u):
        d, cf, cm = self.d, self.cf, self.cf
        S = np.array(np.zeros([self.nv,self.nu]))
        S[2:6,:4] = np.array([[1,1,1,1],[-d,d,-d,d],[-d,-d,d,d],[-cm/cf,-cm/cf,cm/cf,cm/cf]])
        np.fill_diagonal(S[6:, 4:], 1)
        data.a = np.dot(S,u)
        return data.a

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        return data.a

    def createData(self, pinocchioData):
        return ActuationDataUAM(self, pinocchioData)

class ActuationDataUAM:
    def __init__(self, model, pinocchioData):
        self.pinocchio = pinocchioData
        ndx, nv, nu = model.ndx, model.nv, model.nu
        d, cf, cm = model.d, model.cf, model.cf
        self.a = np.zeros(nv)  # result of calc
        self.A = np.zeros([nv, ndx + nu])  # result of calcDiff
        self.Ax = self.A[:, :ndx]
        self.Au = self.A[:, ndx:]
        self.Au[2:6,:4] = np.array([[1,1,1,1],[-d,d,d,-d],[-d,d,-d,d],[-cm/cf,-cm/cf,cm/cf,cm/cf]])
        np.fill_diagonal(self.Au[6:, 4:], 1)
        # Actuation considering motor forces instead of thrust and moments
        # This is the matrix that, given a force vector representing the four motors, outputs the thrust and moment
        # [      0,      0,     0,     0]
        # [      0,      0,     0,     0]
        # [      1,      1,     1,     1]
        # [     -d,      d,    -d,     d]
        # [     -d,     -d,     d,     d]
        # [ -cm/cf, -cm/cf, cm/cf, cm/cf]





class ActuationModelFreeFloating:
    '''
    This model transforms an actuation u into a joint torque tau.
    We implement here the simplest model: tau = S.T*u, where S is constant.
    '''
    def __init__(self, pinocchioModel):
        self.pinocchio = pinocchioModel
        if (pinocchioModel.joints[1].shortname() != 'JointModelFreeFlyer'):
            warnings.warn('Strange that the first joint is not a freeflyer')
        self.nq = pinocchioModel.nq
        self.nv = pinocchioModel.nv
        self.nx = self.nq + self.nv
        self.ndx = self.nv * 2
        self.nu = self.nv - 6

    def calc(self, data, x, u):
        data.a[6:] = u
        return data.a

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        return data.a

    def createData(self, pinocchioData):
        return ActuationDataFreeFloating(self, pinocchioData)


class ActuationDataFreeFloating:
    def __init__(self, model, pinocchioData):
        self.pinocchio = pinocchioData
        ndx, nv, nu = model.ndx, model.nv, model.nu
        self.a = np.zeros(nv)  # result of calc
        self.A = np.zeros([nv, ndx + nu])  # result of calcDiff
        self.Ax = self.A[:, :ndx]
        self.Au = self.A[:, ndx:]
        np.fill_diagonal(self.Au[6:, :], 1)


class ActuationModelFull:
    '''
    This model transforms an actuation u into a joint torque tau.
    We implement here the trivial model: tau = u
    '''
    def __init__(self, pinocchioModel):
        self.pinocchio = pinocchioModel
        self.nq = pinocchioModel.nq
        self.nv = pinocchioModel.nv
        self.nx = self.nq + self.nv
        self.ndx = self.nv * 2
        self.nu = self.nv

    def calc(self, data, x, u):
        data.a[:] = u
        return data.a

    def calcDiff(self, data, x, u, recalc=True):
        if recalc:
            self.calc(data, x, u)
        return data.a

    def createData(self, pinocchioData):
        return ActuationDataFull(self, pinocchioData)


class ActuationDataFull:
    def __init__(self, model, pinocchioData):
        self.pinocchio = pinocchioData
        ndx, nv, nu = model.ndx, model.nv, model.nu
        self.a = np.zeros(nv)  # result of calc
        self.A = np.zeros([nv, ndx + nu])  # result of calcDiff
        self.Ax = self.A[:, :ndx]
        self.Au = self.A[:, ndx:]
        np.fill_diagonal(self.Au[:, :], 1)
