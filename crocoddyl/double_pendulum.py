import numpy as np
import pinocchio
import crocoddyl

class DifferentialActionModelDoublePendulum(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, state, actuationModel, costModel):
        crocoddyl.DifferentialActionModelAbstract.__init__(self, state, state.nv, costModel.nr)
        self.DifferentialActionDataType = DifferentialActionDataDoublePendulum
        self.pinocchio = pinocchioModel
        self.actuation = actuationModel
        self.costs = costModel

    @property
    
    def calc(self, data, x, u=None):
        if u is None:
            u = self.unone
        nq, nv = self.nq, self.nv
        q = a2m(x[:nq])
        v = a2m(x[-nv:])
        tauq[:] = self.actuation.calc(data.actuation, x, u)

        pinocchio.computeAllTerms(self.pinocchio, data.pinocchio, q, v)
        data.M = data.pinocchio.M
        data.Minv = np.linalg.inv(data.M)
        data.r = data.tauq - m2a(data.pinocchio.nle)
        data.xout[:] = np.dot(data.Minv, data.r)

        # --- Cost
        pinocchio.forwardKinematics(self.pinocchio, data.pinocchio, q, v)
        pinocchio.updateFramePlacements(self.pinocchio, data.pinocchio)
        data.cost = self.costs.calc(data.costs, x, u)
        return data.xout, data.cost

    def calcDiff(self, data, x, u=None, recalc=True):
        if u is None:
            u = self.unone
        if recalc:
            xout, cost = self.calc(data, x, u)
        nq, nv = self.nq, self.nv
        q = a2m(x[:nq])
        v = a2m(x[-nv:])
        tauq = a2m(u)
        a = a2m(data.xout)
        # --- Dynamics
        if self.forceAba:
            pinocchio.computeABADerivatives(self.pinocchio, data.pinocchio, q, v, tauq)
            data.Fx[:, :nv] = data.pinocchio.ddq_dq
            data.Fx[:, nv:] = data.pinocchio.ddq_dv
            data.Fu[:, :] = data.pinocchio.Minv
        else:
            pinocchio.computeRNEADerivatives(self.pinocchio, data.pinocchio, q, v, a)
            data.Fx[:, :nv] = -np.dot(data.Minv, data.pinocchio.dtau_dq)
            data.Fx[:, nv:] = -np.dot(data.Minv, data.pinocchio.dtau_dv)
            data.Fu[:, :] = data.Minv
        # --- Cost
        pinocchio.computeJointJacobians(self.pinocchio, data.pinocchio, q)
        pinocchio.updateFramePlacements(self.pinocchio, data.pinocchio)
        self.costs.calcDiff(data.costs, x, u, recalc=False)
        return data.xout, data.cost


class DifferentialActionDataDoublePendulum(DifferentialActionDataAbstract):
    def __init__(self, model):
        self.pinocchio = model.pinocchio.createData()
        costData = model.costs.createData(self.pinocchio)
        DifferentialActionDataAbstract.__init__(self, model, costData)
