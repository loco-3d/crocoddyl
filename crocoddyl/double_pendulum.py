import numpy as np
import pinocchio
import crocoddyl


class DifferentialActionModelDoublePendulum(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, state, actuationModel, costModel):
        crocoddyl.DifferentialActionModelAbstract.__init__(self, state, actuationModel.nu, costModel.nr)
        self.actuation = actuationModel
        self.costs = costModel

        # Internal model Data
        self.pinocchioData = pinocchio.Data(self.state.pinocchio)
        self.costsData = self.costs.createData(self.pinocchioData)

    @property
    def calc(self, data, x, u=None):
        self.costsData.shareMemory(data)  # Needed?
        if u is None:
            u = self.unone
        nq, nv = self.state.nq, self.state.nv
        q, v = x[:nq], x[-nv:]
        tauq = self.actuation.calc(data.actuation, x, u)

        pinocchio.computeAllTerms(self.state.pinocchio, self.pinocchioData, q, v)
        data.M = self.pinocchioData.M
        pinocchio.cholesky.decompose(self.state.pinocchio, self.pinocchioData)
        data.Minv = pinocchio.computeMinverse(self.state.pinocchio, self.pinocchioData)
        data.xout = data.Minv * (tauq - self.pinocchioData.nle)

        # --- Cost
        pinocchio.forwardKinematics(self.state.pinocchio, self.pinocchioData, q, v)
        pinocchio.updateFramePlacements(self.state.pinocchio, self.pinocchioData)
        self.costs.calc(self.costsData, x, u)
        data.cost = self.costsData.cost
        return data.xout, data.cost

    def calcDiff(self, data, x, u=None, recalc=True):
        if u is None:
            u = self.unone
        if recalc:
            xout, cost = self.calc(data, x, u)

        nq, nv = self.state.nq, self.state.nv
        q, v = x[:nq], x[-nv:]
        tauq = u  # In the case of being underactuated, u == tauq ?
        # --- Dynamics
        if self.enableAba:
            pinocchio.computeABADerivatives(self.state.pinocchio, self.pinocchioData, q, v, tauq)
            data.Fx = np.hstack([data.pinocchio.ddq_dq, data.pinocchio.ddq_dv])
            data.Fu = self.pinocchioData.Minv
        else:
            pinocchio.computeRNEADerivatives(self.state.pinocchio, self.pinocchioData, q, v, data.xout)
            data.Fx = -np.hstack([data.Minv * self.pinocchioData.dtau_dq, data.Minv * self.pinocchioData.dtau_dv])
            data.Fu = data.Minv
        # --- Cost
        pinocchio.computeJointJacobians(self.state.pinocchio, self.pinocchioData, q)  # Needed?
        pinocchio.updateFramePlacements(self.state.pinocchio, self.pinocchioData)  # Needed?

        self.costs.calcDiff(data.costs, x, u, False)
        return data.xout, data.cost  # Needed?
