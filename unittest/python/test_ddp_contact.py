import numpy as np
from numpy.linalg import norm

import pinocchio
from crocoddyl_legacy import (ActivationModelWeightedQuad, ActuationModelFreeFloating, ContactModel6D, ContactModelMultiple,
                       CostModelControl, CostModelFrameTranslation, CostModelState, CostModelSum,
                       DifferentialActionModelFloatingInContact, DifferentialActionModelFullyActuated,
                       IntegratedActionModelEuler, ShootingProblem, SolverDDP, StatePinocchio, a2m, loadTalosArm)

pinocchio.switchToNumpyMatrix()


def absmax(A):
    return np.max(abs(A))


def absmin(A):
    return np.min(abs(A))


opPointName = 'gripper_left_fingertip_2_link'
contactName = 'root_joint'

opPointName, contactName = contactName, opPointName


class FF:
    def __init__(self):
        robot = self.robot = loadTalosArm(freeFloating=True)
        rmodel = self.rmodel = robot.model

        qmin = rmodel.lowerPositionLimit
        qmin[:7] = -1
        rmodel.lowerPositionLimit = qmin
        qmax = rmodel.upperPositionLimit
        qmax[:7] = 1
        rmodel.upperPositionLimit = qmax
        State = self.State = StatePinocchio(rmodel)
        actModel = self.actModel = ActuationModelFreeFloating(rmodel)
        contactModel = self.contactModel = ContactModelMultiple(rmodel)
        contact6 = ContactModel6D(rmodel, rmodel.getFrameId(contactName), ref=pinocchio.SE3.Identity(), gains=[0., 0.])
        contactModel.addContact(name='contact', contact=contact6)
        costModel = self.costModel = CostModelSum(rmodel, nu=actModel.nu)
        self.cost1 = CostModelFrameTranslation(rmodel,
                                               nu=actModel.nu,
                                               frame=rmodel.getFrameId(opPointName),
                                               ref=np.array([.5, .4, .3]))
        stateWeights = np.array([0] * 6 + [0.01] * (rmodel.nv - 6) + [10] * rmodel.nv)
        self.cost2 = CostModelState(rmodel,
                                    State,
                                    ref=State.zero(),
                                    nu=actModel.nu,
                                    activation=ActivationModelWeightedQuad(stateWeights**2))
        self.cost3 = CostModelControl(rmodel, nu=actModel.nu)
        costModel.addCost(name="pos", weight=10, cost=self.cost1)
        costModel.addCost(name="regx", weight=0.1, cost=self.cost2)
        costModel.addCost(name="regu", weight=0.01, cost=self.cost3)

        self.dmodel = DifferentialActionModelFloatingInContact(rmodel, actModel, contactModel, costModel)
        self.model = IntegratedActionModelEuler(self.dmodel)
        self.data = self.model.createData()

        self.cd1 = self.data.differential.costs.costs['pos']
        self.cd2 = self.data.differential.costs.costs['regx']
        self.cd3 = self.data.differential.costs.costs['regu']

        self.ddata = self.data.differential
        self.rdata = self.data.differential.pinocchio

        self.x = self.State.rand()
        self.q = a2m(self.x[:rmodel.nq])
        self.v = a2m(self.x[rmodel.nq:])
        self.u = np.random.rand(self.model.nu)

    def calc(self, x=None, u=None):
        return self.model.calc(self.data, x if x is not None else self.x, u if u is not None else self.u)


class Fix:
    def __init__(self):
        robot = self.robot = loadTalosArm()
        rmodel = self.rmodel = robot.model
        rmodel.armature = np.matrix([0] * rmodel.nv).T
        for j in rmodel.joints[1:]:
            if j.shortname() != 'JointModelFreeFlyer':
                rmodel.armature[j.idx_v:j.idx_v + j.nv] = 1
        State = self.State = StatePinocchio(rmodel)
        self.cost1 = CostModelFrameTranslation(rmodel,
                                               nu=rmodel.nv,
                                               frame=rmodel.getFrameId(opPointName),
                                               ref=np.array([.5, .4, .3]))
        self.cost2 = CostModelState(rmodel, State, ref=State.zero(), nu=rmodel.nv)
        self.cost3 = CostModelControl(rmodel, nu=rmodel.nv)

        costModel = CostModelSum(rmodel)
        costModel.addCost(name="pos", weight=10, cost=self.cost1)
        costModel.addCost(name="regx", weight=0.1, cost=self.cost2)
        costModel.addCost(name="regu", weight=0.01, cost=self.cost3)

        self.dmodel = DifferentialActionModelFullyActuated(rmodel, costModel)
        self.model = IntegratedActionModelEuler(self.dmodel)

        self.data = self.model.createData()

        self.cd1 = self.data.differential.costs.costs['pos']
        self.cd2 = self.data.differential.costs.costs['regx']
        self.cd3 = self.data.differential.costs.costs['regu']

        self.ddata = self.data.differential
        self.rdata = self.data.differential.pinocchio

        self.x = self.State.rand()
        self.q = a2m(self.x[:rmodel.nq])
        self.v = a2m(self.x[rmodel.nq:])
        self.u = np.random.rand(self.model.nu)

    def calc(self, x=None, u=None):
        return self.model.calc(self.data, x if x is not None else self.x, u if u is not None else self.u)


ff = FF()
fix = Fix()

ff.q[:7].flat = ff.State.zero()[:7].flat
ff.v[:6] = 0
fix.q[:] = ff.q[7:]
fix.v[:] = ff.v[6:]
fix.u[:] = ff.u[:]
fix.x[:] = np.concatenate([fix.q, fix.v]).flat
ff.x[:] = np.concatenate([ff.q, ff.v]).flat

ff.model.timeStep = fix.model.timeStep = 5e-3

xfix, cfix = fix.model.calc(fix.data, fix.x, fix.u)
xff, cff = ff.model.calc(ff.data, ff.x, ff.u)

if ff.contactModel['contact'].frame == 1:
    assert (norm(cff - cfix) < 1e-6)
    assert (norm(xff[7:ff.rmodel.nq] - xfix[:fix.rmodel.nq]) < 1e-6)
    assert (norm(xff[ff.rmodel.nq + 6:] - xfix[fix.rmodel.nq:]) < 1e-6)

ff.model.timeStep = fix.model.timeStep = 1e-2
T = 20

xref = fix.State.rand()
xref[fix.rmodel.nq:] = 0
fix.calc(xref)

f = ff
f.u[:] = (0 * pinocchio.rnea(fix.rmodel, fix.rdata, fix.q, fix.v * 0, fix.v * 0)).flat
f.v[:] = 0
f.x[f.rmodel.nq:] = f.v.flat

# f.u[:] = np.zeros(f.model.nu)
f.model.differential.costs['pos'].weight = 1
f.model.differential.costs['regx'].weight = 0.01
f.model.differential.costs['regu'].weight = 0.0001

fterm = f.__class__()
fterm.model.differential.costs['pos'].weight = 1000
fterm.model.differential.costs['regx'].weight = 1
fterm.model.differential.costs['regu'].weight = 0.01

problem = ShootingProblem(f.x, [f.model] * T, fterm.model)
u0s = [f.u] * T
x0s = problem.rollout(u0s)

# disp = lambda xs: disptraj(f.robot, xs)
ddp = SolverDDP(problem)
# ddp.callback = [ CallbackDDPLogger(), CallbackDDPVerbose() ]
ddp.th_stop = 1e-18
ddp.solve(maxiter=1000)
