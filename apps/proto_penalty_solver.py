'''
Validation of the penalty solver for tackling a terminal constraint
with some accuracies.
'''

import matplotlib.pylab as plt

import pinocchio
from continuous import DifferentialActionModelPositioning, IntegratedActionModelEuler
from pinocchio.utils import *
from refact import ShootingProblem, SolverDDP

m2a = lambda m: np.array(m.flat)
a2m = lambda a: np.matrix(a).T
absmax = lambda A: np.max(abs(A))
absmin = lambda A: np.min(abs(A))

path = '/home/nmansard/src/cddp/examples/'

urdf = path + 'talos_data/robots/talos_left_arm.urdf'
robot = pinocchio.robot_wrapper.RobotWrapper.BuildFromURDF(urdf, [path])
# ,pinocchio.JointModelFreeFlyer()

# urdf = path + 'hyq_description/robots/hyq_no_sensors.urdf'
# robot = pinocchio.robot_wrapper.RobotWrapper.BuildFromURDF(urdf, [path], pinocchio.JointModelFreeFlyer())
qmin = robot.model.lowerPositionLimit
qmin[:7] = -1
robot.model.lowerPositionLimit = qmin
qmax = robot.model.upperPositionLimit
qmax[:7] = 1
robot.model.upperPositionLimit = qmax

rmodel = robot.model
rdata = rmodel.createData()

dmodel = DifferentialActionModelPositioning(rmodel)
model = IntegratedActionModelEuler(dmodel)

# --- DDP INTEGRATIVE TEST
T = 20
model.timeStep = 1e-2
termmodel = IntegratedActionModelEuler(DifferentialActionModelPositioning(rmodel))
termmodel.differential.costs['pos'].weight = 1
termmodel.differential.costs['regx'].weight = 0.1

x0 = np.concatenate([m2a(rmodel.neutralConfiguration), np.zeros(rmodel.nv)])
model.differential.xref = x0.copy()
problem = ShootingProblem(x0, [model] * T, termmodel)
ddp = SolverDDP(problem)
ugrav = m2a(
    pinocchio.rnea(rmodel, rdata, a2m(problem.initialState[:rmodel.nq]), a2m(np.zeros(rmodel.nv)),
                   a2m(np.zeros(rmodel.nv))))

# xddp,uddp,dddp = ddp.solve(verbose=True)

termmodel.differential.costs['pos'].weight = 10
ddp.setCandidate()
# xddp,uddp,dddp = ddp.solve(verbose=True)
ddp.computeDirection()
try:
    ddp.tryStep(1)
except ArithmeticError as err:
    assert (err.args[0] == 'forward error')
'''
termmodel.differential.costs['pos'].weight = 100
xs,us,done = ddp.solve(verbose=True)
assert(done)
'''


def disp(xs, dt=0.1):
    import time
    for x in xs:
        robot.display(a2m(x[:robot.nq]))
        time.sleep(dt)


# robot.initDisplay()

ddp.callback = [CallbackDDPLogger()]

termmodel.differential.costs['pos'].weight = 1
ddp.solve(verbose=True)
endEff = problem.terminalData.differential.pinocchio.oMf[model.differential.costs['pos'].cost.frame]
for i in range(1, 10):
    termmodel.differential.costs['pos'].weight = 10**i
    ddp.solve(maxiter=5, init_xs=ddp.xs, init_us=ddp.us, verbose=True, isFeasible=True, regInit=1e-3)
    print '\n', endEff.translation.T, '\n'

ddp.solve(maxiter=500, init_xs=ddp.xs, init_us=ddp.us, verbose=True, isFeasible=True, regInit=1e-3)
print '\n', endEff.translation.T, '\n'

plt.plot([c[-1] for c in ddp.callback.costs])
plt.show()
