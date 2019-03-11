import conf_talos_warm_start as conf
import locomote
import numpy as np
import pinocchio
from centroidal_utils import createMultiphaseShootingProblem, createPhiFromContactSequence, createSwingTrajectories
from crocoddyl import (ActionModelImpact, CallbackDDPVerbose, CallbackSolverDisplay, ShootingProblem, SolverDDP,
                       StatePinocchio, a2m, m2a)
from locomote import ContactSequenceHumanoid

np.set_printoptions(linewidth=400, suppress=True)
robot = conf.robot
rmodel = robot.model
rdata = robot.data
rmodel.defaultState = np.concatenate([m2a(robot.q0), np.zeros(rmodel.nv)])

# ----------------Load Contact Phases-----------------------
cs = ContactSequenceHumanoid(0)
cs.loadFromXML(conf.MUSCOD_CS_OUTPUT_FILENAME, conf.CONTACT_SEQUENCE_XML_TAG)

# ----------------Define References-------------------------

swing_ref = createSwingTrajectories(rmodel, rdata, conf.X_init, conf.contact_patches, 0.005)
phi_c = createPhiFromContactSequence(rmodel, rdata, cs, conf.contact_patches.keys())

# ----------------Define Problem----------------------------
models = createMultiphaseShootingProblem(rmodel, rdata, conf.contact_patches, cs, phi_c, swing_ref, conf.DT)

# disp = lambda xs: disptraj(robot, xs)

problem = ShootingProblem(m2a(conf.X_init[0]), models[:-1], models[-1])

# Set contacts in the data elements. Ugly.
# This is defined for IAMEuler. If using IAMRK4, differential is a list. so we need to change.
for d in problem.runningDatas:
    if hasattr(d, "differential"):  # Because we also have the impact models without differntial.
        for (patchname, contactData) in d.differential.contact.contacts.iteritems():
            if "forces_" + patchname in d.differential.costs.costs:
                d.differential.costs["forces_" + patchname].contact = contactData
# ---------Ugliness over------------------------

# -----------Create inital trajectory---------------


class Init:
    def __init__(self, *args, **kwargs):
        self = 0
        self.X = []
        self.U = []
        x_tsid = np.matrix(conf.X_init).T
        conf.ddq_init.append(np.zeros(rmodel.nv))
        dx_tsid = np.vstack([x_tsid[rmodel.nv:, :], np.matrix(conf.ddq_init).T])
        t_tsid = np.linspace(0., cs.contact_phases[-1].time_trajectory[-1], len(conf.ddq_init) + 1)
        x_spl = locomote.CubicHermiteSpline(a2m(t_tsid), x_tsid, dx_tsid)

        state = StatePinocchio(rmodel)
        dt = conf.DT

        def x_eval(t):
            return m2a(x_spl.eval(t)[0])

        for i, (m, d) in enumerate(zip(problem.runningModels, problem.runningDatas)):
            # Impact models have zero timestep,
            # thus they are copying the same state as the beginning of the next action model.
            # State and control are defined at the beginning of the time step.
            if isinstance(m, ActionModelImpact):
                self.X.append(x_eval((i + 1) * dt))
                self.U.append(np.zeros(m.nu))
                # print("impact at ",i)
            else:
                xp = x_eval(i * dt)
                xn = x_eval((i + 1) * dt)
                self.X.append(xp)
                dx = state.diff(xp, xn)
                acc = dx[rmodel.nv:] / dt
                u = pinocchio.rnea(rmodel, rdata, a2m(xp[:rmodel.nq]), a2m(xp[rmodel.nq:]), a2m(acc))
                m.differential.calc(d.differential, self.X[-1])
                contactJ = d.differential.contact.J
                f = np.dot(np.linalg.pinv(contactJ.T[:6, :]), u[:6])
                u -= (np.dot(contactJ.transpose(), f))
                self.U.append(np.array(u[6:]).squeeze().copy())
        self.X.append(conf.X_init[-1])


init = Init()
# ---------------Display Initial Trajectory--------------
if conf.DISPLAY:
    robot.initDisplay(loadModel=True)
if conf.DISPLAY:
    for x in init.X:
        robot.display(a2m(x[:robot.nq]))
        # sleep(0.005)
# ----------------------

ddp = SolverDDP(problem)
ddp.callback = [CallbackDDPVerbose()]  # CallbackSolverTimer()]
if conf.RUNTIME_DISPLAY:
    ddp.callback.append(CallbackSolverDisplay(robot, 4))
ddp.th_stop = 1e-9
ddp.solve(maxiter=100, regInit=0.1, init_xs=init.X, init_us=init.U)
# ---------------Display Final Trajectory--------------
if conf.DISPLAY:
    for x in init.X:
        robot.display(a2m(x[:robot.nq]))
        # sleep(0.005)
# ----------------------
