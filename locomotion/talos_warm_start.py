import pinocchio
import conf_talos_warm_start as conf
from time import sleep
from locomote import ContactSequenceHumanoid
from centroidal_utils import createPhiFromContactSequence, createMultiphaseShootingProblem, createSwingTrajectories
from crocoddyl import EmptyClass, m2a, a2m
from crocoddyl import ShootingProblem, SolverDDP
from crocoddyl import CallbackDDPVerbose, CallbackSolverDisplay, CallbackDDPLogger
import numpy as np
np.set_printoptions(linewidth=400, suppress=True)
robot = conf.robot
rmodel = robot.model
rdata = robot.data
rmodel.defaultState = np.concatenate([m2a(rmodel.neutralConfiguration),np.zeros(rmodel.nv)])

#---------------Display Initial Trajectory--------------
if conf.DISPLAY and False:
  robot.initDisplay(loadModel=True)
  for x in conf.X_init:  
    robot.display(a2m(x[:robot.nq]))
    #sleep(0.005)

#----------------Load Contact Phases-----------------------
cs = ContactSequenceHumanoid(0)
cs.loadFromXML(conf.MUSCOD_CS_OUTPUT_FILENAME, conf.CONTACT_SEQUENCE_XML_TAG)

#----------------Define References-------------------------
init = EmptyClass()
init.x = conf.X_init
init.u = conf.U_init

swing_ref = createSwingTrajectories(rmodel, rdata, init.x, conf.contact_patches, conf.DT)
phi_c = createPhiFromContactSequence(rmodel, rdata, cs, conf.contact_patches.keys())

#----------------Define Problem----------------------------
models = createMultiphaseShootingProblem(rmodel, rdata, conf.contact_patches,
                                         cs, phi_c, swing_ref, conf.DT)

disp = lambda xs: disptraj(robot,xs)

problem = ShootingProblem(m2a(init.x[0]), models[:-1], models[-1])

#Set contacts in the data elements. Ugly.
#This is defined for IAMEuler. If using IAMRK4, differential is a list. so we need to change.
for d in problem.runningDatas:
  for (patchname, contactData) in d.differential.contact.contacts.iteritems():
    d.differential.costs["forces_"+patchname].contact = contactData
#---------Ugliness over------------------------

ddp = SolverDDP(problem)
ddp.callback = [CallbackDDPVerbose()]#, CallbackSolverDisplay(robot,4)]
ddp.th_stop = 1e-9
ddp.solve(maxiter=1000,regInit=0.1,init_xs=init.x)
