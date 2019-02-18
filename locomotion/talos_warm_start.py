import pinocchio
import conf_talos_warm_start as conf
from time import sleep
from locomote import ContactSequenceHumanoid
from centroidal_utils import createPhiFromContactSequence, createMultiphaseShootingProblem, createSwingTrajectories
from crocoddyl import m2a, a2m
from crocoddyl import ShootingProblem, SolverDDP
from crocoddyl import ActionModelImpact
from crocoddyl import CallbackDDPVerbose, CallbackSolverDisplay, CallbackDDPLogger, CallbackSolverTimer
import numpy as np
np.set_printoptions(linewidth=400, suppress=True)
robot = conf.robot
rmodel = robot.model
rdata = robot.data
rmodel.defaultState = np.concatenate([m2a(robot.q0),np.zeros(rmodel.nv)])

#----------------Load Contact Phases-----------------------
cs = ContactSequenceHumanoid(0)
cs.loadFromXML(conf.MUSCOD_CS_OUTPUT_FILENAME, conf.CONTACT_SEQUENCE_XML_TAG)

#----------------Define References-------------------------

swing_ref = createSwingTrajectories(rmodel, rdata, conf.X_init, conf.contact_patches, conf.DT)
phi_c = createPhiFromContactSequence(rmodel, rdata, cs, conf.contact_patches.keys())

#----------------Define Problem----------------------------
models = createMultiphaseShootingProblem(rmodel, rdata, conf.contact_patches,
                                         cs, phi_c, swing_ref, conf.DT)

disp = lambda xs: disptraj(robot,xs)

problem = ShootingProblem(m2a(conf.X_init[0]), models[:-1], models[-1])

#Set contacts in the data elements. Ugly.
#This is defined for IAMEuler. If using IAMRK4, differential is a list. so we need to change.
for d in problem.runningDatas:
  if hasattr(d, "differential"): #Because we also have the impact models without differntial.
    for (patchname, contactData) in d.differential.contact.contacts.iteritems():
      if "forces_"+patchname in d.differential.costs.costs:
        d.differential.costs["forces_"+patchname].contact = contactData
#---------Ugliness over------------------------


#-----------Create inital trajectory---------------
init = lambda t:0
init.X = []
init.U = []
i = 0
for m in problem.runningModels:
  #Impact models have zero timestep, thus they are copying the same state as the beginning of the next action model
  #State and control are defined at the beginning of the time step.
  if isinstance(m, ActionModelImpact):
    init.X.append(conf.X_init[i+1])
    init.U.append(np.zeros(m.nu))
    print "impact at ",i
  else:
    init.X.append(conf.X_init[i])
    init.U.append(conf.U_init[i])
    i+=1
assert(i==len(conf.U_init))
assert(i==len(conf.X_init)-1)
init.X.append(conf.X_init[-1])
#---------------Display Initial Trajectory--------------
if conf.DISPLAY:
  robot.initDisplay(loadModel=True)
if conf.DISPLAY and False:
  for x in init.X:  
    robot.display(a2m(x[:robot.nq]))
    #sleep(0.005)
#----------------------

ddp = SolverDDP(problem)
ddp.callback = [CallbackDDPVerbose(), CallbackSolverTimer()]
if conf.DISPLAY:
  ddp.callback.append(CallbackSolverDisplay(robot,4))
ddp.th_stop = 1e-9
ddp.solve(maxiter=1000,regInit=0.1,init_xs=init.X,init_us=init.U)
#---------------Display Final Trajectory--------------
if conf.DISPLAY:
  for x in init.X:  
    robot.display(a2m(x[:robot.nq]))
    #sleep(0.005)
#----------------------
