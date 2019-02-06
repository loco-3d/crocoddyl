import pinocchio
import conf_talos_warm_start as conf
from time import sleep
from locomote import ContactSequenceHumanoid
from centroidal_utils import createPhiFromContactSequence, createMultiphaseShootingProblem, createSwingTrajectories
from crocoddyl import EmptyClass, m2a
import numpy as np
robot = conf.robot
rmodel = robot.model
rdata = robot.data
rmodel.defaultState = np.concatenate([m2a(rmodel.neutralConfiguration),np.zeros(rmodel.nv)])

#---------------Display Initial Trajectory--------------
if conf.DISPLAY:
  robot.initDisplay(loadModel=True)
  for x in conf.X_init:  
    robot.display(x[:robot.nq])
    sleep(0.005)

#----------------Load Contact Phases-----------------------
cs = ContactSequenceHumanoid(0)
cs.loadFromXML(conf.MUSCOD_CS_OUTPUT_FILENAME, conf.CONTACT_SEQUENCE_XML_TAG)

#----------------Define References-------------------------
init = EmptyClass()
init.x0 = conf.X_init
init.u0 = conf.U_init

swing_ref = createSwingTrajectories(rmodel, rdata, init.x0, conf.contact_patches, conf.DT)
phi_c = createPhiFromContactSequence(rmodel, rdata, cs, conf.contact_patches.keys())

#----------------Define Problem----------------------------
problem = createMultiphaseShootingProblem(rmodel, rdata, conf.contact_patches,
                                          cs, phi_c, swing_ref, conf.DT)
