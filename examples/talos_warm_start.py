import pinocchio
import conf_talos_1step as conf
from time import sleep
from locomote import ContactSequenceHumanoid
from crocoddyl import createPhiFromContactSequence
robot = conf.robot
rmodel = robot.model
rdata = robot.data

#---------------Display Initial Trajectory--------------
if conf.DISPLAY:
  robot.initDisplay(loadModel=True)
  for x in conf.X_init:  
    robot.display(x[:robot.nq])
    sleep(0.005)

#----------------Load Contact Phases-----------------------
cs = ContactSequenceHumanoid(0)
cs.loadFromXML(conf.MUSCOD_CS_OUTPUT_FILENAME, conf.CONTACT_SEQUENCE_XML_TAG)

cc = createPhiFromContactSequence(rmodel, rdata, cs)
