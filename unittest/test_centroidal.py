import os
from locomote import ContactSequenceHumanoid
from crocoddyl import loadTalos
from crocoddyl import createPhiFromContactSequence
import numpy as np

TRAJ_DIR = os.getcwd()+"/../examples/traj_1step/"
MUSCOD_CS_OUTPUT_FILENAME=TRAJ_DIR+"contact_sequence_trajectory.xml"
CONTACT_SEQUENCE_XML_TAG = "contact_sequence"

robot = loadTalos()
rmodel = robot.model
rdata = robot.data

cs = ContactSequenceHumanoid(0)
cs.loadFromXML(MUSCOD_CS_OUTPUT_FILENAME, CONTACT_SEQUENCE_XML_TAG)

cc = createPhiFromContactSequence(rmodel, rdata, cs)
eps = 1e-4
for spl in cs.ms_interval_data[:-1]:
  for i in xrange(len(spl.time_trajectory)):
    x = spl.state_trajectory[i]
    dx = spl.dot_state_trajectory[i]
    u = spl.control_trajectory[i]
    t = spl.time_trajectory[i]

    assert(np.isclose(cc.com_vcom.eval(t)[0], x[:6,:], atol=eps).all())
    #assert(np.isclose(cc.vcom_acom.eval(t)[0], dx[:6,:]).all())
    assert(np.isclose(cc.hg.eval(t)[0][3:,:], x[6:,:], atol=eps).all())
    assert(np.isclose(cc.forces.eval(t)[0][:,:], u[:,:],atol=eps).all())
    
