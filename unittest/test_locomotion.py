import numpy as np
from crocoddyl import loadTalos
from crocoddyl.locomotion import ContactSequenceWrapper
from locomote import ContactSequenceHumanoid

MUSCOD_CS_OUTPUT_FILENAME = TRAJ_DIR = "data/test_locomotion.xml"
CONTACT_SEQUENCE_XML_TAG = "contact_sequence"

robot = loadTalos()
rmodel = robot.model
rdata = robot.data

cs = ContactSequenceHumanoid(0)
cs.loadFromXML(MUSCOD_CS_OUTPUT_FILENAME, CONTACT_SEQUENCE_XML_TAG)
csw = ContactSequenceWrapper(cs, {"LF_patch": "leg_left_6_joint", "RF_patch": "leg_right_6_joint"})
csw.createCentroidalPhi(rmodel, rdata)
cc = csw.phi_c
eps = 1e-4

assert (len(cs.ms_interval_data[:-1]) > 0)

for spl in cs.ms_interval_data[:-1]:
    for i in range(len(spl.time_trajectory)):
        x = spl.state_trajectory[i]
        dx = spl.dot_state_trajectory[i]
        u = spl.control_trajectory[i]
        t = spl.time_trajectory[i]
        assert (np.isclose(cc.com_vcom.eval(t)[0], x[:6, :], atol=eps).all())
        # assert(np.isclose(cc.vcom_acom.eval(t)[0], dx[:6,:]).all())
        assert (np.isclose(cc.hg.eval(t)[0][3:, :], x[6:, :], atol=eps).all())
        assert (np.isclose(cc.forces["RF_patch"].eval(t)[0][:, :], u[:6, :], atol=eps).all())  # RF
        assert (np.isclose(cc.forces["LF_patch"].eval(t)[0][:, :], u[6:12, :], atol=eps).all())  # LF
