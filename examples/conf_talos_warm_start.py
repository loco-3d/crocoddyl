from crocoddyl import getTalosPathFromRos, loadTalos
import pinocchio
import pickle
import os
#---------------------------------------------------------
robot = loadTalos()
#---------------------------------------------------------

DISPLAY = False
DT= 0.005

#-------------------------INITIAL TRAJECTORY--------------
DT_INIT = 0.005
TRAJ_DIR = os.getcwd()+"/traj_1step/"

X_init = pickle.load( open(TRAJ_DIR+"X_init.out","rb")) #loads the state x
U_init = pickle.load( open(TRAJ_DIR+"U_init.out","rb")) # loads the control u

#loads the forces ("lambda") in the local frame in order LF, RF
#where lf: 'leg_left_6_joint', 'rf' : 'leg_right_6_joint'
f_init = pickle.load( open(TRAJ_DIR+"f_init.out","rb")) 
ddq_init = pickle.load( open(TRAJ_DIR+"ddq_init.out","rb")) #loads the acceleration

#-----------------------Contact Sequence-----------------

MUSCOD_CS_OUTPUT_FILENAME=TRAJ_DIR+"contact_sequence_trajectory.xml"
CONTACT_SEQUENCE_XML_TAG = "contact_sequence"
