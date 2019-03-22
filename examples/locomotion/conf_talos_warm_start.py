from crocoddyl import getTalosPathFromRos, loadTalos
import pinocchio
import pickle
import os
#---------------------------------------------------------
robot = loadTalos()
robot.model.armature[6:] =1.
#---------------------------------------------------------

DISPLAY = False
RUNTIME_DISPLAY = False
DT= 0.01

#-------------------------INITIAL TRAJECTORY--------------
TRAJ_DIR = os.getcwd()+"/traj_1step/"

X_init = pickle.load( open(TRAJ_DIR+"X_init.out","rb")) #loads the state x
ddq_init = pickle.load( open(TRAJ_DIR+"ddq_init.out","rb"))
#U_init = pickle.load( open(TRAJ_DIR+"U_init.out","rb")) # loads the control u
#f_init = pickle.load( open(TRAJ_DIR+"f_init.out","rb")) # loads the control u


#-----------------------Contact Sequence-----------------

MUSCOD_CS_OUTPUT_FILENAME=TRAJ_DIR+"contact_sequence_trajectory.xml"
CONTACT_SEQUENCE_XML_TAG = "contact_sequence"

# Define the contact frames which are active in the robot.
# allowed keys are: (LF_patch,RF_patch, LH_patch, RH_patch)
contact_patches = {"LF_patch": "leg_left_6_joint",
                   "RF_patch": "leg_right_6_joint"}
