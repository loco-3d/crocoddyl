import pinocchio
import rospkg
import gepetto.corbaserver
import pickle
import os
from time import sleep
rospack = rospkg.RosPack()
MODEL_PATH = rospack.get_path('talos_data')
MESH_DIR = MODEL_PATH+'/../'
URDF_FILENAME = "talos_reduced.urdf"
URDF_MODEL_PATH = MODEL_PATH + "/robots/" + URDF_FILENAME

robot = pinocchio.robot_wrapper.RobotWrapper.BuildFromURDF(URDF_MODEL_PATH, [MESH_DIR],
                                                           pinocchio.JointModelFreeFlyer())
cl = gepetto.corbaserver.Client()
gui = cl.gui
if gui.nodeExists("world"):
  gui.deleteNode("world",True)


robot.initDisplay(loadModel=True)

TRAJ_DIR = os.getcwd()+"/traj_1step/"

X_init = pickle.load( open(TRAJ_DIR+"X_init.out","rb"))
U_init = pickle.load( open(TRAJ_DIR+"U_init.out","rb"))
f_init = pickle.load( open(TRAJ_DIR+"f_init.out","rb"))
ddq_init = pickle.load( open(TRAJ_DIR+"ddq_init.out","rb"))

for x in X_init:
  robot.display(x[:robot.nq])
  sleep(0.005)
