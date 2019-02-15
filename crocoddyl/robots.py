import pinocchio
from pinocchio.utils import *
from pinocchio.robot_wrapper import RobotWrapper

def getTalosPathFromRos():
    '''
    Uses environment variable ROS_PACKAGE_PATH.
    Typically returns /opt/openrobots/share/talos_data
    '''
    import rospkg
    rospack = rospkg.RosPack()
    MODEL_PATH = rospack.get_path('talos_data')+'/../'
    return MODEL_PATH


def readParamsFromSrdf(robot, SRDF_PATH, verbose):
  rmodel = robot.model
  
  pinocchio.loadRotorParameters(rmodel, SRDF_PATH, verbose)
  rmodel.armature = \
                    np.multiply(rmodel.rotorInertia.flat, np.square(rmodel.rotorGearRatio.flat))
  try:
    pinocchio.loadReferenceConfigurations(rmodel, SRDF_PATH, verbose)
    robot.q0.flat[:] = rmodel.referenceConfigurations["half_sitting"].copy()
  except:
    print "loadReferenceConfigurations did not work. Please check your Pinocchio Version"
    try:
      pinocchio.getNeutralConfiguration(rmodel, SRDF_PATH, verbose)
      robot.q0.flat[:] = rmodel.neutralConfiguration.copy()
    except:
      robot.q0.flat[:] = pinocchio.neutral(rmodel)
  return
  
def loadTalosArm(modelPath='/opt/openrobots/share',freeFloating=False):
    URDF_FILENAME = "talos_left_arm.urdf"
    URDF_SUBPATH = "/talos_data/robots/" + URDF_FILENAME
    SRDF_FILENAME = "talos.srdf"
    SRDF_SUBPATH = "/talos_data/srdf/" + SRDF_FILENAME
    robot = RobotWrapper.BuildFromURDF(modelPath+URDF_SUBPATH, [modelPath],
                                       pinocchio.JointModelFreeFlyer() if freeFloating else None)
    rmodel = robot.model

    readParamsFromSrdf(robot, modelPath+SRDF_SUBPATH, False)
    if freeFloating: assert((rmodel.armature[:6]==0.).all())
    
    if freeFloating:
        u = robot.model.upperPositionLimit; u[:7] =  1;  robot.model.upperPositionLimit = u
        l = robot.model.lowerPositionLimit; l[:7] = -1;  robot.model.lowerPositionLimit = l
        
    return robot

def loadTalos(modelPath='/opt/openrobots/share'):
    URDF_FILENAME = "talos_reduced.urdf"
    SRDF_FILENAME = "talos.srdf"
    SRDF_SUBPATH = "/talos_data/srdf/" + SRDF_FILENAME
    URDF_SUBPATH = "/talos_data/urdf/" + URDF_FILENAME
    robot = RobotWrapper.BuildFromURDF(modelPath+URDF_SUBPATH, [modelPath],
                                                               pinocchio.JointModelFreeFlyer())
    # Load SRDF file
    readParamsFromSrdf(robot, modelPath+SRDF_SUBPATH, False)
    rmodel = robot.model
    assert((rmodel.armature[:6]==0.).all())
    
    """
    robot.q0.flat[:] =  [0,0,1.0192720229567027,0,0,0,1,0.0,0.0,-0.411354,0.859395,-0.448041,-0.001708,0.0,0.0,-0.411354,0.859395,-0.448041,-0.001708,0,0.006761,0.25847,0.173046,-0.0002,-0.525366,0,0,0.1,0.5,-0.25847,-0.173046,0.0002,-0.525366,0,0,0.1,0.5,0,0]
    """
    return robot


def loadTalosLegs(modelPath='/opt/openrobots/share'):
    from pinocchio import JointModelFreeFlyer,JointModelRX,JointModelRY,JointModelRZ
    robot = loadTalos(modelPath=modelPath)
    SRDF_FILENAME = "talos.srdf"
    SRDF_SUBPATH = "/talos_data/srdf/" + SRDF_FILENAME
    legMaxId = 14

    m1 = robot.model
    m2 = pinocchio.Model()
    for j,M,name,parent,Y in zip(m1.joints,m1.jointPlacements,m1.names,m1.parents,m1.inertias):
        if j.id<legMaxId:
            jid = m2.addJoint(parent,locals()[j.shortname()](),M,name)
            assert( jid == j.id )
            m2.appendBodyToJoint(jid,Y,pinocchio.SE3.Identity())
    m2.upperPositionLimit=np.matrix([1.]*19).T
    m2.lowerPositionLimit=np.matrix([-1.]*19).T
    #q2 = robot.q0[:19]
    for f in m1.frames:
        if f.parent<legMaxId: m2.addFrame(f)
            
    g2 = pinocchio.GeometryModel()
    for g in robot.visual_model.geometryObjects:
        if g.parentJoint<14:
            g2.addGeometryObject(g)

    robot.model=m2
    robot.data= m2.createData()
    robot.visual_model = g2
    #robot.q0=q2
    robot.visual_data = pinocchio.GeometryData(g2)


    # Load SRDF file
    robot.q0=np.matrix(np.resize(robot.q0, robot.model.nv)).T
    readParamsFromSrdf(robot, modelPath+SRDF_SUBPATH, False)

    assert((m2.armature[:6]==0.).all())
    return robot

def loadHyQ(modelPath='examples/hyq_description'):
    # Loading the URDF model from the internal submodule (located at
    # example/hyq_description)
    # TODO remove the submodule and load from hyq-data bynary. Note that this
    # task requires the creation of hyq-data binary.
    from pinocchio import JointModelFreeFlyer
    import os
    FILENAME = str(os.path.dirname(os.path.abspath(__file__))) + "/../"
    URDF_FILENAME = "hyq_no_sensors.urdf"
    URDF_SUBPATH = "/robots/" + URDF_FILENAME
    robot = RobotWrapper.BuildFromURDF(FILENAME+modelPath+URDF_SUBPATH, [modelPath],
                                       pinocchio.JointModelFreeFlyer())
    # TODO define default position inside srdf
    return robot
