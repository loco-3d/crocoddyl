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

def loadTalosArm(modelPath='/opt/openrobots/share',freeFloating=False):
    URDF_FILENAME = "talos_left_arm.urdf"
    URDF_SUBPATH = "/talos_data/robots/" + URDF_FILENAME
    robot = RobotWrapper.BuildFromURDF(modelPath+URDF_SUBPATH, [modelPath],
                                       pinocchio.JointModelFreeFlyer() if freeFloating else None)
    if freeFloating:
        u = robot.model.upperPositionLimit; u[:7] =  1;  robot.model.upperPositionLimit = u
        l = robot.model.lowerPositionLimit; l[:7] = -1;  robot.model.lowerPositionLimit = l
        
    return robot

def loadTalos(modelPath='/opt/openrobots/share'):
    URDF_FILENAME = "talos_reduced_v2.urdf"
    SRDF_FILENAME = "talos.srdf"
    SRDF_SUBPATH = "/talos_data/srdf/" + SRDF_FILENAME
    URDF_SUBPATH = "/talos_data/urdf/" + URDF_FILENAME
    robot = RobotWrapper.BuildFromURDF(modelPath+URDF_SUBPATH, [modelPath],
                                                               pinocchio.JointModelFreeFlyer())
    # Load SRDF file
    rmodel = robot.model
    pinocchio.getNeutralConfiguration(rmodel, modelPath+SRDF_SUBPATH, False)
    pinocchio.loadRotorParameters(rmodel, modelPath+SRDF_SUBPATH, False)
    rmodel.armature = \
              np.multiply(rmodel.rotorInertia.flat, np.square(rmodel.rotorGearRatio.flat))
    assert((rmodel.armature[:6]==0.).all())

    robot.q0.flat[:] = rmodel.neutralConfiguration
    
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
    pinocchio.getNeutralConfiguration(m2, modelPath+SRDF_SUBPATH, False)
    pinocchio.loadRotorParameters(m2, modelPath+SRDF_SUBPATH, False)
    m2.armature = \
            np.multiply(m2.rotorInertia.flat, np.square(m2.rotorGearRatio.flat))
    assert((m2.armature[:6]==0.).all())
    robot.q0 = m2.neutralConfiguration.copy()
    return robot