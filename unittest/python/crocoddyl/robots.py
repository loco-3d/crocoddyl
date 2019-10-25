import numpy as np
import pinocchio
from pinocchio.robot_wrapper import RobotWrapper


def getTalosPathFromRos():
    '''
    Uses environment variable ROS_PACKAGE_PATH.
    Typically returns /opt/openrobots/share/talos_data
    '''
    import rospkg
    rospack = rospkg.RosPack()
    MODEL_PATH = rospack.get_path('talos_data') + '/../'
    return MODEL_PATH


def readParamsFromSrdf(robot, SRDF_PATH, verbose):
    rmodel = robot.model

    pinocchio.loadRotorParameters(rmodel, SRDF_PATH, verbose)
    rmodel.armature = np.multiply(rmodel.rotorInertia.flat, np.square(rmodel.rotorGearRatio.flat))
    pinocchio.loadReferenceConfigurations(rmodel, SRDF_PATH, verbose)
    robot.q0.flat[:] = rmodel.referenceConfigurations["half_sitting"].copy()
    return


def loadTalosArm(modelPath='/opt/openrobots/share/example-robot-data', freeFloating=False):
    URDF_FILENAME = "talos_left_arm.urdf"
    URDF_SUBPATH = "/talos_data/robots/" + URDF_FILENAME
    SRDF_FILENAME = "talos.srdf"
    SRDF_SUBPATH = "/talos_data/srdf/" + SRDF_FILENAME
    robot = RobotWrapper.BuildFromURDF(modelPath + URDF_SUBPATH, [modelPath],
                                       pinocchio.JointModelFreeFlyer() if freeFloating else None)
    rmodel = robot.model

    readParamsFromSrdf(robot, modelPath + SRDF_SUBPATH, False)
    if freeFloating:
        assert ((rmodel.armature[:6] == 0.).all())

    if freeFloating:
        u = robot.model.upperPositionLimit
        u[:7] = 1
        robot.model.upperPositionLimit = u
        limit = robot.model.lowerPositionLimit
        limit[:7] = -1
        robot.model.lowerPositionLimit = limit
    return robot


def loadTalos(modelPath='/opt/openrobots/share/example-robot-data'):
    URDF_FILENAME = "talos_reduced.urdf"
    SRDF_FILENAME = "talos.srdf"
    SRDF_SUBPATH = "/talos_data/srdf/" + SRDF_FILENAME
    URDF_SUBPATH = "/talos_data/urdf/" + URDF_FILENAME
    robot = RobotWrapper.BuildFromURDF(modelPath + URDF_SUBPATH, [modelPath], pinocchio.JointModelFreeFlyer())
    # Load SRDF file
    readParamsFromSrdf(robot, modelPath + SRDF_SUBPATH, False)
    rmodel = robot.model
    assert ((rmodel.armature[:6] == 0.).all())
    """
    robot.q0.flat[:] =  [0,0,1.0192720229567027,0,0,0,1,0.0,0.0,-0.411354,0.859395,-0.448041,-0.001708,
    0.0,0.0,-0.411354,0.859395,-0.448041,-0.001708,0,0.006761,0.25847,0.173046,-0.0002,-0.525366,0,0,
    0.1,0.5,-0.25847,-0.173046,0.0002,-0.525366,0,0,0.1,0.5,0,0]
    """
    return robot


def loadTalosLegs(modelPath='/opt/openrobots/share/example-robot-data'):
    robot = loadTalos(modelPath=modelPath)
    SRDF_FILENAME = "talos.srdf"
    SRDF_SUBPATH = "/talos_data/srdf/" + SRDF_FILENAME
    legMaxId = 14

    m1 = robot.model
    m2 = pinocchio.Model()
    for j, M, name, parent, Y in zip(m1.joints, m1.jointPlacements, m1.names, m1.parents, m1.inertias):
        if j.id < legMaxId:
            jid = m2.addJoint(parent, getattr(pinocchio, j.shortname())(), M, name)
            up = m2.upperPositionLimit
            down = m2.lowerPositionLimit
            up[m2.joints[jid].idx_q:m2.joints[jid].idx_q + j.nq] = m1.upperPositionLimit[j.idx_q:j.idx_q + j.nq]
            down[m2.joints[jid].idx_q:m2.joints[jid].idx_q + j.nq] = m1.lowerPositionLimit[j.idx_q:j.idx_q + j.nq]
            m2.upperPositionLimit = up
            m2.lowerPositionLimit = down
            assert (jid == j.id)
            m2.appendBodyToJoint(jid, Y, pinocchio.SE3.Identity())

    u = m2.upperPositionLimit
    u[:7] = 1
    m2.upperPositionLimit = u
    limit = m2.lowerPositionLimit
    limit[:7] = -1
    m2.lowerPositionLimit = limit

    # q2 = robot.q0[:19]
    for f in m1.frames:
        if f.parent < legMaxId:
            m2.addFrame(f)

    g2 = pinocchio.GeometryModel()
    for g in robot.visual_model.geometryObjects:
        if g.parentJoint < 14:
            g2.addGeometryObject(g)

    robot.model = m2
    robot.data = m2.createData()
    robot.visual_model = g2
    # robot.q0=q2
    robot.visual_data = pinocchio.GeometryData(g2)

    # Load SRDF file
    robot.q0 = np.matrix(np.resize(robot.q0, robot.model.nq)).T
    readParamsFromSrdf(robot, modelPath + SRDF_SUBPATH, False)

    assert ((m2.armature[:6] == 0.).all())
    return robot


def loadHyQ(modelPath='/opt/openrobots/share/example-robot-data'):
    URDF_FILENAME = "hyq_no_sensors.urdf"
    URDF_SUBPATH = "/hyq_description/robots/" + URDF_FILENAME
    robot = RobotWrapper.BuildFromURDF(modelPath + URDF_SUBPATH, [modelPath], pinocchio.JointModelFreeFlyer())
    # TODO define default position inside srdf
    robot.q0.flat[7:] = [-0.2, 0.75, -1.5, -0.2, -0.75, 1.5, -0.2, 0.75, -1.5, -0.2, -0.75, 1.5]
    robot.q0[2] = 0.57750958
    robot.model.referenceConfigurations["half_sitting"] = robot.q0
    return robot


def loadKinton(modelPath='/opt/openrobots/share/example-robot-data'):
    URDF_FILENAME = "kinton_arm.urdf"
    URDF_SUBPATH = "/kinton_description/urdf/" + URDF_FILENAME
    robot = RobotWrapper.BuildFromURDF(modelPath + URDF_SUBPATH, [modelPath], pinocchio.JointModelFreeFlyer())
    robot.q0.flat[7:] = [0, 0, 0, 0, 0, 0]
    robot.model.referenceConfigurations["initial_pose"] = robot.q0
    robot.q0.flat[7:] = [0, 0.2, 1.7, 2, 0, 0]
    robot.model.referenceConfigurations["centered"] = robot.q0
    return robot


def loadKintonArm(modelPath='/opt/openrobots/share/example-robot-data'):
    URDF_FILENAME = "kinton_arm.urdf"
    URDF_SUBPATH = "/kinton_description/urdf/" + URDF_FILENAME
    robot = RobotWrapper.BuildFromURDF(modelPath + URDF_SUBPATH, [modelPath])
    return robot


def load2dofPlanar(modelPath='/home/pepms/robotics/other-tools/robot-data'):
    URDF_FILENAME = "2dof_planar.urdf"
    URDF_SUBPATH = "/2dof_planar/urdf/" + URDF_FILENAME
    robot = RobotWrapper.BuildFromURDF(modelPath + URDF_SUBPATH, [modelPath], None)
    robot.q0.flat = [np.pi]
    return robot


def loadHector(modelPath='/home/jmarti/robotics/other-tools/robot-data'):
    URDF_FILENAME = "quadrotor_base.urdf"
    URDF_SUBPATH = "/hector-description/urdf/" + URDF_FILENAME
    robot = RobotWrapper.BuildFromURDF(modelPath + URDF_SUBPATH, [modelPath], pinocchio.JointModelFreeFlyer())
    return robot
