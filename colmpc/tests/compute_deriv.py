import coal
import numpy as np
import pinocchio as pin
from numpy import c_, cross, eye, r_
from pinocchio import skew

pin.SE3.__repr__ = pin.SE3.__str__


def compute_dist(rmodel, gmodel, q, vq, idg1, idg2):
    """Computing the distance between the two shapes."""

    rdata = rmodel.createData()
    gdata = gmodel.createData()

    # Updating the position of the joints & the geometry objects.
    pin.forwardKinematics(rmodel, rdata, q, vq)
    pin.updateGeometryPlacements(rmodel, rdata, gmodel, gdata)

    # Poses and geometries of the shapes
    shape1_placement = gdata.oMg[idg1]
    shape2_placement = gdata.oMg[idg2]
    shape1 = gmodel.geometryObjects[idg1]
    shape2 = gmodel.geometryObjects[idg2]

    req = coal.DistanceRequest()
    req.gjk_max_iterations = 20000
    req.abs_err = 0
    req.gjk_tolerance = 1e-9
    res = coal.DistanceResult()

    distance = coal.distance(
        shape1.geometry,
        shape1_placement,
        shape2.geometry,
        shape2_placement,
        req,
        res,
    )
    return distance


def compute_d_dist_dq(rmodel, gmodel, q: np.ndarray, vq, idg1, idg2):
    rdata = rmodel.createData()
    gdata = gmodel.createData()

    # Updating the position of the joints & the geometry objects.
    pin.forwardKinematics(rmodel, rdata, q, vq)
    pin.updateGeometryPlacements(rmodel, rdata, gmodel, gdata)

    # Poses and geometries of the shapes
    shape1_placement = gdata.oMg[idg1]
    shape2_placement = gdata.oMg[idg2]

    shape1 = gmodel.geometryObjects[idg1]
    shape2 = gmodel.geometryObjects[idg2]

    jacobian1 = pin.computeFrameJacobian(
        rmodel,
        rdata,
        q,
        shape1.parentFrame,
        pin.LOCAL_WORLD_ALIGNED,
    )

    jacobian2 = pin.computeFrameJacobian(
        rmodel,
        rdata,
        q,
        shape2.parentFrame,
        pin.LOCAL_WORLD_ALIGNED,
    )
    req = coal.DistanceRequest()
    req.gjk_max_iterations = 20000
    req.abs_err = 0
    req.gjk_tolerance = 1e-9
    res = coal.DistanceResult()
    # Computing the distance
    distance = coal.distance(
        shape1.geometry,
        shape1_placement,
        shape2.geometry,
        shape2_placement,
        req,
        res,
    )
    x1 = res.getNearestPoint1()
    x2 = res.getNearestPoint2()

    ## Transport the jacobian of frame 1 into the jacobian associated to x1
    # Vector from frame 1 center to p1
    f1p1 = x1 - rdata.oMf[shape1.parentFrame].translation
    # The following 2 lines are the easiest way to understand the transformation
    # although not the most efficient way to compute it.
    f1Mp1 = pin.SE3(np.eye(3), f1p1)
    jacobian1 = f1Mp1.actionInverse @ jacobian1

    ## Transport the jacobian of frame 2 into the jacobian associated to x2
    # Vector from frame 2 center to p2
    f2p2 = x2 - rdata.oMf[shape2.parentFrame].translation
    # The following 2 lines are the easiest way to understand the transformation
    # although not the most efficient way to compute it.
    f2Mp2 = pin.SE3(np.eye(3), f2p2)
    jacobian2 = f2Mp2.actionInverse @ jacobian2

    CP1_SE3 = pin.SE3.Identity()
    CP1_SE3.translation = x1

    CP2_SE3 = pin.SE3.Identity()
    CP2_SE3.translation = x2
    J = (x1 - x2).T / distance @ (jacobian1[:3] - jacobian2[:3])
    return J


def compute_Ldot(rmodel, gmodel, q, vq, idg1, idg2):
    rdata = rmodel.createData()
    gdata = gmodel.createData()
    # Compute robot placements and velocity at step 0 and 1
    pin.forwardKinematics(rmodel, rdata, q, vq)
    pin.updateFramePlacements(rmodel, rdata)
    pin.updateGeometryPlacements(rmodel, rdata, gmodel, gdata)

    elips1 = gmodel.geometryObjects[idg1].geometry
    elips2 = gmodel.geometryObjects[idg2].geometry

    idf1 = gmodel.geometryObjects[idg1].parentFrame
    idf2 = gmodel.geometryObjects[idg2].parentFrame

    M1 = gdata.oMg[idg1].copy()
    c1 = M1.translation
    M2 = gdata.oMg[idg2].copy()
    c2 = M2.translation

    # Get body velocity at step 0
    pin.forwardKinematics(rmodel, rdata, q, vq)
    nu1 = pin.getFrameVelocity(rmodel, rdata, idf1, pin.LOCAL_WORLD_ALIGNED).copy()
    v1, w1 = nu1.linear, nu1.angular
    nu2 = pin.getFrameVelocity(rmodel, rdata, idf2, pin.LOCAL_WORLD_ALIGNED).copy()
    v2, w2 = nu2.linear, nu2.angular

    req = coal.DistanceRequest()
    req.gjk_max_iterations = 20000
    req.abs_err = 0
    req.gjk_tolerance = 1e-9
    res = coal.DistanceResult()
    _ = coal.distance(
        elips1,
        gdata.oMg[idg1],
        elips2,
        gdata.oMg[idg2],
        req,
        res,
    )
    sol_x1 = res.getNearestPoint1()
    sol_x2 = res.getNearestPoint2()

    Ldot = (
        (sol_x1 - sol_x2) @ (v1 - v2)
        - np.cross(sol_x1 - sol_x2, sol_x1 - c1) @ w1
        + np.cross(sol_x1 - sol_x2, sol_x2 - c2) @ w2
    )

    return Ldot


def compute_ddot(rmodel, gmodel, q, vq, idg1, idg2):
    rdata = rmodel.createData()
    gdata = gmodel.createData()
    # Compute robot placements and velocity at step 0 and 1
    pin.forwardKinematics(rmodel, rdata, q, vq)
    pin.updateFramePlacements(rmodel, rdata)
    pin.updateGeometryPlacements(rmodel, rdata, gmodel, gdata)

    elips1 = gmodel.geometryObjects[idg1].geometry
    elips2 = gmodel.geometryObjects[idg2].geometry

    idf1 = gmodel.geometryObjects[idg1].parentFrame
    idf2 = gmodel.geometryObjects[idg2].parentFrame

    M1 = gdata.oMg[idg1].copy()
    c1 = M1.translation
    M2 = gdata.oMg[idg2].copy()
    c2 = M2.translation

    # Get body velocity at step 0
    pin.forwardKinematics(rmodel, rdata, q, vq)
    nu1 = pin.getFrameVelocity(rmodel, rdata, idf1, pin.LOCAL_WORLD_ALIGNED).copy()
    v1, w1 = nu1.linear, nu1.angular
    nu2 = pin.getFrameVelocity(rmodel, rdata, idf2, pin.LOCAL_WORLD_ALIGNED).copy()
    v2, w2 = nu2.linear, nu2.angular

    req = coal.DistanceRequest()
    req.gjk_max_iterations = 20000
    req.abs_err = 0
    req.gjk_tolerance = 1e-9
    res = coal.DistanceResult()
    distance = coal.distance(
        elips1,
        gdata.oMg[idg1],
        elips2,
        gdata.oMg[idg2],
        req,
        res,
    )
    sol_x1 = res.getNearestPoint1()
    sol_x2 = res.getNearestPoint2()

    Ldot = (
        (sol_x1 - sol_x2) @ (v1 - v2)
        - np.cross(sol_x1 - sol_x2, sol_x1 - c1) @ w1
        + np.cross(sol_x1 - sol_x2, sol_x2 - c2) @ w2
    )

    return Ldot / distance


def compute_d_d_dot_dq_dq_dot(rmodel, gmodel, q, vq, idg1, idg2):
    rdata = rmodel.createData()
    gdata = gmodel.createData()
    # Compute robot placements and velocity at step 0 and 1
    pin.forwardKinematics(rmodel, rdata, q, vq)
    pin.updateFramePlacements(rmodel, rdata)
    pin.updateGeometryPlacements(rmodel, rdata, gmodel, gdata)

    elips1 = gmodel.geometryObjects[idg1].geometry
    elips2 = gmodel.geometryObjects[idg2].geometry

    idf1 = gmodel.geometryObjects[idg1].parentFrame
    idf2 = gmodel.geometryObjects[idg2].parentFrame

    M1 = gdata.oMg[idg1].copy()
    R1, c1 = M1.rotation, M1.translation
    M2 = gdata.oMg[idg2].copy()
    R2, c2 = M2.rotation, M2.translation

    radii1 = elips1.radii
    radii2 = elips2.radii

    D1 = np.diagflat([1 / r**2 for r in radii1])
    D2 = np.diagflat([1 / r**2 for r in radii2])

    A1 = R1 @ D1 @ R1.T
    M2 = gdata.oMg[idg2].copy()
    R2, c2 = M2.rotation, M2.translation
    A2 = R2 @ D2 @ R2.T

    # Get body velocity at step 0
    nu1 = pin.getFrameVelocity(rmodel, rdata, idf1, pin.LOCAL_WORLD_ALIGNED).copy()
    v1, w1 = nu1.linear, nu1.angular
    nu2 = pin.getFrameVelocity(rmodel, rdata, idf2, pin.LOCAL_WORLD_ALIGNED).copy()
    v2, w2 = nu2.linear, nu2.angular

    req = coal.DistanceRequest()
    req.gjk_max_iterations = 20000
    req.abs_err = 0
    req.gjk_tolerance = 1e-9
    res = coal.DistanceResult()
    distance = coal.distance(
        elips1,
        gdata.oMg[idg1],
        elips2,
        gdata.oMg[idg2],
        req,
        res,
    )
    sol_x1 = res.getNearestPoint1()
    sol_x2 = res.getNearestPoint2()

    Ldot = (
        (sol_x1 - sol_x2) @ (v1 - v2)
        - np.cross(sol_x1 - sol_x2, sol_x1 - c1) @ w1
        + np.cross(sol_x1 - sol_x2, sol_x2 - c2) @ w2
    )

    dist_dot = Ldot / distance
    # theta = (c1,c2,r1,r2)
    theta_dot = r_[v1, w1, v2, w2]

    # dist_dot derivative wrt theta
    dL_dtheta = r_[
        sol_x1 - sol_x2,
        -cross(sol_x1 - sol_x2, sol_x1 - c1),
        -(sol_x1 - sol_x2),
        cross(sol_x1 - sol_x2, sol_x2 - c2),
    ]

    assert np.allclose(Ldot, dL_dtheta @ theta_dot)

    sol_lam1, sol_lam2 = (
        -(sol_x1 - c1).T @ (sol_x1 - sol_x2),
        (sol_x2 - c2).T @ (sol_x1 - sol_x2),
    )

    Lyy = r_[
        c_[eye(3) + sol_lam1 * A1, -eye(3), A1 @ (sol_x1 - c1), np.zeros(3)],
        c_[-eye(3), eye(3) + sol_lam2 * A2, np.zeros(3), A2 @ (sol_x2 - c2)],
        [r_[A1 @ (sol_x1 - c1), np.zeros(3), np.zeros(2)]],
        [r_[np.zeros(3), A2 @ (sol_x2 - c2), np.zeros(2)]],
    ]
    Lyc = r_[
        c_[-sol_lam1 * A1, np.zeros([3, 3])],
        c_[np.zeros([3, 3]), -sol_lam2 * A2],
        [r_[-A1 @ (sol_x1 - c1), np.zeros(3)]],
        [r_[np.zeros(3), -A2 @ (sol_x2 - c2)]],
    ]
    Lyr = r_[
        c_[
            sol_lam1 * (A1 @ skew(sol_x1 - c1) - skew(A1 @ (sol_x1 - c1))),
            np.zeros([3, 3]),
        ],
        c_[
            np.zeros([3, 3]),
            sol_lam2 * (A2 @ skew(sol_x2 - c2) - skew(A2 @ (sol_x2 - c2))),
        ],
        [r_[(sol_x1 - c1) @ A1 @ skew(sol_x1 - c1), np.zeros(3)]],
        [r_[np.zeros(3), (sol_x2 - c2) @ A2 @ skew(sol_x2 - c2)]],
    ]
    Lyth = c_[Lyc[:, :3], Lyr[:, :3], Lyc[:, 3:], Lyr[:, 3:]]

    yth = -np.linalg.inv(Lyy) @ Lyth

    dx1 = yth[:3]
    dx2 = yth[3:6]

    ddL_dtheta2 = (
        r_[
            dx1 - dx2,
            -skew(sol_x1 - sol_x2) @ dx1 + skew(sol_x1 - c1) @ (dx1 - dx2),
            -dx1 + dx2,
            skew(sol_x1 - sol_x2) @ dx2 - skew(sol_x2 - c2) @ (dx1 - dx2),
        ]
        + r_[
            np.zeros([3, 12]),
            c_[skew(sol_x1 - sol_x2), np.zeros([3, 9])],
            np.zeros([3, 12]),
            c_[np.zeros([3, 6]), -skew(sol_x1 - sol_x2), np.zeros([3, 3])],
        ]
    )
    # Verif using finite diff

    d_dist_dot_dtheta = (
        theta_dot.T @ ddL_dtheta2 / distance - dist_dot / distance**2 * dL_dtheta
    )

    d_dist_dot_dtheta_dot = dL_dtheta / distance

    # ##################################3
    # Robot derivatives

    pin.computeJointJacobians(rmodel, rdata, q)
    J1 = pin.getFrameJacobian(rmodel, rdata, idf1, pin.LOCAL_WORLD_ALIGNED)
    J2 = pin.getFrameJacobian(rmodel, rdata, idf2, pin.LOCAL_WORLD_ALIGNED)

    assert np.allclose(nu1.vector, J1 @ vq)
    assert np.allclose(nu2.vector, J2 @ vq)

    dtheta_dq = r_[J1, J2]
    assert np.allclose(dtheta_dq @ vq, theta_dot)

    dtheta_dot_dqdot = r_[J1, J2]

    pin.computeForwardKinematicsDerivatives(rmodel, rdata, q, vq, np.zeros(rmodel.nv))

    in1_dnu1_dq, in1_dnu1_dqdot = pin.getFrameVelocityDerivatives(
        rmodel, rdata, idf1, pin.LOCAL
    )
    in2_dnu2_dq, in2_dnu2_dqdot = pin.getFrameVelocityDerivatives(
        rmodel, rdata, idf2, pin.LOCAL
    )

    inLWA1_dv1_dq = R1 @ in1_dnu1_dq[:3] - skew(v1) @ R1 @ in1_dnu1_dqdot[3:]
    inLWA1_dw1_dq = R1 @ in1_dnu1_dq[3:]
    inLWA2_dv2_dq = R2 @ in2_dnu2_dq[:3] - skew(v2) @ R2 @ in2_dnu2_dqdot[3:]
    inLWA2_dw2_dq = R2 @ in2_dnu2_dq[3:]

    dtheta_dot_dq = r_[inLWA1_dv1_dq, inLWA1_dw1_dq, inLWA2_dv2_dq, inLWA2_dw2_dq]
    # TODO: here a 0* is needed. WHHHHYYYYYYYYYYYYYYYYYYYYY!

    d_dist_dot_dq = (
        d_dist_dot_dtheta @ dtheta_dq + d_dist_dot_dtheta_dot @ dtheta_dot_dq
    )
    d_dist_dot_dqdot = d_dist_dot_dtheta_dot @ dtheta_dot_dqdot
    # print(f"ddistdotdqdot {d_dist_dot_dqdot}")
    return d_dist_dot_dq, d_dist_dot_dqdot
