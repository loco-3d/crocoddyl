###############################################################################
# BSD 3-Clause License
#
# Copyright (C) 2026, Heriot-Watt University
# Copyright note valid unless otherwise stated in individual files.
# All rights reserved.
###############################################################################


import copy
import warnings

import numpy as np
import pinocchio
import scipy.linalg as scl

import crocoddyl


def a2m(a):
    return np.matrix(a).T


def m2a(m):
    return np.array(m).squeeze()


def rev_enumerate(lname):
    return reversed(list(enumerate(lname)))


def absmax(A):
    return np.max(abs(A))


def raiseIfNan(A, error=None):
    if error is None:
        error = scl.LinAlgError("NaN in array")
    if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(abs(np.asarray(A)) > 1e30):
        raise error


class StateVectorDerived(crocoddyl.StateAbstract):
    def __init__(self, nx):
        crocoddyl.StateAbstract.__init__(self, nx, nx)

    def zero(self):
        return np.matrix(np.zeros(self.nx)).T

    def rand(self):
        rng = np.random.default_rng()
        return np.matrix(rng.random(self.nx)).T

    def diff(self, x0, x1):
        return x1 - x0

    def integrate(self, x, dx):
        return x + dx

    def Jdiff(self, x1, x2, firstsecond=crocoddyl.Jcomponent.both):
        if firstsecond == crocoddyl.Jcomponent.both:
            return [
                self.Jdiff(x1, x2, crocoddyl.Jcomponent.first),
                self.Jdiff(x1, x2, crocoddyl.Jcomponent.second),
            ]

        J = np.zeros([self.ndx, self.ndx])
        if firstsecond == crocoddyl.Jcomponent.first:
            J[:, :] = -np.eye(self.ndx)
        elif firstsecond == crocoddyl.Jcomponent.second:
            J[:, :] = np.eye(self.ndx)
        return J

    def Jintegrate(self, x, dx, firstsecond=crocoddyl.Jcomponent.both):
        if firstsecond == crocoddyl.Jcomponent.both:
            return [
                self.Jintegrate(x, dx, crocoddyl.Jcomponent.first),
                self.Jintegrate(x, dx, crocoddyl.Jcomponent.second),
            ]
        return np.eye(self.ndx)


class StateMultibodyDerived(crocoddyl.StateAbstract):
    def __init__(self, pinocchioModel):
        crocoddyl.StateAbstract.__init__(
            self, pinocchioModel.nq + pinocchioModel.nv, 2 * pinocchioModel.nv
        )
        self.model = pinocchioModel

    def zero(self):
        q = pinocchio.neutral(self.model)
        v = pinocchio.utils.zero(self.nv)
        return np.concatenate([q, v])

    def rand(self):
        q = pinocchio.randomConfiguration(self.model)
        v = pinocchio.utils.rand(self.nv)
        return np.concatenate([q, v])

    def diff(self, x0, x1):
        q0 = x0[: self.nq]
        q1 = x1[: self.nq]
        v0 = x0[-self.nv :]
        v1 = x1[-self.nv :]
        dq = pinocchio.difference(self.model, q0, q1)
        return np.concatenate([dq, v1 - v0])

    def integrate(self, x, dx):
        q = x[: self.nq]
        v = x[-self.nv :]
        dq = dx[: self.nv]
        dv = dx[-self.nv :]
        qn = pinocchio.integrate(self.model, q, dq)
        return np.concatenate([qn, v + dv])

    def Jdiff(self, x1, x2, firstsecond=crocoddyl.Jcomponent.both):
        if firstsecond == crocoddyl.Jcomponent.both:
            return [
                self.Jdiff(x1, x2, crocoddyl.Jcomponent.first),
                self.Jdiff(x1, x2, crocoddyl.Jcomponent.second),
            ]

        if firstsecond == crocoddyl.Jcomponent.first:
            dx = self.diff(x2, x1)
            q = x2[: self.model.nq]
            dq = dx[: self.model.nv]
            Jdq = pinocchio.dIntegrate(self.model, q, dq)[1]
            return np.matrix(-scl.block_diag(np.linalg.inv(Jdq), np.eye(self.nv)))
        elif firstsecond == crocoddyl.Jcomponent.second:
            dx = self.diff(x1, x2)
            q = x1[: self.nq]
            dq = dx[: self.nv]
            Jdq = pinocchio.dIntegrate(self.model, q, dq)[1]
            return np.matrix(scl.block_diag(np.linalg.inv(Jdq), np.eye(self.nv)))

    def Jintegrate(self, x, dx, firstsecond=crocoddyl.Jcomponent.both):
        if firstsecond == crocoddyl.Jcomponent.both:
            return [
                self.Jintegrate(x, dx, crocoddyl.Jcomponent.first),
                self.Jintegrate(x, dx, crocoddyl.Jcomponent.second),
            ]

        q = x[: self.nq]
        dq = dx[: self.nv]
        Jq, Jdq = pinocchio.dIntegrate(self.model, q, dq)
        if firstsecond == crocoddyl.Jcomponent.first:
            return np.matrix(scl.block_diag(np.linalg.inv(Jq), np.eye(self.nv)))
        elif firstsecond == crocoddyl.Jcomponent.second:
            return np.matrix(scl.block_diag(np.linalg.inv(Jdq), np.eye(self.nv)))


class SquashingSmoothSatDerived(crocoddyl.SquashingModelAbstract):
    def __init__(self, u_lb, u_ub, ns):
        self.u_lb = u_lb
        self.u_ub = u_ub
        self.smooth = 0.1
        crocoddyl.SquashingModelAbstract.__init__(self, ns)

    def calc(self, data, s):
        a = np.power(self.smooth * (self.u_ub - self.u_lb), 2)
        data.u = 0.5 * (
            self.u_lb + np.power(a + np.power(s - self.u_lb, 2), 0.5)
        ) + 0.5 * (self.u_ub - np.power(a + np.power(s - self.u_ub, 2), 0.5))

    def calcDiff(self, data, s):
        a = np.power(self.smooth * (self.u_ub - self.u_lb), 2)
        du_ds = 0.5 * (
            np.multiply(
                np.power(a + np.power((s - self.u_lb), 2), -0.5), (s - self.u_lb)
            )
            - np.multiply(
                np.power(a + np.power((s - self.u_ub), 2), -0.5), (s - self.u_ub)
            )
        )
        np.fill_diagonal(data.du_ds, du_ds)


class JointDynamicsDerived(crocoddyl.JointDynamicsModelAbstract):
    def __init__(self, id, nq, nv, nu=None, p=None):
        if nu is None:
            nu = nv
        super().__init__(id, nq, nv, nu)
        self._p = np.array([] if p is None else p, dtype=float)

    def calc(self, data, q, v, u):
        data.friction = np.zeros(self.nv)
        data.tau = np.pad(u, (0, self.nv - self.nu))

    def calcDiff(self, data, q, v, u):
        data.dtau_dq = np.zeros((self.nv, self.nv))
        data.dtau_dv = np.zeros((self.nv, self.nv))
        data.dtau_du = np.eye(self.nv, self.nu)
        data.Mtau = np.eye(self.nu, self.nv)

    def commands(self, data, q, v, tau):
        data.friction = np.zeros(self.nv)
        data.u = tau[: self.nu]

    def get_np(self):
        return self._p.size

    def set_parameters(self, p):
        self._p = np.array(p, dtype=float)

    def get_parameters(self):
        return self._p.copy()

    def get_parametrization(self):
        return self._p.copy()

    def updateParametrizationDerivative(self, dgamma_dp):
        return np.eye(self.np)

    def computeJointTorqueRegressor(self, joint_dtau_dp, q, v, u):
        result = np.zeros((self.nv, self.np))
        n = min(self.nv, self.np)
        result[:n, :n] = np.eye(n)
        return result

    def createData(self):
        return crocoddyl.JointDynamicsDataAbstract(self)


class UnicycleModelDerived(crocoddyl.ActionModelAbstract):
    def __init__(self):
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(3), 2, 5)
        self.dt = 0.1
        self.costWeights = [10.0, 1.0]

    def calc(self, data, x, u=None):
        if u is None:
            data.xnext[:] = x
            data.r[:3] = self.costWeights[0] * x
            data.cost = 0.5 * sum(data.r**2)
        else:
            v, w = u
            px, py, theta = x
            c, s, dt = np.cos(theta), np.sin(theta), self.dt
            data.xnext[0] = px + c * v * dt
            data.xnext[1] = py + s * v * dt
            data.xnext[2] = theta + w * dt
            data.r[:3] = self.costWeights[0] * x
            data.r[3:] = self.costWeights[1] * u
            data.cost = 0.5 * sum(data.r**2)

    def calcDiff(self, data, x, u=None):
        if u is None:
            data.Lx[:] = x * ([self.costWeights[0] ** 2] * self.state.nx)
        else:
            v = u[0]
            theta = x[2]
            data.Lx[:] = x * ([self.costWeights[0] ** 2] * self.state.nx)
            c, s, dt = np.cos(theta), np.sin(theta), self.dt
            data.Fx[0, 2] = -s * v * dt
            data.Fx[1, 2] = c * v * dt
            data.Fu[0, 0] = c * dt
            data.Fu[1, 0] = s * dt
            data.Fu[2, 1] = dt
            data.Lu[:] = u * ([self.costWeights[1] ** 2] * self.nu)

    def createData(self):
        data = UnicycleDataDerived(self)
        return data


class UnicycleDataDerived(crocoddyl.ActionDataAbstract):
    def __init__(self, model):
        crocoddyl.ActionDataAbstract.__init__(self, model)
        nx, nu = model.state.nx, model.nu
        self.Lxx[range(nx), range(nx)] = [model.costWeights[0] ** 2] * nx
        self.Luu[range(nu), range(nu)] = [model.costWeights[1] ** 2] * nu
        self.Fx[0, 0] = 1
        self.Fx[1, 1] = 1
        self.Fx[2, 2] = 1


class LQRModelDerived(crocoddyl.ActionModelAbstract):
    def __init__(self, nx, nu, driftFree=True):
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(nx), nu)
        self.A = np.eye(self.state.nx)
        self.B = np.eye(self.state.nx)[:, : self.nu]
        self.Q = np.eye(self.state.nx)
        self.R = np.eye(self.nu)
        self.N = np.zeros((self.state.nx, self.nu))
        self.f = [np.zeros(self.state.nx) if driftFree else np.ones(self.state.nx)]
        self.q = np.ones(self.state.nx)
        self.r = np.ones(self.nu)

    @classmethod
    def fromLQR(cls, A, B, Q, R, N, f, q, r):
        model = cls(A.shape[1], B.shape[1], False)
        model.A = A
        model.B = B
        model.Q = Q
        model.R = R
        model.N = N
        model.f = f
        model.q = q
        model.r = r
        return model

    def calc(self, data, x, u=None):
        if u is None:
            data.xnext[:] = x
            data.cost = 0.5 * np.dot(x.T, np.dot(self.Q, x))
            data.cost += np.dot(self.q.T, x)
        else:
            data.xnext[:] = np.dot(self.A, x) + np.dot(self.B, u) + self.f
            data.cost = 0.5 * np.dot(x.T, np.dot(self.Q, x))
            data.cost += 0.5 * np.dot(u.T, np.dot(self.R, u))
            data.cost += np.dot(x.T, np.dot(self.N, u))
            data.cost += np.dot(self.q.T, x) + np.dot(self.r.T, u)

    def calcDiff(self, data, x, u=None):
        if u is None:
            data.Lx[:] = self.q + np.dot(self.Q, x)
        else:
            data.Lx[:] = self.q + np.dot(self.Q, x) + np.dot(self.N, u)
            data.Lu[:] = self.r + np.dot(self.R, u) + np.dot(self.N.T, x)

    def createData(self):
        data = LQRDataDerived(self)
        return data


class LQRDataDerived(crocoddyl.ActionDataAbstract):
    def __init__(self, model):
        crocoddyl.ActionDataAbstract.__init__(self, model)
        self.Fx[:, :] = model.A
        self.Fu[:, :] = model.B
        self.Lxx[:, :] = model.Q
        self.Luu[:, :] = model.R
        self.Lxu[:, :] = model.N


class StateCostModelDerived(crocoddyl.CostModelAbstract):
    def __init__(self, state, activation=None, xref=None, nu=None):
        activation = (
            activation
            if activation is not None
            else crocoddyl.ActivationModelQuad(state.ndx)
        )
        self.xref = xref if xref is not None else state.zero()
        if nu is None:
            crocoddyl.CostModelAbstract.__init__(self, state, activation)
        else:
            crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)

    def calc(self, data, x, u=None):
        data.residual.r[:] = self.state.diff(self.xref, x)
        self.activation.calc(data.activation, data.residual.r)
        data.cost = data.activation.a_value

    def calcDiff(self, data, x, u=None):
        # The old code was looking like this.
        # But, the std::vector<Eigen::MatrixXd> returned by Jdiff is destroyed
        # before the assignment.
        # To avoid this issue, we store the std::vector in a variable.
        # data.residual.Rx[:] = self.state.Jdiff(
        #     self.xref, x, crocoddyl.Jcomponent.second
        # )[0]

        diff = self.state.Jdiff(self.xref, x, crocoddyl.Jcomponent.second)
        data.residual.Rx[:] = diff[0]
        self.activation.calcDiff(data.activation, data.residual.r)
        data.Lx[:] = np.dot(data.residual.Rx.T, data.activation.Ar)
        data.Lxx[:, :] = np.dot(
            data.residual.Rx.T, np.dot(data.activation.Arr, data.residual.Rx)
        )


class ControlCostModelDerived(crocoddyl.CostModelAbstract):
    def __init__(self, state, activation=None, uref=None, nu=None):
        nu = nu if nu is not None else state.nv
        activation = (
            activation if activation is not None else crocoddyl.ActivationModelQuad(nu)
        )
        self.uref = uref if uref is not None else pinocchio.utils.zero(nu)
        crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)

    def calc(self, data, x, u=None):
        if u is None:
            data.cost = 0.0
        else:
            data.residual.r[:] = u - self.uref
            self.activation.calc(data.activation, data.residual.r)
            data.cost = data.activation.a_value

    def calcDiff(self, data, x, u=None):
        if u is not None:
            self.activation.calcDiff(data.activation, data.residual.r)
            data.Lu[:] = data.activation.Ar
            data.Luu[:, :] = data.activation.Arr


class CoMPositionCostModelDerived(crocoddyl.CostModelAbstract):
    def __init__(self, state, activation=None, cref=None, nu=None):
        activation = (
            activation if activation is not None else crocoddyl.ActivationModelQuad(3)
        )
        if nu is None:
            crocoddyl.CostModelAbstract.__init__(self, state, activation)
        else:
            crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)
        self.cref = cref

    def calc(self, data, x, u=None):
        data.residual.r[:] = data.shared.pinocchio.com[0] - self.cref
        self.activation.calc(data.activation, data.residual.r)
        data.cost = data.activation.a_value

    def calcDiff(self, data, x, u=None):
        self.activation.calcDiff(data.activation, data.residual.r)
        data.residual.Rx[:] = np.hstack(
            [
                data.shared.pinocchio.Jcom,
                pinocchio.utils.zero((self.activation.nr, self.state.nv)),
            ]
        )
        data.Lx[:] = np.hstack(
            [
                np.dot(data.shared.pinocchio.Jcom.T, data.activation.Ar),
                np.zeros(self.state.nv),
            ]
        )
        data.Lxx[:, :] = np.vstack(
            [
                np.hstack(
                    [
                        np.dot(
                            data.shared.pinocchio.Jcom.T,
                            np.dot(data.activation.Arr, data.shared.pinocchio.Jcom),
                        ),
                        np.zeros((self.state.nv, self.state.nv)),
                    ]
                ),
                np.zeros((self.state.nv, self.state.ndx)),
            ]
        )


class FramePlacementCostModelDerived(crocoddyl.CostModelAbstract):
    def __init__(self, state, activation=None, frame_id=None, placement=None, nu=None):
        activation = (
            activation if activation is not None else crocoddyl.ActivationModelQuad(6)
        )
        if nu is None:
            crocoddyl.CostModelAbstract.__init__(self, state, activation)
        else:
            crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)
        self._frame_id = frame_id
        self._placement = placement

    def calc(self, data, x, u=None):
        data.rMf = self._placement.inverse() * data.shared.pinocchio.oMf[self._frame_id]
        data.residual.r[:] = pinocchio.log(data.rMf).vector
        self.activation.calc(data.activation, data.residual.r)
        data.cost = data.activation.a_value

    def calcDiff(self, data, x, u=None):
        pinocchio.updateFramePlacements(self.state.pinocchio, data.shared.pinocchio)
        data.rJf[:, :] = pinocchio.Jlog6(data.rMf)
        data.fJf[:, :] = pinocchio.getFrameJacobian(
            self.state.pinocchio,
            data.shared.pinocchio,
            self._frame_id,
            pinocchio.ReferenceFrame.LOCAL,
        )
        data.J[:, :] = np.dot(data.rJf, data.fJf)
        self.activation.calcDiff(data.activation, data.residual.r)
        data.residual.Rx[:] = np.hstack(
            [data.J, np.zeros((self.activation.nr, self.state.nv))]
        )
        data.Lx[:] = np.hstack(
            [np.dot(data.J.T, data.activation.Ar), np.zeros(self.state.nv)]
        )
        data.Lxx[:, :] = np.vstack(
            [
                np.hstack(
                    [
                        np.dot(data.J.T, np.dot(data.activation.Arr, data.J)),
                        np.zeros((self.state.nv, self.state.nv)),
                    ]
                ),
                np.zeros((self.state.nv, self.state.ndx)),
            ]
        )

    def createData(self, collector):
        data = FramePlacementCostDataDerived(self, collector)
        return data


class FramePlacementCostDataDerived(crocoddyl.CostDataAbstract):
    def __init__(self, model, collector):
        crocoddyl.CostDataAbstract.__init__(self, model, collector)
        self.rMf = pinocchio.SE3.Identity()
        self.rJf = pinocchio.Jlog6(self.rMf)
        self.fJf = np.zeros((6, model.state.nv))
        self.rJf = np.zeros((6, 6))
        self.J = np.zeros((6, model.state.nv))


class FrameTranslationCostModelDerived(crocoddyl.CostModelAbstract):
    def __init__(
        self, state, activation=None, frame_id=None, translation=None, nu=None
    ):
        activation = (
            activation if activation is not None else crocoddyl.ActivationModelQuad(3)
        )
        if nu is None:
            crocoddyl.CostModelAbstract.__init__(self, state, activation)
        else:
            crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)
        self._frame_id = frame_id
        self._translation = translation

    def calc(self, data, x, u=None):
        data.residual.r[:] = (
            data.shared.pinocchio.oMf[self._frame_id].translation - self._translation
        )
        self.activation.calc(data.activation, data.residual.r)
        data.cost = data.activation.a_value

    def calcDiff(self, data, x, u=None):
        pinocchio.updateFramePlacements(self.state.pinocchio, data.shared.pinocchio)
        data.R[:, :] = data.shared.pinocchio.oMf[self._frame_id].rotation
        data.J[:, :] = np.dot(
            data.R,
            pinocchio.getFrameJacobian(
                self.state.pinocchio,
                data.shared.pinocchio,
                self._frame_id,
                pinocchio.ReferenceFrame.LOCAL,
            )[:3, :],
        )
        self.activation.calcDiff(data.activation, data.residual.r)
        data.residual.Rx[:] = np.hstack(
            [data.J, np.zeros((self.activation.nr, self.state.nv))]
        )
        data.Lx[:] = np.hstack(
            [np.dot(data.J.T, data.activation.Ar), np.zeros(self.state.nv)]
        )
        data.Lxx[:, :] = np.vstack(
            [
                np.hstack(
                    [
                        np.dot(data.J.T, np.dot(data.activation.Arr, data.J)),
                        np.zeros((self.state.nv, self.state.nv)),
                    ]
                ),
                np.zeros((self.state.nv, self.state.ndx)),
            ]
        )

    def createData(self, collector):
        data = FrameTranslationDataDerived(self, collector)
        return data


class FrameTranslationDataDerived(crocoddyl.CostDataAbstract):
    def __init__(self, model, collector):
        crocoddyl.CostDataAbstract.__init__(self, model, collector)
        self.R = np.eye(3)
        self.J = np.zeros((3, model.state.nv))


class FrameRotationCostModelDerived(crocoddyl.CostModelAbstract):
    def __init__(self, state, activation=None, frame_id=None, rotation=None, nu=None):
        activation = (
            activation if activation is not None else crocoddyl.ActivationModelQuad(3)
        )
        if nu is None:
            crocoddyl.CostModelAbstract.__init__(self, state, activation)
        else:
            crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)
        self._frame_id = frame_id
        self._rotation = rotation

    def calc(self, data, x, u=None):
        data.rRf[:, :] = np.dot(
            self._rotation.T, data.shared.pinocchio.oMf[self._frame_id].rotation
        )
        data.residual.r[:] = pinocchio.log3(data.rRf)
        self.activation.calc(data.activation, data.residual.r)
        data.cost = data.activation.a_value

    def calcDiff(self, data, x, u=None):
        pinocchio.updateFramePlacements(self.state.pinocchio, data.shared.pinocchio)
        data.rJf[:, :] = pinocchio.Jlog3(data.rRf)
        data.fJf[:, :] = pinocchio.getFrameJacobian(
            self.state.pinocchio,
            data.shared.pinocchio,
            self._frame_id,
            pinocchio.ReferenceFrame.LOCAL,
        )[3:, :]
        data.J[:, :] = np.dot(data.rJf, data.fJf)
        self.activation.calcDiff(data.activation, data.residual.r)
        data.residual.Rx[:] = np.hstack(
            [data.J, np.zeros((self.activation.nr, self.state.nv))]
        )
        data.Lx[:] = np.hstack(
            [np.dot(data.J.T, data.activation.Ar), np.zeros(self.state.nv)]
        )
        data.Lxx[:, :] = np.vstack(
            [
                np.hstack(
                    [
                        np.dot(data.J.T, np.dot(data.activation.Arr, data.J)),
                        np.zeros((self.state.nv, self.state.nv)),
                    ]
                ),
                np.zeros((self.state.nv, self.state.ndx)),
            ]
        )

    def createData(self, collector):
        data = FrameRotationCostDataDerived(self, collector)
        return data


class FrameRotationCostDataDerived(crocoddyl.CostDataAbstract):
    def __init__(self, model, collector):
        crocoddyl.CostDataAbstract.__init__(self, model, collector)
        self.rRf = np.eye(3)
        self.rJf = np.zeros((3, 3))
        self.fJf = np.zeros((3, model.state.nv))
        self.J = np.zeros((3, model.state.nv))


class FrameVelocityCostModelDerived(crocoddyl.CostModelAbstract):
    def __init__(self, state, activation=None, frame_id=None, velocity=None, nu=None):
        activation = (
            activation if activation is not None else crocoddyl.ActivationModelQuad(6)
        )
        if nu is None:
            crocoddyl.CostModelAbstract.__init__(self, state, activation)
        else:
            crocoddyl.CostModelAbstract.__init__(self, state, activation, nu)
        self._frame_id = frame_id
        self._velocity = velocity

    def calc(self, data, x, u=None):
        data.residual.r[:] = (
            pinocchio.getFrameVelocity(
                self.state.pinocchio,
                data.shared.pinocchio,
                self._frame_id,
                pinocchio.LOCAL,
            )
            - self._velocity
        ).vector
        self.activation.calc(data.activation, data.residual.r)
        data.cost = data.activation.a_value

    def calcDiff(self, data, x, u=None):
        v_partial_dq, v_partial_dv = pinocchio.getJointVelocityDerivatives(
            self.state.pinocchio,
            data.shared.pinocchio,
            data.joint,
            pinocchio.ReferenceFrame.LOCAL,
        )
        self.activation.calcDiff(data.activation, data.residual.r)
        data.residual.Rx[:] = np.hstack(
            [np.dot(data.fXj, v_partial_dq), np.dot(data.fXj, v_partial_dv)]
        )
        data.Lx[:] = np.dot(data.residual.Rx.T, data.activation.Ar)
        data.Lxx[:, :] = np.dot(
            data.residual.Rx.T, np.dot(data.activation.Arr, data.residual.Rx)
        )

    def createData(self, collector):
        data = FrameVelocityCostDataDerived(self, collector)
        return data


class FrameVelocityCostDataDerived(crocoddyl.CostDataAbstract):
    def __init__(self, model, collector):
        crocoddyl.CostDataAbstract.__init__(self, model, collector)
        self.fXj = (
            model.state.pinocchio.frames[model._frame_id].placement.inverse().action
        )
        if tuple(int(i) for i in pinocchio.__version__.split(".")) >= (3, 0, 0):
            self.joint = model.state.pinocchio.frames[model._frame_id].parentJoint
        else:
            self.joint = model.state.pinocchio.frames[model._frame_id].parent


class SolverFDDP(crocoddyl.SolverAbstract):
    def __init__(
        self,
        problem,
        dyn_solver=crocoddyl.DynamicsSolverType.FeasShoot,
        term_solver=crocoddyl.EqualitySolverType.LuNull,
    ):
        crocoddyl.SolverAbstract.__init__(self, problem)
        # Allocate data
        self.allocateData()
        # Search and convergence parameters
        Ts = int(self.problem.T / max(3, self.problem.nthreads))
        self.setDynamicsSolver(dyn_solver, Ts)
        self.term_solver = term_solver
        self.th_grad = 1e-12
        self.th_noImprovement = np.finfo(float).eps ** 0.8
        # Regularization parameters
        self.reg_incFactor = 10
        self.reg_decFactor = 5
        self.th_stepDec = 0.25
        self.th_stepInc = 0.25
        self.th_minImprove = 1e-2  # [0, 100.]
        # Constraint parameters
        self.th_acceptNegStep = 8.0
        self.th_acceptMinStep = 0.01  # [alpha_min, 0.02]
        self.rho = 0.3
        self.th_minffeas = np.sqrt(np.finfo(float).eps / (1 - self.rho))
        self.upsilon = 0.0
        self.upsilon_decFactor = 0.5
        self.zero_upsilon = False

    def computeDirection(self, recalc=True):
        # Update the batch's derivatives
        if recalc:
            self.calcDir()
        # Update the search direction associated with the batch's internal constraints
        self.backwardPass()
        # Update search direction associated with the batch's constraint-to-go conditions
        if self.problem.terminalModel.nh_T != 0:
            self.linearRollout()
            self.batchPass()
            self.updateDir()
        elif self.dyn_solver != crocoddyl.DynamicsSolverType.SingleShoot:
            self.linearRollout()

    def computeCandidate(self, stepLength=1.0):
        # Update primal, dual and slack variables
        self.forwardPass(stepLength)
        self.updateDualsAndSlacks(stepLength)

    def forwardPass(self, stepLength=1):
        if self.dyn_solver == crocoddyl.DynamicsSolverType.FeasShoot:
            self.feasShootForwardPass(stepLength)
        elif self.dyn_solver == crocoddyl.DynamicsSolverType.MultiShoot:
            self.multiShootForwardPass(stepLength)
        elif self.dyn_solver == crocoddyl.DynamicsSolverType.HybridShoot:
            self.hybridShootForwardPass(stepLength)
        elif self.dyn_solver == crocoddyl.DynamicsSolverType.SingleShoot:
            self.singleShootForwardPass(stepLength)
        else:
            self.feasShootForwardPass(stepLength)

    # This is a virtual finction
    def updateDualsAndSlacks(self, stepLength=1):
        # Update the dual variables and slacks
        pass

    def stoppingCriteria(self):
        self.feas = self.ffeas + self.gfeas + self.hfeas
        self.stop = max(self.feas, abs(self.dVexp_full) / (1.0 + abs(self.cost)))
        return copy.deepcopy(self.stop)

    def expectedImprovement(self):
        # We define dVexp = Vexp - Vexptry as done for dV
        # The expected cost changes with the dynamics gaps.
        self.DV[:] = np.zeros(3)
        if self.dyn_solver == crocoddyl.DynamicsSolverType.SingleShoot:
            self.DV[0] -= self.fs[-1].T @ self.Vx[-1]
            self.DV[0] -= 0.5 * self.fs[-1].T @ self.Vxx_f[-1]
            for t in range(self.problem.T):  # in parallel
                nu = self.problem.runningModels[t].nu
                if nu != 0:
                    self.DV[1] += self.k[t].T @ self.Qu[t]
                    self.DV[2] -= self.k[t].T @ self.Quuk[t]
                self.DV[0] -= self.fs[t].T @ self.Vx[t]
                self.DV[0] -= 0.5 * self.fs[t].T @ self.Vxx_f[t]
        else:
            for t in range(self.problem.T):  # in parallel
                m = self.problem.runningModels[t]
                d = self.problem.runningDatas[t]
                ndx, nu = m.state.ndx, m.nu
                self.Lxx_dx[t][:] = d.Lxx @ self.dxs[t]
                self.Luu_du[t][:] = d.Luu @ self.dus[t]
                self.Lxu_du[t][:] = d.Lxu.reshape((ndx, nu)) @ self.dus[t]
                self.DV[1] -= self.dxs[t].T @ d.Lx
                self.DV[1] -= self.dus[t].T @ d.Lu
                self.DV[2] -= self.dxs[t].T @ self.Lxx_dx[t]
                self.DV[2] -= self.dus[t].T @ self.Luu_du[t]
                self.DV[2] -= 2 * self.dxs[t].T @ self.Lxu_du[t]
            d = self.problem.terminalData
            self.Lxx_dx[-1][:] = d.Lxx @ self.dxs[-1]
            self.DV[1] -= self.dxs[-1].T @ d.Lx
            self.DV[2] -= self.dxs[-1].T @ self.Lxx_dx[-1]
        return self.DV

    def computeMeritFunctionImprovement(self):
        # In single shooting, we do not consider the dynamics feasibility in the merit
        # function. This is because the dynamics are always satisfied.
        if self.dyn_solver == crocoddyl.DynamicsSolverType.SingleShoot:
            self.ffeas = 0.0
            self.ffeas_try = 0.0
            self.dfeas -= self.ffeas - self.ffeas_try
        self.dPhi = self.dV + self.upsilon * self.dfeas

    def computeExpectedMeritFunctionImprovement(self):
        self.dPhiexp = self.dVexp + self.stepLength * self.upsilon * self.dfeas

    # This is a virtual function
    def checkAcceptance(self):
        # Check if we should accept or not the step. The criterio is as follows.
        # When expected to decrease the merit function value (dPhiexp > 0), we analyse
        # if we are actually decreasing or not (dPhi > 0 or dPhi < 0) and define different
        # criterio. For the first case (dPhi > 0), we use the Armijo condition with the
        # merit function. Instead, for the second case, we use the Armijo condition with the
        # cost function as this encourage progress and the possibility of increasing the cost
        # when expectations are unrealistic. Moreover, when it is expected to increase the
        # merit function, our strategy is to accept an increment in the merit function if
        # the feasibility passes our stopping criteria or in the cost function otherwise. This
        # approach enables our solver to increase both infeasibility and cost in order to
        # ensure convergence; it increases the algorithm's globalization. Finally, we accept
        # any improvement for step lengths smaller than th_acceptMinStep. This ensures
        # any possible progress in the iteration.
        acceptStep = False
        if (
            abs(self.dPhi) <= self.th_noImprovement
            and abs(self.dPhiexp) <= self.th_noImprovement
        ):
            acceptStep = True
        elif self.dPhiexp >= 0.0:
            if self.dPhi > 0.0:
                if (
                    self.dPhi > self.th_acceptStep * self.dPhiexp
                    or abs(self.DV[1]) < self.th_grad
                ):
                    acceptStep = True
            elif (
                self.dV > self.th_acceptStep * self.dVexp
                or abs(self.DV[1]) < self.th_grad
            ):
                acceptStep = True
        else:
            if self.feas <= self.th_stop:
                if self.dPhi > self.th_acceptNegStep * self.dPhiexp:
                    acceptStep = True
            elif self.dV > self.th_acceptNegStep * self.dVexp:
                acceptStep = True
        # TODO: accept dImpr > 0 when allocated time has been reached (c++)
        if self.stepLength <= self.th_acceptMinStep and self.dImpr > 0.0:
            acceptStep = True
        return acceptStep

    # This is a virtual function
    def updateMeritFunction(self):
        # Update the penalty parameter for computing the merit function and its
        # directional derivative For more details see Section 3.3 of "An Interior
        # Point Algorithm for Large Scale Nonlinear Programming"
        if self.iter == 0 and self.zero_upsilon:
            self.upsilon = 0.0
        if (
            self.feas >= self.th_minffeas
            and self.dyn_solver != crocoddyl.DynamicsSolverType.SingleShoot
        ):
            # We incorporate a barrier-reduction strategy that still maintains a
            # the directional derivative be sufficiently negative (as explained
            # in Nocedal's texbook page 542) while allowing for a reduction when
            # it is possible.
            self.upsilon = max(
                self.upsilon * self.upsilon_decFactor,
                self.dVexp_full / ((1 - self.rho) * self.feas),
            )

    def backwardPass(self):
        self.Vx[-1][:] = self.problem.terminalData.Lx
        self.Vxx[-1][:, :] = self.problem.terminalData.Lxx
        if self.preg != 0.0:
            ndx = self.problem.terminalModel.state.ndx
            self.Vxx[-1][range(ndx), range(ndx)] += self.preg
        # Compute and store the Vxx_f gradient
        self.Vxx_f[-1] = self.Vxx[-1] @ self.fs[-1]
        for t, (m, d) in rev_enumerate(
            zip(self.problem.runningModels, self.problem.runningDatas)
        ):
            # Update action-value function
            self.computeActionValueFunction(t, m, d)
            # Update policy
            self.computePolicy(t)
            # Update value function
            self.computeValueFunction(t, m)
            raiseIfNan(self.Vxx[t], ArithmeticError("backward error"))
            raiseIfNan(self.Vx[t], ArithmeticError("backward error"))

    def batchPass(self):
        # Update the direction and feed-forward term to account for the terminal constraint.
        # To do so, we first compute the unscaled search direction accounting for
        # the terminal constraint as follows
        m = self.problem.terminalModel
        ndx, nh_T = m.state.ndx, m.nh_T
        self.Vxc[-1][:, :] = -self.problem.terminalData.Hx.reshape(nh_T, ndx).T
        for t, d in rev_enumerate(self.problem.runningDatas):
            # Update action-value function associated with the batch's constraint-to-go conditions
            self.computeBatchActionValueFunction(t, d)
            # Update feed-forward policy associated with the batch's constraint-to-go conditions
            self.computeBatchPolicy(t)
            # Update value function associated with the batch's constraint-to-go conditions
            self.computeBatchValueFunction(t)
        self.dXc[0][:, :] *= np.zeros((ndx, nh_T))
        for t, (m, d) in enumerate(
            zip(self.problem.runningModels, self.problem.runningDatas)
        ):  # sequence
            ndx, nu = m.state.ndx, m.nu
            self.dUc[t][:, :] = -self.Kc[t]
            self.dUc[t][:, :] -= self.K[t] @ self.dXc[t]
            self.dXc[t + 1][:, :] = d.Fx @ self.dXc[t]
            self.dXc[t + 1][:, :] += d.Fu.reshape(ndx, nu) @ self.dUc[t]

    def updateDir(self):
        d = self.problem.terminalData
        nh_T = self.problem.terminalModel.nh_T
        self.dHc[:, :] = d.Hx @ self.dXc[-1]
        self.hc[:] = d.h
        self.hc[:] += d.Hx @ self.dxs[-1]
        if (
            self.term_solver == crocoddyl.EqualitySolverType.LuNull
            or self.term_solver == crocoddyl.EqualitySolverType.QrNull
        ):
            # Resizing dHc-related data
            self.dHc_rank = np.linalg.matrix_rank(self.dHc)
            self.Yc = np.resize(self.Yc, (nh_T, self.dHc_rank))
            self.Yhc = np.resize(self.Yhc, self.dHc_rank)
            self.dHcY = np.resize(self.Yc, (nh_T, self.dHc_rank))
            self.YdHcY = np.resize(self.Yc, (self.dHc_rank, self.dHc_rank))
            self.YdHcY_inv_YHc = np.resize(self.Yhc, self.dHc_rank)
            # Compute terminal multiplier using nullspace parametrization. Instead of parametrizing Hx,
            # we opt to equivalent parametrize dHc. This approach is much efficient.
            self.computeNullTerminalMultiplier()
        else:
            try:
                self.YdHcY_llt = scl.cho_factor(self.dHc)
            except scl.LinAlgError:
                raise ArithmeticError("backward error")
            self.beta_plus[:] = scl.cho_solve(self.YdHcY_llt, self.hc)
        # Finally, we update the feed-forward term and search direction.
        for t in range(self.problem.T):  # in parallel
            self.dus[t][:] -= self.dUc[t] @ self.beta_plus
            self.dxs[t + 1][:] -= self.dXc[t + 1] @ self.beta_plus
            self.k[t] -= self.Kc[t] @ self.beta_plus
            self.Quuk[t] = self.Quu[t] @ self.k[t]

    # This is virtual function
    def computeActionValueFunction(self, t, model, data):
        ndx, nu = model.state.ndx, model.nu
        Vx_p = self.Vx[t + 1]
        Vxx_p = self.Vxx[t + 1]
        FxTVxx_p = data.Fx.T @ Vxx_p
        # Update Vx with Vxx f term
        Vx_p += self.Vxx_f[t + 1]
        self.Qx[t][:] = data.Lx + data.Fx.T @ Vx_p
        self.Qxx[t][:, :] = data.Lxx + FxTVxx_p @ data.Fx
        if self.preg != 0.0:
            self.Qxx[t][range(ndx), range(ndx)] += self.preg
        if nu != 0:
            FuTVxx_p = data.Fu.T @ Vxx_p
            self.Qu[t][:] = data.Lu + data.Fu.T @ Vx_p
            self.Quu[t][:, :] = data.Luu + FuTVxx_p @ data.Fu
            self.Qxu[t][:, :] = (data.Lxu + FxTVxx_p @ data.Fu).reshape((ndx, nu))
            if self.preg != 0.0:
                self.Quu[t][range(nu), range(nu)] += self.preg
        # Return value
        Vx_p -= self.Vxx_f[t + 1]

    # This is virtual function
    def computeBatchActionValueFunction(self, t, data):
        model = self.problem.runningModels[t]
        ndx, nu = model.state.ndx, model.nu
        self.Quc[t][:, :] = data.Fu.reshape(ndx, nu).T @ self.Vxc[t + 1]
        self.Qxc[t][:, :] = data.Fx.T @ self.Vxc[t + 1]

    # This is virtual function
    def computePolicy(self, t):
        nu = self.problem.runningModels[t].nu
        try:
            if nu > 0:
                self.Quu_llt[t] = scl.cho_factor(self.Quu[t])
                self.K[t][:, :] = scl.cho_solve(self.Quu_llt[t], self.Qxu[t].T)
                self.k[t][:] = scl.cho_solve(self.Quu_llt[t], self.Qu[t])
            else:
                pass
        except scl.LinAlgError:
            raise ArithmeticError("backward error")

    # This is virtual function
    def computeBatchPolicy(self, t):
        nu = self.problem.runningModels[t].nu
        if nu > 0:
            self.Kc[t][:] = scl.cho_solve(self.Quu_llt[t], self.Quc[t])

    # This is virtual function
    def computeValueFunction(self, t, model):
        nu = model.nu
        self.Vx[t][:] = self.Qx[t]
        self.Vxx[t][:, :] = self.Qxx[t]
        if nu != 0:
            self.Quuk[t][:] = self.Quu[t] @ self.k[t]
            self.Vx[t][:] -= self.K[t].T @ self.Qu[t]
            self.Vxx[t][:, :] -= self.Qxu[t] @ self.K[t]
        self.Vxx[t][:, :] = 0.5 * (self.Vxx[t][:, :] + self.Vxx[t][:, :].T)
        self.Vxx_f[t] = self.Vxx[t] @ self.fs[t]

    # This is virtual function
    def computeBatchValueFunction(self, t):
        self.Vxc[t][:, :] = self.Qxc[t]
        self.Vxc[t][:, :] -= self.Qxu[t] @ self.Kc[t]

    def linearRollout(self):
        self.dxs[0][:] = self.fs[0]
        for t, (m, d) in enumerate(
            zip(self.problem.runningModels, self.problem.runningDatas)
        ):
            ndx, nu = m.state.ndx, m.nu
            self.dxs[t + 1][:] = d.Fx @ self.dxs[t]
            self.dxs[t + 1][:] += self.fs[t + 1]
            if nu != 0:
                self.dus[t][:] = -self.k[t]
                self.dus[t][:] -= self.K[t] @ self.dxs[t]
                self.dxs[t + 1][:] += d.Fu.reshape(ndx, nu) @ self.dus[t]

    def feasShootForwardPass(self, stepLength, warning="ignore"):
        xs, us = self.xs, self.us
        xtry, utry = self.xs_try, self.us_try
        self.cost_try = 0.0
        xtry[0] = self.problem.runningModels[0].state.integrate(
            xs[0], stepLength * self.dxs[0]
        )
        self.fs_try[0] = self.fs[0] * (1 - stepLength)
        for t, (m, d) in enumerate(
            zip(self.problem.runningModels, self.problem.runningDatas)
        ):
            self.dx[t] = m.state.diff(xs[t], xtry[t])
            if m.nu != 0:
                utry[t] = us[t] - stepLength * self.k[t]
                utry[t] -= self.K[t] @ self.dx[t]
                with warnings.catch_warnings():
                    warnings.simplefilter(warning)
                    m.calc(d, xtry[t], utry[t])
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter(warning)
                    m.calc(d, xtry[t])
            self.fs_try[t + 1] = self.fs[t + 1] * (1 - stepLength)
            xtry[t + 1] = m.state.integrate(d.xnext, -self.fs_try[t + 1])
            self.cost_try += d.cost
            raiseIfNan(self.cost_try, ArithmeticError("forward error"))
            raiseIfNan(xtry[t + 1], ArithmeticError("forward error"))
        with warnings.catch_warnings():
            warnings.simplefilter(warning)
            self.problem.terminalModel.calc(self.problem.terminalData, xtry[-1])
            self.cost_try += self.problem.terminalData.cost
        raiseIfNan(self.cost_try, ArithmeticError("forward error"))

    def multiShootForwardPass(self, stepLength, warning="ignore"):
        xs, us = self.xs, self.us
        xtry, utry = self.xs_try, self.us_try
        # Update the dynamics gap for each node
        self.cost_try = 0.0
        xtry[0] = self.problem.runningModels[0].state.integrate(
            xs[0], stepLength * self.dxs[0]
        )
        self.fs_try[0] = self.fs[0] * (1 - stepLength)
        for t, (m) in enumerate(self.problem.runningModels):  # in parallel
            xtry[t + 1] = m.state.integrate(xs[t + 1], stepLength * self.dxs[t + 1])
        for t, (m, d) in enumerate(
            zip(self.problem.runningModels, self.problem.runningDatas)
        ):  # in parallel
            if m.nu != 0:
                utry[t] = us[t] + stepLength * self.dus[t]
                with warnings.catch_warnings():
                    warnings.simplefilter(warning)
                    m.calc(d, xtry[t], utry[t])
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter(warning)
                    m.calc(d, xtry[t])
            self.fs_try[t + 1] = m.state.diff(xtry[t + 1], d.xnext)
            self.cost_try += d.cost
            raiseIfNan(self.cost_try, ArithmeticError("forward error"))
            raiseIfNan(d.xnext, ArithmeticError("forward error"))
        with warnings.catch_warnings():
            warnings.simplefilter(warning)
            self.problem.terminalModel.calc(self.problem.terminalData, xtry[-1])
            self.cost_try += self.problem.terminalData.cost
        raiseIfNan(self.cost_try, ArithmeticError("forward error"))

    def hybridShootForwardPass(self, stepLength, warning="ignore"):
        xs, us = self.xs, self.us
        xtry, utry = self.xs_try, self.us_try
        # Update the initial state of each shooting node
        xtry[0] = self.problem.runningModels[0].state.integrate(
            xs[0], stepLength * self.dxs[0]
        )
        for Ti in self.Ts[1:]:  # in parallel
            m = self.problem.runningModels[Ti - 1]
            xtry[Ti] = m.state.integrate(xs[Ti], stepLength * self.dxs[Ti])
        # Perform the feasibility-driven nonlinear rollout for each shooting node
        self.cost_try = 0.0
        for i in range(len(self.Ts) - 1):  # in parallel
            Tinit, Tend = self.Ts[i], self.Ts[i + 1]
            for t in range(Tinit, Tend):  # in sequence
                m = self.problem.runningModels[t]
                d = self.problem.runningDatas[t]
                if m.nu != 0:
                    self.dx[t] = m.state.diff(xs[t], xtry[t])
                    utry[t] = us[t] - stepLength * self.k[t]
                    utry[t] -= self.K[t] @ self.dx[t]
                    with warnings.catch_warnings():
                        warnings.simplefilter(warning)
                        m.calc(d, xtry[t], utry[t])
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter(warning)
                        m.calc(d, xtry[t])
                self.cost_try += d.cost
                raiseIfNan(self.cost_try, ArithmeticError("forward error"))
                raiseIfNan(d.xnext, ArithmeticError("forward error"))
                if t + 1 != Tend:
                    self.fs_try[t + 1] = self.fs[t + 1] * (1 - stepLength)
                    xtry[t + 1] = m.state.integrate(d.xnext, -self.fs_try[t + 1])
        with warnings.catch_warnings():
            warnings.simplefilter(warning)
            self.problem.terminalModel.calc(self.problem.terminalData, xtry[-1])
        self.cost_try += self.problem.terminalData.cost
        raiseIfNan(self.cost_try, ArithmeticError("forward error"))
        # Update the initial gap of each shooting node
        self.fs_try[0] = self.fs[0] * (1 - stepLength)
        for Ti in self.Ts[1:]:  # in parallel
            m = self.problem.runningModels[Ti - 1]
            d = self.problem.runningDatas[Ti - 1]
            self.fs_try[Ti] = m.state.diff(xtry[Ti], d.xnext)

    def singleShootForwardPass(self, stepLength, warning="ignore"):
        xs, us = self.xs, self.us
        xtry, utry = self.xs_try, self.us_try
        self.cost_try = 0.0
        xnext = self.problem.x0
        for t, (m, d) in enumerate(
            zip(self.problem.runningModels, self.problem.runningDatas)
        ):
            xtry[t] = xnext.copy()
            if m.nu != 0:
                self.dx[t] = m.state.diff(xs[t], xtry[t])
                utry[t] = us[t] - stepLength * self.k[t]
                utry[t] -= self.K[t] @ self.dx[t]
                with warnings.catch_warnings():
                    warnings.simplefilter(warning)
                    m.calc(d, xtry[t], utry[t])
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter(warning)
                    m.calc(d, xtry[t])
            xnext = d.xnext
            self.cost_try += d.cost
            raiseIfNan(self.cost_try, ArithmeticError("forward error"))
            raiseIfNan(xnext, ArithmeticError("forward error"))
        xtry[-1] = xnext.copy()
        with warnings.catch_warnings():
            warnings.simplefilter(warning)
            self.problem.terminalModel.calc(self.problem.terminalData, xtry[-1])
            self.cost_try += self.problem.terminalData.cost
        raiseIfNan(self.cost_try, ArithmeticError("forward error"))

    def computeNullTerminalMultiplier(self):
        self.Yc[:, :] = scl.orth(self.dHc)
        self.Yhc[:] = self.Yc.T @ self.hc
        self.dHcY[:, :] = self.dHc @ self.Yc
        self.YdHcY[:, :] = self.Yc.T @ self.dHcY
        try:
            self.YdHcY_llt = scl.cho_factor(self.YdHcY)
        except scl.LinAlgError:
            raise ArithmeticError("backward error")
        self.YdHcY_inv_YHc[:] = scl.cho_solve(self.YdHcY_llt, self.Yhc)
        self.beta_plus[:] = self.Yc @ self.YdHcY_inv_YHc

    def updateCandidate(self):
        self.cost = copy.deepcopy(self.cost_try)
        if self.dyn_solver == crocoddyl.DynamicsSolverType.SingleShoot:
            self.ffeas = 0.0
        else:
            self.ffeas = copy.deepcopy(self.ffeas_try)
        self.gfeas = copy.deepcopy(self.gfeas_try)
        self.hfeas = copy.deepcopy(self.hfeas_try)
        self.merit = self.cost + self.upsilon * (self.ffeas + self.gfeas + self.hfeas)

    def decreaseRegularizationCriteria(self):
        return (
            self.stepLength >= self.th_stepDec and abs(self.dImpr) > self.th_minImprove
        )

    def increaseRegularizationCriteria(self):
        return (
            self.stepLength >= self.th_stepInc and abs(self.dImpr) <= self.th_minImprove
        ) or not self.acceptStep

    def decreaseRegularization(self):
        self.preg /= self.reg_decFactor
        self.preg = max(self.preg, self.reg_min)
        self.dreg = self.preg

    def increaseRegularization(self):
        self.preg *= self.reg_incFactor
        self.preg = min(self.preg, self.reg_max)
        self.dreg = self.preg

    # This is virtual function
    def allocateData(self):
        models = [*self.problem.runningModels.tolist(), self.problem.terminalModel]
        # Value function and improvement data
        self.Vxx = [np.zeros([m.state.ndx, m.state.ndx]) for m in models]
        self.Vx = [np.zeros([m.state.ndx]) for m in models]
        self.Vxx_f = [np.zeros([m.state.ndx]) for m in models]
        self.Lxx_dx = [np.zeros([m.state.ndx]) for m in models]
        self.Luu_du = [np.zeros([m.nu]) for m in self.problem.runningModels]
        self.Lxu_du = [np.zeros([m.state.ndx]) for m in self.problem.runningModels]
        # Action-value function data
        self.Qxx = [
            np.zeros([m.state.ndx, m.state.ndx]) for m in self.problem.runningModels
        ]
        self.Qxu = [np.zeros([m.state.ndx, m.nu]) for m in self.problem.runningModels]
        self.Quu = [np.zeros([m.nu, m.nu]) for m in self.problem.runningModels]
        self.Qx = [np.zeros([m.state.ndx]) for m in self.problem.runningModels]
        self.Qu = [np.zeros([m.nu]) for m in self.problem.runningModels]
        self.Quuk = [np.zeros(m.nu) for m in self.problem.runningModels]
        self.Quu_llt = [None] * self.problem.T
        # Policy data
        self.K = [np.zeros([m.nu, m.state.ndx]) for m in self.problem.runningModels]
        self.k = [np.zeros([m.nu]) for m in self.problem.runningModels]
        # Next state, control and gaps data
        self.dx = [np.zeros([m.state.ndx]) for m in self.problem.runningModels]
        self.dxs = [np.zeros([m.state.ndx]) for m in models]
        self.dus = [np.zeros([m.nu]) for m in self.problem.runningModels]
        # Terminal constraint data
        nh_T = self.problem.terminalModel.nh_T
        self.Qxc = [np.zeros([m.state.ndx, nh_T]) for m in self.problem.runningModels]
        self.Quc = [np.zeros([m.nu, nh_T]) for m in self.problem.runningModels]
        self.Vxc = [np.zeros([m.state.ndx, nh_T]) for m in models]
        self.dXc = [np.zeros([m.state.ndx, nh_T]) for m in models]
        self.dUc = [np.zeros([m.nu, nh_T]) for m in self.problem.runningModels]
        self.Kc = [np.zeros([m.nu, nh_T]) for m in self.problem.runningModels]
        self.dHc = np.zeros((nh_T, nh_T))
        self.hc = np.zeros(nh_T)
        self.Yc = np.zeros((nh_T, nh_T))
        self.Yhc = np.zeros(nh_T)
        self.dHcY = np.zeros((nh_T, nh_T))
        self.YdHcY = np.zeros((nh_T, nh_T))
        self.YdHcY_llt = None
        self.YdHcY_inv_YHc = np.zeros((nh_T, nh_T))
        self.beta_plus = np.zeros(nh_T)

    def setDynamicsSolver(self, type, Tshoot=0):
        if type == crocoddyl.DynamicsSolverType.HybridShoot and Tshoot <= 0:
            print(
                "Invalid argument: the number of rollout nodes should be bigger than 0."
            )
            return
        self.dyn_solver = type
        if type == crocoddyl.DynamicsSolverType.HybridShoot:
            self.Tshoot = Tshoot
            self.Ts = [0]
            for i in range(0, self.problem.T, self.Tshoot):
                if i + self.Tshoot < self.problem.T:
                    self.Ts.append(i + self.Tshoot)
                else:
                    self.Ts.append(self.problem.T)
