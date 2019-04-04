import numpy as np
from crocoddyl import (ActionModelNumDiff, DifferentialActionModelLQR, IntegratedActionModelRK4, a2m,
                       get_attr_analytical, m2a)
from numpy.random import rand
from testutils import assertNumDiff, df_dx

np.set_printoptions(linewidth=np.nan, suppress=True)
# --------------frcom scipy.stats.ortho_group----------

# -------------------------------------------------------------------------------

nq = 10
nu = 5
nv = nq

dmodel = DifferentialActionModelLQR(nq, nu, driftFree=False)
ddata = dmodel.createData()
model = IntegratedActionModelRK4(dmodel)
data = model.createData()

x = model.State.rand()

# u = np.random.rand( model.nu )
u = rand(model.nu)
xn, c = model.calc(data, x, u)

model.timeStep = 1

mnum = ActionModelNumDiff(model, withGaussApprox=False)
dnum = mnum.createData()

model.calcDiff(data, x, u)


def get_k(q, v):
    x_ = np.vstack([q, v])
    model.calc(data, m2a(x_), u)
    return [a2m(ki) for ki in data.ki]


def get_ku(u):
    model.calc(data, x, m2a(u))
    return [a2m(ki) for ki in data.ki]


def get_xn(u):
    model.calc(data, x, m2a(u))
    return a2m(data.xnext)  # .copy()


def get_au(u):
    a, _ = model.differential.calc(data.differential[0], x, m2a(u))
    return a2m(a)


def get_y(q, v):
    x_ = np.vstack([q, v])
    model.calc(data, m2a(x_), u)
    return [a2m(y) for y in data.y]


dxn_du = df_dx(lambda _u: get_xn(_u), a2m(u))


def dk_du(i):
    return df_dx(lambda _u: get_ku(_u)[i], a2m(u))


def dk_dq(i):
    return df_dx(lambda _q: get_k(_q, a2m(x[nq:]))[i], a2m(x[:nq]))


def dk_dv(i):
    return df_dx(lambda _v: get_k(a2m(x[:nq]), _v)[i], a2m(x[nq:]))


def dy_dq(i):
    return df_dx(lambda _q: get_y(_q, a2m(x[nq:]))[i], a2m(x[:nq]))


def dy_dv(i):
    return df_dx(lambda _v: get_y(a2m(x[:nq]), _v)[i], a2m(x[nq:]))


def e_k(i):
    return data.dki_dx[i][:, :nv] - dk_dq(i)


assertNumDiff(data.Fu, dxn_du,
              1e4 * mnum.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
for i in range(4):
    assertNumDiff(data.dki_du[i], dk_du(i),
                  1e4 * mnum.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)

for i in range(4):
    assertNumDiff(data.dki_dx[i][:, :nv], dk_dq(i),
                  1e4 * mnum.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
    assertNumDiff(data.dki_dx[i][:, nv:], dk_dv(i),
                  1e4 * mnum.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)

for i in range(4):
    assertNumDiff(data.dy_dx[i][:, :nv], dy_dq(i),
                  1e4 * mnum.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
    assertNumDiff(data.dy_dx[i][:, nv:], dy_dv(i),
                  1e4 * mnum.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)

mnum.calcDiff(dnum, x, u)
assertNumDiff(data.Fx, dnum.Fx,
              1e2 * mnum.disturbance)  # threshold was 2.11e-6, is now 2.11e-6 (see assertNumDiff.__doc__)
assertNumDiff(data.Fu, dnum.Fu,
              1e2 * mnum.disturbance)  # threshold was 2.11e-6, is now 2.11e-6 (see assertNumDiff.__doc__)
assertNumDiff(data.Lu, dnum.Lu,
              1e4 * mnum.disturbance)  # threshold was 2.05e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(data.Lx, dnum.Lx,
              1e4 * mnum.disturbance)  # threshold was 2.05e-4, is now 2.11e-4 (see assertNumDiff.__doc__)

eps = mnum.disturbance
Lxx0 = df_dx(lambda _x: get_attr_analytical(_x, u, "Lx"), a2m(x), h=eps)

Lxu0 = df_dx(lambda _u: get_attr_analytical(x, _u, "Lx"), a2m(u), h=eps)

Luu0 = df_dx(lambda _u: get_attr_analytical(x, _u, "Lu"), a2m(u), h=eps)

assertNumDiff(Lxx0, data.Lxx,
              1e4 * mnum.disturbance)  # threshold was 1.45e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(Lxu0, data.Lxu,
              1e4 * mnum.disturbance)  # threshold was 1.45e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(Luu0, data.Luu,
              1e4 * mnum.disturbance)  # threshold was 1.45e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
