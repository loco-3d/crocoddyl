from sympy import Symbol, nan, zeros

T = 2
x0ref = Symbol('x0ref')

lx0_ = []
lu0_ = []
lxx = []
luu = []
lxu = []
lux = []
fx = []
fu = []
f = []

xg = []
ug = []
lx = []
lu = []

for t in range(T):
    for n in ['lx0_', 'lu0_', 'lxx', 'luu', 'lxu', 'fx', 'fu', 'f', 'xg', 'ug']:
        globals()[n].append(Symbol("%s%1d" % (n, t)))

    lux.append(lxu[-1])
    lx.append(lxx[-1] * xg[-1] + lxu[-1] * ug[-1] + lx0_[-1])
    lu.append(lux[-1] * xg[-1] + luu[-1] * ug[-1] + lu0_[-1])

xg.append(Symbol('xg%1d' % T))
lx0_.append(Symbol('lx0_%1d' % T))
lxx.append(Symbol('lxx%1d' % T))
lx.append(lxx[-1] * xg[-1] + lx0_[-1])

hess = zeros(2 * T + 1, 2 * T + 1)
grad = zeros(2 * T + 1, 1)
jac = zeros(T + 1, 2 * T + 1)
cval = zeros(T + 1, 1)

for t in range(T):
    hess[t, t] = lxx[t]
    hess[T + 1 + t, t] = lxu[t]
    hess[t, T + 1 + t] = lux[t]
    hess[T + 1 + t, T + 1 + t] = luu[t]

    grad[t] = lx[t]
    grad[T + 1 + t] = lu[t]

    jac[t + 1, t + 1] = 1
    jac[t + 1, t] = -fx[t]
    jac[t + 1, T + 1 + t] = -fu[t]

    cval[t + 1] = fx[t] * xg[t] + fu[t] * ug[t] + f[t] - xg[t + 1]

hess[T, T] = lxx[-1]
grad[T] = lx[-1]
jac[0, 0] = 1
cval[0] = x0ref - xg[0]

kkt = hess.col_insert(2 * T + 1, jac.T)
kkt2 = jac.col_insert(2 * T + 1, zeros(T + 1, T + 1))
kkt = kkt.row_insert(2 * T + 1, kkt2)

kktref = (-grad).row_insert(2 * T + 1, cval)

primaldual = kkt.inv() * kktref

dxkkt = []
dukkt = []
xkkt = []
ukkt = []
for t in range(T):
    dxkkt.append(primaldual[t].simplify())
    xkkt.append((xg[t] + dxkkt[t]).simplify())
    dukkt.append(primaldual[T + 1 + t].simplify())
    ukkt.append((ug[t] + dukkt[t]).simplify())

dxkkt.append(primaldual[T].simplify())
xkkt.append((xg[T] + dxkkt[T]).simplify())

# --- ddp ---
inv = lambda a: 1 / a

vx = [nan] * T + [lx[-1]]
vxx = [nan] * T + [lxx[-1]]
qx = [nan] * T
qu = [nan] * T
qxx = [nan] * T
qxu = [nan] * T
qux = [nan] * T
quu = [nan] * T
K = [nan] * T
k = [nan] * T

for t in reversed(range(T)):
    qx[t] = lx[t] + fx[t] * vx[t + 1] + fx[t] * vxx[t + 1] * (fx[t] * xg[t] + fu[t] * ug[t] + f[t] - xg[t + 1])
    qu[t] = lu[t] + fu[t] * vx[t + 1] + fu[t] * vxx[t + 1] * (fx[t] * xg[t] + fu[t] * ug[t] + f[t] - xg[t + 1])
    qxx[t] = lxx[t] + fx[t] * vxx[t + 1] * fx[t]
    qxu[t] = lxu[t] + fx[t] * vxx[t + 1] * fu[t]
    quu[t] = luu[t] + fu[t] * vxx[t + 1] * fu[t]
    qux[t] = qxu[t]

    # Annulation of du derivative: qu + qux dx + quu du => K=quu^-1 qux, k=quu^-1 qu
    K[t] = inv(quu[t]) * qux[t]
    k[t] = inv(quu[t]) * qu[t]

    # Substitution of K0,k0 in hamiltonian:
    vx[t] = qx[t] - qux[t] * k[t]
    vxx[t] = qxx[t] - qux[t] * K[t]

xddp = [nan] * (T + 1)
uddp = [nan] * T

xddp[0] = x0ref

for t in range(T):
    uddp[t] = (ug[t] - k[t] - K[t] * (xddp[t] - xg[t])).simplify()
    xddp[t + 1] = (fx[t] * xddp[t] + fu[t] * uddp[t] + f[t]).simplify()

assert (xddp[0] - xkkt[0] == 0)

# --- test ---
hard = {
    x0ref: 2.,
    lx0_[0]: 0.36015221,
    lu0_[0]: 0.37365342,
    lxx[0]: 0.09588336,
    lxu[0]: -0.13771499,
    luu[0]: 0.21301384,
    fx[0]: -0.40090531,
    fu[0]: 0.86021485,
    f[0]: 0.13771499,
    lx0_[1]: 0.36015221,
    lxx[1]: 0.09588336,
}

simple = {
    xg[0]: 0,
    xg[1]: 0,
    xg[2]: 0,
    ug[0]: 0,
    ug[1]: 0,
    lux[0]: 0,
    lux[1]: 0,
    fx[0]: 1,
    fx[1]: 1,
    # f[0]: 1,
    # f[1]: 1
}
print(ukkt[0].subs(simple).simplify().factor() - uddp[0].subs(simple).simplify().factor())

guess = {
    xg[0]: 3,
    ug[0]: 10,
    xg[1]: 19,
}
