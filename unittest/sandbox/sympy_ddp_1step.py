from sympy import Eq, Symbol, nan, solve, zeros

T = 1

lx0_ = []  # Gradient at x=0
lu0_ = []  # Gradient at u=0
lxx = []
luu = []
lxu = []
lux = []
fx = []
fu = []
f = []  # Dynamic drift (xnext + f = Fx x + Fu u)

xg = []  # Initial guess for x
ug = []  # Initial guess for u
lx = []  # Gradient computed at the initial guess
lu = []  # Gradient computed at the initial guess

for t in range(T):
    for n in ['lx0_', 'lu0_', 'lxx', 'luu', 'lxu', 'fx', 'fu', 'f', 'xg', 'ug']:
        globals()[n].append(Symbol("%s%1d" % (n, t)))

    xg[-1] = 0
    ug[-1] = 0
    lux.append(lxu[-1])
    lx.append(lxx[-1] * xg[-1] + lxu[-1] * ug[-1] + lx0_[-1])
    lu.append(lux[-1] * xg[-1] + luu[-1] * ug[-1] + lu0_[-1])

xg.append(Symbol('xg%1d' % T))
xg[-1] = 0
f.append(Symbol('f%1d' % T))
lx0_.append(Symbol('lx0_%1d' % T))
lxx.append(Symbol('lxx%1d' % T))
lx.append(lxx[-1] * xg[-1] + lx0_[-1])

# Create the KKT problem
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

    jac[t + 1, t + 1] = 1
    jac[t + 1, t] = -fx[t]
    jac[t + 1, T + 1 + t] = -fu[t]

    cval[t + 1] = fx[t] * xg[t] + fu[t] * ug[t] - (f[t + 1] + xg[t + 1])

hess[T, T] = lxx[-1]
grad[T] = lx[-1]
jac[0, 0] = 1
cval[0] = -f[0] - xg[0]

kkt = hess.col_insert(2 * T + 1, jac.T)
kkt2 = jac.col_insert(2 * T + 1, zeros(T + 1, T + 1))
kkt = kkt.row_insert(2 * T + 1, kkt2)

kktref = (-grad).row_insert(2 * T + 1, cval)

# Solve the KKT Problem
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

# --- ddp --- ----------------------------------------------------------------------
# --- ddp --- ----------------------------------------------------------------------
# --- ddp --- ----------------------------------------------------------------------


def inv(a):
    return 1 / a


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
    qx[t] = lx[t] + fx[t] * vx[t + 1] + fx[t] * vxx[t + 1] * (fx[t] * xg[t] + fu[t] * ug[t] - f[t + 1] - xg[t + 1])
    qu[t] = lu[t] + fu[t] * vx[t + 1] + fu[t] * vxx[t + 1] * (fx[t] * xg[t] + fu[t] * ug[t] - f[t + 1] - xg[t + 1])
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

xddp[0] = -f[0]

for t in range(T):
    uddp[t] = (ug[t] - k[t] - K[t] * (xddp[t] - xg[t])).simplify()
    xddp[t + 1] = (fx[t] * xddp[t] + fu[t] * uddp[t] - f[t + 1]).simplify()

assert (xddp[0] - xkkt[0] == 0)

assert (xddp[0] - xkkt[0] == 0)

# --- test ---
simple = {
    xg[0]: 0,
    xg[1]: 0,
    # xg[2]: 0,
    ug[0]: 0,
    # ug[1]: 0,
    lux[0]: 0,
    # lux[1]: 0,
    fx[0]: 1,
}

print(ukkt[0].subs(simple).simplify().factor() - uddp[0].subs(simple).simplify().factor())

# ----
unk = []
A = [[0 for i in range(5)] for j in range(5)]
letters = ['a', 'b', 'c', 'd', 'e']

for i in range(5):
    for j in range(i, 5):
        A[i][j] = '%s%d' % (letters[i], j)

for a_ in A:
    for a in a_:
        if a != 0:
            globals()[a] = Symbol(a)

unk += [a for a in reduce(lambda x, y: x + y, A, []) if a != 0]

U = Matrix(A)
A = [['s%d' % i if i == j else 0 for i in range(5)] for j in range(5)]

for a_ in A:
    for a in a_:
        if a is not 0: globals()[a] = Symbol(a)
unk += [a for a in reduce(lambda x, y: x + y, A, []) if a != 0]

D = Matrix(A)

#, a4:0, d4:0, a3:0, c3:0 })

### BACKWARD
R = (U * D * U.T).subs({a0: 1, b1: 1, c2: 1, d3: 1, e4: 1})
perm = [3, 4, 0, 1, 2]
kktb = kkt[perm, perm]

R0 = R
sol = {a0: 1, b1: 1, c2: 1, d3: 1, e4: 1}
for i in range(4, -1, -1):
    for j in range(i, -1, -1):
        R = R.subs(sol)
        res = solve(Eq(R[i, j], kktb[i, j]), *unk)
        #print "RES = ",res
        #assert(len(res)==1)
        if isinstance(res, list): res = res[0]
        sol.update(res)
        #print "SOL = ",sol

Df = [d[0, 0] for d in D.subs(sol).get_diag_blocks()]
solf = sol
print(Df)

# FORWARD
R = (U.T * D * U).subs({a0: 1, b1: 1, c2: 1, d3: 1, e4: 1})
perm = [2, 1, 0, 4, 3]
kktb = kkt[perm, perm]

R0 = R
sol = {a0: 1, b1: 1, c2: 1, d3: 1, e4: 1}
unk = list(reversed(unk))
for i in range(5):  # (4,-1,-1):
    for j in range(i + 1):  # (i,-1,-1):
        R = R.subs(sol)
        res = solve(Eq(R[i, j], kktb[i, j]), *unk)
        # print "RES = ",res
        # assert(len(res)==1)
        if isinstance(res, list):
            res = res[0]
        sol.update(res)
        # print "SOL = ",sol

Db = [d[0, 0] for d in D.subs(sol).get_diag_blocks()]
solb = sol

# U.subs(sol) and D.subs(sol) contains the LDLT decomposition of KKT
print(Db)
