from numpy import allclose, hstack, vstack, zeros
from quadprog import solve_qp
"""
Solve a Quadratic Program defined as
minimize  (1/2) * x.T * H * x + g.T * x
subject to
C * x <= d
A * x == b
The solution is saved and returned in QPSolution object
def <solver>Wrapper(H, g, C, d, A, b, initvals)
:param H : numpy.array
:param g : numpy.array
:param C : numpy.array
:param d : numpy.array
:param A : numpy.array, optional
:param b : numpy.array, optional
:param initvals : numpy.array, optional
"""


class QPSolution:
    """
    Solution of a Quadratic Program minimize f(x) subject to ineq and eq constraints.

    argmin: solution of the problem. x*
    optimum: value of the objective at argmin. f(x*)
    active: set of indices of the active constraints.
    dual: value of the lagrangian multipliers.
    niter: number of iterations needed to solve the problem.
    """
    def __init__(self, nx, nc):
        self.argmin = zeros(nx)
        self.optimum = 0.
        self.active = zeros(nx).astype(bool)
        self.dual = zeros(nc)
        self.niter = 0.


def quadprogWrapper(H, g, C=None, d=None, A=None, b=None, initvals=None):
    """
    Quadprog <https://pypi.python.org/pypi/quadprog/>.
    The quadprog solver only considers the lower entries of `H`, therefore it
    will use a wrong cost function if a non-symmetric matrix is provided.
    """
    assert (allclose(H, H.T, atol=1e-10))
    if initvals is not None:
        print("quadprog: note that warm-start values ignored by wrapper")
    if A is not None and C is not None:
        qp_C = -vstack([A, C]).T
        qp_b = -hstack([b, d])
        meq = A.shape[0] if A.ndim > 1 else 1
    elif A is None and C is not None:  # no equality constraint
        qp_C = -C.T
        qp_b = -d
        meq = 0
    elif A is not None and C is None:
        qp_C = A.T
        qp_b = b
        meq = A.shape[0] if A.ndim > 1 else 1
    else:  # if A is None and C is None:
        qp_C = None
        qp_b = None
        meq = 0

    nx = H.shape[0]
    nc = qp_C.shape[0]
    sol = QPSolution(nx, nc)
    """
    Quadprog API
    Solve a strictly convex quadratic program

    Minimize     1/2 x^T G x - a^T x
    Subject to   C.T x >= b
    """
    sol.argmin, sol.optimum, _, (sol.niter, _), sol.dual, iact = solve_qp(H, -g, qp_C, qp_b, meq)
    sol.dual *= -1.
    for i in iact:
        sol.active[i % nx] = True
    return sol
