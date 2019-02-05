import numpy as np
from locomote import CubicHermiteSpline
import pinocchio
from crocoddyl import m2a, a2m
from locomote import CubicHermiteSpline

class CentroidalPhi():

  class zero():
    def __init__(self, dim):
      self.dim = dim
    def eval(self, t): return (np.matrix(np.zeros((self.dim,1))),
                               np.matrix(np.zeros((self.dim,1))))
    
  def __init__(self, com_vcom=None, hg=None, forces=None):
    if com_vcom is None: self.com_vcom = self.zero(6)
    else:                  self.com_vcom = com_vcom
    if hg is None:       self.hg = self.zero(6)
    else:                  self.hg = hg      
    if forces is None:   self.forces = self.zero(12)
    else:                  self.forces = forces
      
  def __add__(self,other):
    if isinstance(self.com_vcom,self.zero):
      return CentroidalPhi(other.com_vcom, other.hg, other.forces)
    if isinstance(other.com_vcom,other.zero):
      return CentroidalPhi(self.com_vcom, self.hg, self.forces)
    return CentroidalPhi(self.com_vcom+other.com_vcom,self.hg+other.hg,
                      self.forces+other.forces)
  def __sub__(self,other):
    if isinstance(self.com_vcom,self.zero):
      return NotImplementedError
    if isinstance(other.com_vcom,other.zero):    
      return NotImplementedError
    return CentroidalPhi(self.com_vcom-other.com_vcom,self.hg-other.hg,
                      self.forces-other.forces)

  
def createPhiFromContactSequence(rmodel, rdata, cs):
  mass = pinocchio.crba(rmodel, rdata, pinocchio.neutral(rmodel))[0,0]
  t_traj =None
  forces_traj =None
  dforces_traj =None
  x_traj =None
  dx_traj=None
  #-----Get Length of Timeline------------------------
  t_traj = []
  for spl in cs.ms_interval_data[:-1]:
    t_traj += list(spl.time_trajectory)
  t_traj = np.array(t_traj)
  N = len(t_traj)

  #------Get values of state and control--------------
  cc = lambda t:0
  cc.f = np.zeros((N, 12)); cc.df = np.zeros((N,12));
  cc.com_vcom = np.zeros((N,6)); cc.vcom_acom =np.zeros((N, 6));
  cc.hg = np.zeros((N, 6));      cc.dhg = np.zeros((N, 6));

  n = 0;
  f_rearrange = range(6,12)+range(0,6)
  for i,spl in enumerate(cs.ms_interval_data[:-1]):
    x = m2a(spl.state_trajectory)
    dx = m2a(spl.dot_state_trajectory)
    u = m2a(spl.control_trajectory)
    nt = len(x)

    tt = t_traj[n:n+nt]
    cc.com_vcom[n:n+nt,:] = x[:,:6];     cc.vcom_acom[n:n+nt,:] = dx[:,:6]
    cc.hg[n:n+nt, 3:] = x[:,-3:];        cc.dhg[n:n+nt, 3:] = dx[:,-3:]
    cc.hg[n:n+nt, :3] = mass*x[:,3:6];   cc.dhg[n:n+nt, :3] = mass*dx[:,3:6]

    #--Control output of MUSCOD is a discretized piecewise polynomial.
    #------Convert the one piece to Points and Derivatives.
    poly_u, dpoly_u = polyfitND(tt, u, deg=3, full=True, eps=1e-5)
    #Note that centroidal plannar returns the forces in the sequence RF,LF,RH,LH.
    #The wholebody solver follows the urdf parsing given by LF,RF,LH,RH
    f_poly = lambda t: np.array([poly_u[i](t) for i in f_rearrange])
    f_dpoly = lambda t: np.array([dpoly_u[i](t) for i in f_rearrange])
    cc.f[n:n+nt,:] = np.array([f_poly(t) for t in tt])
    cc.df[n:n+nt,:] = np.array([f_dpoly(t) for t in tt])
    
    n += nt
  
  centroidal = CentroidalPhi()
  centroidal.com_vcom = CubicHermiteSpline(a2m(t_traj), a2m(cc.com_vcom), a2m(cc.vcom_acom))
  centroidal.hg = CubicHermiteSpline(a2m(t_traj), a2m(cc.hg), a2m(cc.dhg))
  centroidal.forces = CubicHermiteSpline(a2m(t_traj), a2m(cc.f), a2m(cc.df))

  return centroidal

def polyfitND(x, y, deg, eps, rcond=None, full=False, w=None, cov=False):
  """ Return the polynomial fitting (and its derivative)
  for a set of points x, y where y=p(x).
  This is a wrapper of np.polyfit for more than one dimensions.
  
  :params x: Points where polynomial is being computed. e.g. on a timeline.
  :params y: value of the curve at x, y=p(x). each row corresponds to a point.
  :params deg: degree of the polynomial being computed.
  :params eps: tolerance to confirm the fidelity of the polynomial fitting.
  :params rcond: condition number of the fit. Default None
  :params full: True to return the svd. False to return just coeffs. Default False.
  :params w: weights for y coordinates. Default None.
  :params cov:estimate and covariance of the estimate. Default False.

  :returns array p where each row corresponds to the coeffs of a dimension in format np.poly1d
  """
  assert len(x.shape)==1; #x is an array.
  p = np.array([None,]*y.shape[1])
  dp = np.array([None,]*y.shape[1])
  for i,y_a in enumerate(y.T):
    p_a , residual, _, _, _ = np.polyfit(x, np.asarray(y_a).squeeze(),
                                         deg, rcond, full, w, cov);
    p[i] = np.poly1d(p_a[:])
    dp[i] = np.poly1d(p_a[:]).deriv(1)
    assert(residual <=eps)
  return p, dp
