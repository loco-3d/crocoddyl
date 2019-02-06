import pinocchio
import numpy as np
from collections import OrderedDict
from locomote import CubicHermiteSpline
from crocoddyl import m2a, a2m, EmptyClass
from crocoddyl import CostModelSum,CostModelState,CostModelControl,CostModelForce,\
  CostModelFrameVelocity,CostModelFramePlacement,CostModelCoM,\
  ImpulseModelMultiple,DifferentialActionModelFloatingInContact,IntegratedActionModelEuler,\
  StatePinocchio,ContactModel6D,ContactModelMultiple,\
  ActuationModelFreeFloating, ActivationModelWeightedQuad

class EESplines(OrderedDict):
  def __add__(self,other):
    return EESplines([[patch, self[patch] + other[patch]] for patch in self.keys()]);
  def __sub__(self,other):
    return EESplines([[patch, self[patch] - other[patch]] for patch in self.keys()]);

class CentroidalPhi:

  class zero():
    def __init__(self, dim):
      self.dim = dim
    def eval(self, t): return (np.matrix(np.zeros((self.dim,1))),
                               np.matrix(np.zeros((self.dim,1))))

  def __init__(self, patch_names, com_vcom=None, hg=None, forces=None):
    if com_vcom is None: self.com_vcom = self.zero(6)
    else:                self.com_vcom = com_vcom
    if hg is None:       self.hg = self.zero(6)
    else:                self.hg = hg
    if forces is None:   self.forces = EESplines()
    else:                self.forces = forces
    
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

def createSwingTrajectories(rmodel, rdata, x, contact_patches, dt):
  p = OrderedDict(); m = OrderedDict()
  N = len(x)
  abscissa = a2m(np.linspace(0.,dt*(N-1), N))
  swing_ref = EESplines()
  for patch in contact_patches.keys():
    p = np.zeros((3,N));  m = np.zeros((3,N))
    for i in xrange(N):
      q = x[i][:rmodel.nq]
      v = x[i][-rmodel.nv:]
      pinocchio.forwardKinematics(rmodel, rdata, q, v)
      p[:,i] = m2a(pinocchio.updateFramePlacement(rmodel, rdata,
                                  rmodel.getFrameId(contact_patches[patch])).translation)
      m[:,i] = m2a(pinocchio.getFrameVelocity(rmodel, rdata,
                                  rmodel.getFrameId(contact_patches[patch])).linear)
    swing_ref.update([[patch, CubicHermiteSpline(abscissa, p, m)]])
  return swing_ref

def createMultiphaseShootingProblem(rmodel, rdata, patch_name_map, cs, phi_c, swing_ref, dt):
  """
  Create a Multiphase Shooting problem from the output of the centroidal solver.
  
  :params rmodel: robot model of type pinocchio::model
  :params rdata: robot data of type pinocchio::data
  :params patch_name_map: dictionary mapping of contact_patch->robot_framename. e.g. "LF_Patch":leg_6_joint
  :params cs: contact sequence of type locomote::ContactSequenceHumanoid
  :params phi_c: centroidal reference of type CentroidalPhi
  :params swing_ref: end-effector trajectories of type EESplines
  :params dt: Scalar timestep between nodes.

  :returns list of IntegratedActionModels
  """

  #Define Cost weights
  w = EmptyClass()
  w.com = 100.;    w.state = 0.1;    w.control = 0.001;
  w.swing_patch = 100.; w.forces = 0.01;

  #Define state cost vector for WeightedActivation
  w.xweight = np.array([0]*6+[0.01]*(rmodel.nv-6)+[10.]*rmodel.nv)
  #for patch in swing_patch:  w.swing_patch.append(100.);

  problem_models = []
  actuationff = ActuationModelFreeFloating(rmodel)
  State = StatePinocchio(rmodel)  

  for phase in cs.contact_phases:
    t0 = phase.time_trajectory[0];    t1 = phase.time_trajectory[-1]
    N = int(round((t1-t0)/dt))+1
    contact_model = ContactModelMultiple(rmodel)

    # Add contact constraints for the active contact patches.
    # Add SE3 cost for the non-active contact patches.
    swing_patch = [];
    for patch in patch_name_map.keys():
      if getattr(phase, patch).active:
        active_contact = ContactModel6D(rmodel,
                                        frame=rmodel.getFrameId(patch_name_map[patch]),
                                        ref = getattr(phase,patch).placement)
        contact_model.addContact(patch, active_contact)
      else:
        swing_patch.append(patch)

    # Define the cost and action models for each timestep in the contact phase.
    for t in np.linspace(t0,t1,N)[:-1]:
      cost_model = CostModelSum(rmodel, actuationff.nu);
      #CoM Cost
      cost_com = CostModelCoM(rmodel, ref=m2a(phi_c.com_vcom.eval(t)[0][:3,:]),
                              nu = actuationff.nu);
      cost_model.addCost("CoM",cost_com, w.com)

      #Forces Cost
      for patch in contact_model.contacts.keys():
        cost_force = CostModelForce(rmodel,
                                    contactModel=contact_model.contacts[patch],
                                    ref =m2a(phi_c.forces[patch].eval(t)[0]),
                                    nu = actuationff.nu);
        cost_model.addCost("forces_"+patch,cost_force, w.forces)
      #Swing patch cost
      for patch in swing_patch:
        cost_swing = CostModelFramePlacement(rmodel,
                                             frame=rmodel.getFrameId(patch_name_map[patch]),
                                             ref=pinocchio.SE3(np.identity(3),
                                                               swing_ref[patch].eval(t)[0]),
                                             nu=actuationff.nu)
        cost_model.addCost("swing_"+patch, cost_swing, w.swing_patch)

      #State Regularization
      cost_regx = CostModelState(rmodel, State, ref=rmodel.defaultState,
                                 nu = actuationff.nu,
                                 activation=ActivationModelWeightedQuad(w.xweight**2))
      cost_model.addCost("regx", cost_regx, w.state)
      #Control Regularization
      cost_regu = CostModelControl(rmodel, nu = actuationff.nu)
      cost_model.addCost("regu", cost_regu, w.control)
      
      dmodel = DifferentialActionModelFloatingInContact(rmodel, actuationff,
                                                        contact_model, cost_model)
      imodel = IntegratedActionModelEuler(dmodel)
      problem_models.append(imodel)
  return problem_models

def createPhiFromContactSequence(rmodel, rdata, cs, patch_names):
  #Note that centroidal plannar returns the forces in the sequence RF,LF,RH,LH.
  range_def = OrderedDict()
  range_def.update([["RF_patch", range(0,6)]]);
  range_def.update([["LF_patch", range(6,12)]])
  range_def.update([["RH_patch", range(12,18)]])
  range_def.update([["LH_patch", range(18,24)]])
  
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
  phi_c = EmptyClass()
  phi_c.f = OrderedDict(); phi_c.df = OrderedDict();
  for patch in patch_names:
    phi_c.f.update([[patch, np.zeros((N,6))]])
    phi_c.df.update([[patch, np.zeros((N,6))]])
  
  phi_c.com_vcom = np.zeros((N,6)); phi_c.vcom_acom =np.zeros((N, 6));
  phi_c.hg = np.zeros((N, 6));      phi_c.dhg = np.zeros((N, 6));

  n = 0;
  for i,spl in enumerate(cs.ms_interval_data[:-1]):
    x = m2a(spl.state_trajectory)
    dx = m2a(spl.dot_state_trajectory)
    u = m2a(spl.control_trajectory)
    nt = len(x)

    tt = t_traj[n:n+nt]
    phi_c.com_vcom[n:n+nt,:] = x[:,:6];     phi_c.vcom_acom[n:n+nt,:] = dx[:,:6]
    phi_c.hg[n:n+nt, 3:] = x[:,-3:];        phi_c.dhg[n:n+nt, 3:] = dx[:,-3:]
    phi_c.hg[n:n+nt, :3] = mass*x[:,3:6];   phi_c.dhg[n:n+nt, :3] = mass*dx[:,3:6]

    #--Control output of MUSCOD is a discretized piecewise polynomial.
    #------Convert the one piece to Points and Derivatives.
    poly_u, dpoly_u = polyfitND(tt, u, deg=3, full=True, eps=1e-5)

    f_poly = lambda t,r: np.array([poly_u[i](t) for i in r])
    f_dpoly = lambda t,r: np.array([dpoly_u[i](t) for i in r])
    for patch in patch_names:
      phi_c.f[patch][n:n+nt,:]  = np.array([f_poly(t,range_def[patch]) for t in tt])
      phi_c.df[patch][n:n+nt,:] = np.array([f_dpoly(t,range_def[patch]) for t in tt])

    n += nt
  
  centroidal = CentroidalPhi(patch_names)
  centroidal.com_vcom = CubicHermiteSpline(a2m(t_traj), a2m(phi_c.com_vcom),
                                           a2m(phi_c.vcom_acom))
  centroidal.hg = CubicHermiteSpline(a2m(t_traj), a2m(phi_c.hg), a2m(phi_c.dhg))

  for patch in patch_names:
    centroidal.forces[patch] = CubicHermiteSpline(a2m(t_traj), a2m(phi_c.f[patch]),
                                                  a2m(phi_c.df[patch]))
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
