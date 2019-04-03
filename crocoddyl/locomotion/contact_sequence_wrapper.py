from crocoddyl import m2a, a2m
from multicontact_api import CubicHermiteSpline
import pinocchio
from collections import OrderedDict
from spline_utils import polyfitND, findDuplicates, removeDuplicates
from centroidal_phi import CentroidalPhi, EESplines
import numpy as np

class ContactSequenceWrapper:
    """ Wraps the contactsequence obtained from locomotion with:
    1) list of contact-patches containing map of end-effector and frame-name in pinocchio model
    2) end-effector trajectories (swing trajectories)
    3) easier accessors for centroidal trajectories and stored in centroidalphi class.
    """
    def __init__(self, cs, ee_map, ee_splines=None):
        self.cs = cs  #contact sequence file from multicontact-api
        self.ee_map = ee_map   # map of contact patches
        self.ee_splines = ee_splines
        self.centroidalPhi = None
        self.phi_c = CentroidalPhi()

    def createEESplines(self, rmodel, rdata, xs, t_sampling=0.005):
        N = len(xs)
        abscissa = a2m(np.linspace(0.,t_sampling*(N-1), N))
        self.ee_splines = EESplines()
        for patch in self.ee_map.keys():
            p = np.zeros((3,N));  m = np.zeros((3,N))
            for i in xrange(N):
                q = a2m(xs[i][:rmodel.nq])
                v = a2m(xs[i][-rmodel.nv:])
                pinocchio.forwardKinematics(rmodel, rdata, q, v)
                p[:,i] = m2a(pinocchio.updateFramePlacement(rmodel, rdata,
                                                            rmodel.getFrameId(self.ee_map[patch])).translation)
                m[:,i] = m2a(pinocchio.getFrameVelocity(rmodel, rdata,
                                                        rmodel.getFrameId(self.ee_map[patch])).linear)
                self.ee_splines.update([[patch, CubicHermiteSpline(abscissa, p, m)]])
        return


    def createCentroidalPhi(self,rmodel, rdata):
        #centroidal planar (muscod) returns the forces in the sequence RF,LF,RH,LH.
        #TODO: make more generic
        range_def = OrderedDict()
        range_def.update([["RF_patch", range(0,6)]]);
        range_def.update([["LF_patch", range(6,12)]])
        range_def.update([["RH_patch", range(12,18)]])
        range_def.update([["LH_patch", range(18,24)]])

        patch_names = self.ee_map.keys()
        mass = pinocchio.crba(rmodel, rdata, pinocchio.neutral(rmodel))[0,0]
        t_traj =None
        forces_traj =None
        dforces_traj =None
        x_traj =None
        dx_traj=None
        #-----Get Length of Timeline------------------------
        t_traj = []
        for spl in self.cs.ms_interval_data[:-1]:
          t_traj += list(spl.time_trajectory)
        t_traj = np.array(t_traj)
        N = len(t_traj)

        #------Get values of state and control--------------
        phi_c_ = lambda t: 0
        phi_c_.f = OrderedDict(); phi_c_.df = OrderedDict();
        for patch in patch_names:
          phi_c_.f.update([[patch, np.zeros((N,6))]])
          phi_c_.df.update([[patch, np.zeros((N,6))]])

        phi_c_.com_vcom = np.zeros((N,6)); phi_c_.vcom_acom =np.zeros((N, 6));
        phi_c_.hg = np.zeros((N, 6));      phi_c_.dhg = np.zeros((N, 6));

        n = 0;
        for i,spl in enumerate(self.cs.ms_interval_data[:-1]):
          x = m2a(spl.state_trajectory)
          dx = m2a(spl.dot_state_trajectory)
          u = m2a(spl.control_trajectory)
          nt = len(x)

          tt = t_traj[n:n+nt]
          phi_c_.com_vcom[n:n+nt,:] = x[:,:6];     phi_c_.vcom_acom[n:n+nt,:] = dx[:,:6]
          phi_c_.hg[n:n+nt, 3:] = x[:,-3:];        phi_c_.dhg[n:n+nt, 3:] = dx[:,-3:]
          phi_c_.hg[n:n+nt, :3] = mass*x[:,3:6];   phi_c_.dhg[n:n+nt, :3] = mass*dx[:,3:6]

          #--Control output of MUSCOD is a discretized piecewise polynomial.
          #------Convert the one piece to Points and Derivatives.
          poly_u, dpoly_u = polyfitND(tt, u, deg=3, full=True, eps=1e-5)

          f_poly = lambda t,r: np.array([poly_u[i](t) for i in r])
          f_dpoly = lambda t,r: np.array([dpoly_u[i](t) for i in r])
          for patch in patch_names:
            phi_c_.f[patch][n:n+nt,:]  = np.array([f_poly(t,range_def[patch]) for t in tt])
            phi_c_.df[patch][n:n+nt,:] = np.array([f_dpoly(t,range_def[patch]) for t in tt])

          n += nt


        duplicates = findDuplicates(t_traj)

        phi_c_2 = lambda t: 0
        phi_c_2.f = OrderedDict(); phi_c_2.df = OrderedDict();
        for patch in patch_names:
          phi_c_2.f.update([[patch, removeDuplicates(phi_c_.f[patch], duplicates)]])
          phi_c_2.df.update([[patch, removeDuplicates(phi_c_.df[patch], duplicates)]])

        phi_c_2.com_vcom  =removeDuplicates(phi_c_.com_vcom,  duplicates)
        phi_c_2.vcom_acom =removeDuplicates(phi_c_.vcom_acom, duplicates)
        phi_c_2.hg        =removeDuplicates(phi_c_.hg,        duplicates)
        phi_c_2.dhg       =removeDuplicates(phi_c_.dhg,       duplicates)        
        t_traj            =removeDuplicates(t_traj,           duplicates)
        
        self.phi_c.com_vcom = CubicHermiteSpline(a2m(t_traj), a2m(phi_c_2.com_vcom),
                                                 a2m(phi_c_2.vcom_acom))
        self.phi_c.hg = CubicHermiteSpline(a2m(t_traj), a2m(phi_c_2.hg), a2m(phi_c_2.dhg))

        for patch in patch_names:
          self.phi_c.forces[patch] = CubicHermiteSpline(a2m(t_traj), a2m(phi_c_2.f[patch]),
                                                        a2m(phi_c_2.df[patch]))
        return
