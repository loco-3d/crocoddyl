#include "crocoddyl/multibody/actions/free-fwddyn.hpp"
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/aba-derivatives.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>

namespace crocoddyl {

DifferentialActionModelFreeFwdDynamics::DifferentialActionModelFreeFwdDynamics(StateMultibody* const state,
                                                                               CostModelSum* const costs)
    : DifferentialActionModelAbstract(state, state->get_model()->nv, costs->get_nr()),
      costs_(costs),
      pinocchio_(state->get_model()),
      force_aba_(true) {}

DifferentialActionModelFreeFwdDynamics::~DifferentialActionModelFreeFwdDynamics() {}

void DifferentialActionModelFreeFwdDynamics::calc(boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                                  const Eigen::Ref<const Eigen::VectorXd>& x,
                                                  const Eigen::Ref<const Eigen::VectorXd>& u) {
  DifferentialActionDataFreeFwdDynamics* d = static_cast<DifferentialActionDataFreeFwdDynamics*>(data.get());

  const Eigen::VectorXd& q = x.head(nq_);
  const Eigen::VectorXd& v = x.tail(nv_);
  if (force_aba_) {
    d->xout = pinocchio::aba(*pinocchio_, d->pinocchio, q, v, u);
  } else {
  }

  pinocchio::forwardKinematics(*pinocchio_, d->pinocchio, q, v);
  pinocchio::updateFramePlacements(*pinocchio_, d->pinocchio);
  costs_->calc(d->costs, x, u);
  d->cost = d->costs->cost;
  // if u is None:
  //     u = self.unone
  // nq, nv = self.nq, self.nv
  // q = a2m(x[:nq])
  // v = a2m(x[-nv:])
  // tauq = a2m(u)
  // # --- Dynamics
  // if self.forceAba:
  //     data.xout[:] = pinocchio.aba(self.pinocchio, data.pinocchio, q, v, tauq).flat
  // else:
  //     pinocchio.computeAllTerms(self.pinocchio, data.pinocchio, q, v)
  //     data.M = data.pinocchio.M
  //     if hasattr(self.pinocchio, 'armature'):
  //         data.M[range(nv), range(nv)] += self.pinocchio.armature.flat
  //     data.Minv = np.linalg.inv(data.M)
  //     data.xout[:] = data.Minv * (tauq - data.pinocchio.nle).flat
  // # --- Cost
  // pinocchio.forwardKinematics(self.pinocchio, data.pinocchio, q, v)
  // pinocchio.updateFramePlacements(self.pinocchio, data.pinocchio)
  // data.cost = self.costs.calc(data.costs, x, u)
  // return data.xout, data.cost
}

void DifferentialActionModelFreeFwdDynamics::calcDiff(boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                                      const Eigen::Ref<const Eigen::VectorXd>& x,
                                                      const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc) {
  if (recalc) {
    calc(data, x, u);
  }

  DifferentialActionDataFreeFwdDynamics* d = static_cast<DifferentialActionDataFreeFwdDynamics*>(data.get());

  const Eigen::VectorXd& q = x.head(nq_);
  const Eigen::VectorXd& v = x.tail(nv_);
  if (force_aba_) {
    pinocchio::computeABADerivatives(*pinocchio_, d->pinocchio, q, v, u, d->ddq_dq, d->ddq_dv, d->ddq_dtau);
    d->Fx.leftCols(nv_) = d->ddq_dq;
    d->Fx.rightCols(nv_) = d->ddq_dv;
    d->Fu = d->ddq_dtau;
  } else {
  }

  pinocchio::computeJointJacobians(*pinocchio_, d->pinocchio, q);
  // pinocchio::updateFramePlacements(*pinocchio_, d->pinocchio); //TODO why? we run it in calc()
  costs_->calcDiff(d->costs, x, u, false);

  // if u is None:
  //     u = self.unone
  // if recalc:
  //     xout, cost = self.calc(data, x, u)
  // nq, nv = self.nq, self.nv
  // q = a2m(x[:nq])
  // v = a2m(x[-nv:])
  // tauq = a2m(u)
  // a = a2m(data.xout)
  // # --- Dynamics
  // if self.forceAba:
  //     pinocchio.computeABADerivatives(self.pinocchio, data.pinocchio, q, v, tauq)
  //     data.Fx[:, :nv] = data.pinocchio.ddq_dq
  //     data.Fx[:, nv:] = data.pinocchio.ddq_dv
  //     data.Fu[:, :] = data.Minv
  // else:
  //     pinocchio.computeRNEADerivatives(self.pinocchio, data.pinocchio, q, v, a)
  //     data.Fx[:, :nv] = -np.dot(data.Minv, data.pinocchio.dtau_dq)
  //     data.Fx[:, nv:] = -np.dot(data.Minv, data.pinocchio.dtau_dv)
  //     data.Fu[:, :] = data.Minv
  // # --- Cost
  // pinocchio.computeJointJacobians(self.pinocchio, data.pinocchio, q)
  // pinocchio.updateFramePlacements(self.pinocchio, data.pinocchio)
  // self.costs.calcDiff(data.costs, x, u, recalc=False)
  // return data.xout, data.cost
}

boost::shared_ptr<DifferentialActionDataAbstract> DifferentialActionModelFreeFwdDynamics::createData() {
  return boost::make_shared<DifferentialActionDataFreeFwdDynamics>(this);
}

CostModelSum* DifferentialActionModelFreeFwdDynamics::get_costs() const { return costs_; }

pinocchio::Model* DifferentialActionModelFreeFwdDynamics::get_pinocchio() const { return pinocchio_; }

}  // namespace crocoddyl