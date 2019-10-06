///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/actions/contact-fwddyn.hpp"
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>

namespace crocoddyl {

DifferentialActionModelContactFwdDynamics::DifferentialActionModelContactFwdDynamics(
    StateMultibody& state, ActuationModelFloatingBase& actuation, ContactModelMultiple& contacts, CostModelSum& costs,
    const double& JMinvJt_damping, const bool& enable_force)
    : DifferentialActionModelAbstract(state, actuation.get_nu(), costs.get_nr()),
      actuation_(actuation),
      contacts_(contacts),
      costs_(costs),
      pinocchio_(state.get_pinocchio()),
      with_armature_(true),
      armature_(Eigen::VectorXd::Zero(state.get_nv())),
      JMinvJt_damping_(fabs(JMinvJt_damping)),
      enable_force_(enable_force) {
  assert(contacts_.get_nu() == nu_ && "Contacts doesn't have the same control dimension");
  assert(costs_.get_nu() == nu_ && "Costs doesn't have the same control dimension");

  set_u_lb(-1. * pinocchio_.effortLimit.tail(nu_));
  set_u_ub(+1. * pinocchio_.effortLimit.tail(nu_));
}

DifferentialActionModelContactFwdDynamics::~DifferentialActionModelContactFwdDynamics() {}

void DifferentialActionModelContactFwdDynamics::calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                                     const Eigen::Ref<const Eigen::VectorXd>& x,
                                                     const Eigen::Ref<const Eigen::VectorXd>& u) {
  assert(x.size() == state_.get_nx() && "x has wrong dimension");
  assert(u.size() == nu_ && "u has wrong dimension");

  DifferentialActionDataContactFwdDynamics* d = static_cast<DifferentialActionDataContactFwdDynamics*>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const Eigen::VectorXd>, Eigen::Dynamic> q = x.head(state_.get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const Eigen::VectorXd>, Eigen::Dynamic> v = x.tail(state_.get_nv());

  // Computing the forward dynamics with the holonomic constraints defined by the contact model
  pinocchio::computeAllTerms(pinocchio_, d->pinocchio, q, v);
  pinocchio::updateFramePlacements(pinocchio_, d->pinocchio);

  if (!with_armature_) {
    d->pinocchio.M.diagonal() += armature_;
  }
  actuation_.calc(d->actuation, x, u);
  contacts_.calc(d->contacts, x);

#ifndef NDEBUG
  Eigen::FullPivLU<Eigen::MatrixXd> Jc_lu(d->contacts->Jc);

  if (Jc_lu.rank() < d->contacts->Jc.rows()) {
    assert(JMinvJt_damping_ > 0. && "It is needed a damping factor since the contact Jacobian is not full-rank");
  }
#endif

  pinocchio::forwardDynamics(pinocchio_, d->pinocchio, q, v, d->actuation->tau, d->contacts->Jc, d->contacts->a0,
                             JMinvJt_damping_, false);
  d->xout = d->pinocchio.ddq;
  contacts_.updateAcceleration(d->contacts, d->pinocchio.ddq);
  contacts_.updateForce(d->contacts, d->pinocchio.lambda_c);

  // Computing the cost value and residuals
  costs_.calc(d->costs, x, u);
  d->cost = d->costs->cost;
}

void DifferentialActionModelContactFwdDynamics::calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                                         const Eigen::Ref<const Eigen::VectorXd>& x,
                                                         const Eigen::Ref<const Eigen::VectorXd>& u,
                                                         const bool& recalc) {
  assert(x.size() == state_.get_nx() && "x has wrong dimension");
  assert(u.size() == nu_ && "u has wrong dimension");

  unsigned int const& nv = state_.get_nv();
  unsigned int const& nc = contacts_.get_nc();
  const Eigen::VectorBlock<const Eigen::Ref<const Eigen::VectorXd>, Eigen::Dynamic> q = x.head(state_.get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const Eigen::VectorXd>, Eigen::Dynamic> v = x.tail(nv);

  DifferentialActionDataContactFwdDynamics* d = static_cast<DifferentialActionDataContactFwdDynamics*>(data.get());
  if (recalc) {
    calc(data, x, u);
  }

  // Computing the dynamics derivatives
  pinocchio::computeRNEADerivatives(pinocchio_, d->pinocchio, q, v, d->xout, d->contacts->fext);
  pinocchio::getKKTContactDynamicMatrixInverse(pinocchio_, d->pinocchio, d->contacts->Jc, d->Kinv);

  actuation_.calcDiff(d->actuation, x, u, false);
  contacts_.calcDiff(d->contacts, x, false);

  Eigen::Block<Eigen::MatrixXd> a_partial_dtau = d->Kinv.topLeftCorner(nv, nv);
  Eigen::Block<Eigen::MatrixXd> a_partial_da = d->Kinv.topRightCorner(nv, nc);
  Eigen::Block<Eigen::MatrixXd> f_partial_dtau = d->Kinv.bottomLeftCorner(nc, nv);
  Eigen::Block<Eigen::MatrixXd> f_partial_da = d->Kinv.bottomRightCorner(nc, nc);

  d->Fx.leftCols(nv).noalias() = -a_partial_dtau * d->pinocchio.dtau_dq;
  d->Fx.rightCols(nv).noalias() = -a_partial_dtau * d->pinocchio.dtau_dv;
  d->Fx.noalias() -= a_partial_da * d->contacts->da0_dx;
  d->Fx.noalias() += a_partial_dtau * d->actuation->dtau_dx;
  d->Fu.noalias() = a_partial_dtau * d->actuation->dtau_du;

  // Computing the cost derivatives
  if (enable_force_) {
    d->df_dx.leftCols(nv).noalias() = f_partial_dtau * d->pinocchio.dtau_dq;
    d->df_dx.rightCols(nv).noalias() = f_partial_dtau * d->pinocchio.dtau_dv;
    d->df_dx.noalias() += f_partial_da * d->contacts->da0_dx;
    d->df_dx.noalias() -= f_partial_dtau * d->actuation->dtau_dx;
    d->df_du.noalias() = -f_partial_dtau * d->actuation->dtau_du;
    contacts_.updateAccelerationDiff(d->contacts, d->Fx.bottomRows(nv));
    contacts_.updateForceDiff(d->contacts, d->df_dx, d->df_du);
  }
  costs_.calcDiff(d->costs, x, u, false);
}

boost::shared_ptr<DifferentialActionDataAbstract> DifferentialActionModelContactFwdDynamics::createData() {
  return boost::make_shared<DifferentialActionDataContactFwdDynamics>(this);
}

pinocchio::Model& DifferentialActionModelContactFwdDynamics::get_pinocchio() const { return pinocchio_; }

ActuationModelFloatingBase& DifferentialActionModelContactFwdDynamics::get_actuation() const { return actuation_; }

ContactModelMultiple& DifferentialActionModelContactFwdDynamics::get_contacts() const { return contacts_; }

CostModelSum& DifferentialActionModelContactFwdDynamics::get_costs() const { return costs_; }

const Eigen::VectorXd& DifferentialActionModelContactFwdDynamics::get_armature() const { return armature_; }

const double& DifferentialActionModelContactFwdDynamics::get_damping_factor() const { return JMinvJt_damping_; }

void DifferentialActionModelContactFwdDynamics::set_armature(const Eigen::VectorXd& armature) {
  assert(armature.size() == state_.get_nv() && "The armature dimension is wrong, we cannot set it.");
  if (armature.size() != state_.get_nv()) {
    std::cout << "The armature dimension is wrong, we cannot set it." << std::endl;
  } else {
    armature_ = armature;
    with_armature_ = false;
  }
}

void DifferentialActionModelContactFwdDynamics::set_damping_factor(const double& damping) {
  JMinvJt_damping_ = damping;
}

}  // namespace crocoddyl
