///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/actions/impulse-fwddyn.hpp"
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>

namespace crocoddyl {

ActionModelImpulseFwdDynamics::ActionModelImpulseFwdDynamics(boost::shared_ptr<StateMultibody> state,
                                                             boost::shared_ptr<ImpulseModelMultiple> impulses,
                                                             boost::shared_ptr<CostModelSum> costs,
                                                             const double& r_coeff, const double& JMinvJt_damping,
                                                             const bool& enable_force)
    : ActionModelAbstract(state, 0, costs->get_nr()),
      impulses_(impulses),
      costs_(costs),
      pinocchio_(state->get_pinocchio()),
      with_armature_(true),
      armature_(Eigen::VectorXd::Zero(state->get_nv())),
      r_coeff_(r_coeff),
      JMinvJt_damping_(JMinvJt_damping),
      enable_force_(enable_force),
      gravity_(state->get_pinocchio().gravity) {}

ActionModelImpulseFwdDynamics::~ActionModelImpulseFwdDynamics() {}

void ActionModelImpulseFwdDynamics::calc(const boost::shared_ptr<ActionDataAbstract>& data,
                                         const Eigen::Ref<const Eigen::VectorXd>& x,
                                         const Eigen::Ref<const Eigen::VectorXd>& u) {
  assert(static_cast<std::size_t>(x.size()) == state_->get_nx() && "x has wrong dimension");

  const std::size_t& nq = state_->get_nq();
  const std::size_t& nv = state_->get_nv();
  ActionDataImpulseFwdDynamics* d = static_cast<ActionDataImpulseFwdDynamics*>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const Eigen::VectorXd>, Eigen::Dynamic> q = x.head(nq);
  const Eigen::VectorBlock<const Eigen::Ref<const Eigen::VectorXd>, Eigen::Dynamic> v = x.tail(nv);

  // Computing the forward dynamics with the holonomic constraints defined by the contact model
  pinocchio::computeAllTerms(pinocchio_, d->pinocchio, q, v);
  pinocchio::updateFramePlacements(pinocchio_, d->pinocchio);

  if (!with_armature_) {
    d->pinocchio.M.diagonal() += armature_;
  }
  impulses_->calc(d->impulses, x);

#ifndef NDEBUG
  Eigen::FullPivLU<Eigen::MatrixXd> Jc_lu(d->impulses->Jc);

  if (Jc_lu.rank() < d->impulses->Jc.rows()) {
    assert(JMinvJt_damping_ > 0. && "It is needed a damping factor since the contact Jacobian is not full-rank");
  }
#endif

  pinocchio::impulseDynamics(pinocchio_, d->pinocchio, v, d->impulses->Jc, r_coeff_, JMinvJt_damping_);
  d->xnext.head(nq) = q;
  d->xnext.tail(nv) = d->pinocchio.dq_after;
  impulses_->updateVelocity(d->impulses, d->pinocchio.dq_after);
  impulses_->updateForce(d->impulses, d->pinocchio.impulse_c);

  // Computing the cost value and residuals
  costs_->calc(d->costs, x, u);
  d->cost = d->costs->cost;
}

void ActionModelImpulseFwdDynamics::calcDiff(const boost::shared_ptr<ActionDataAbstract>& data,
                                             const Eigen::Ref<const Eigen::VectorXd>& x,
                                             const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc) {
  assert(static_cast<std::size_t>(x.size()) == state_->get_nx() && "x has wrong dimension");

  const std::size_t& nv = state_->get_nv();
  const std::size_t& ni = impulses_->get_ni();
  const Eigen::VectorBlock<const Eigen::Ref<const Eigen::VectorXd>, Eigen::Dynamic> q = x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const Eigen::VectorXd>, Eigen::Dynamic> v = x.tail(nv);

  ActionDataImpulseFwdDynamics* d = static_cast<ActionDataImpulseFwdDynamics*>(data.get());
  if (recalc) {
    calc(data, x, u);
  }

  // Computing the dynamics derivatives
  pinocchio_.gravity.setZero();
  pinocchio::computeRNEADerivatives(pinocchio_, d->pinocchio, q, d->vnone, d->pinocchio.dq_after - v,
                                    d->impulses->fext);
  pinocchio_.gravity = gravity_;
  pinocchio::getKKTContactDynamicMatrixInverse(pinocchio_, d->pinocchio, d->impulses->Jc, d->Kinv);

  pinocchio::computeForwardKinematicsDerivatives(pinocchio_, d->pinocchio, q, d->pinocchio.dq_after, d->vnone);
  impulses_->calcDiff(d->impulses, x, false);

  Eigen::Block<Eigen::MatrixXd> a_partial_dtau = d->Kinv.topLeftCorner(nv, nv);
  Eigen::Block<Eigen::MatrixXd> a_partial_da = d->Kinv.topRightCorner(nv, ni);
  Eigen::Block<Eigen::MatrixXd> f_partial_dtau = d->Kinv.bottomLeftCorner(ni, nv);
  Eigen::Block<Eigen::MatrixXd> f_partial_da = d->Kinv.bottomRightCorner(ni, ni);

  d->Fx.topLeftCorner(nv, nv).setIdentity();
  d->Fx.topRightCorner(nv, nv).setZero();
  d->Fx.bottomLeftCorner(nv, nv).noalias() = -a_partial_dtau * d->pinocchio.dtau_dq;
  d->Fx.bottomLeftCorner(nv, nv).noalias() -= a_partial_da * d->impulses->dv0_dq;
  d->Fx.bottomRightCorner(nv, nv).noalias() = a_partial_dtau * d->pinocchio.M.selfadjointView<Eigen::Upper>();

  // Computing the cost derivatives
  if (enable_force_) {
    d->df_dq.noalias() = f_partial_dtau * d->pinocchio.dtau_dq;
    d->df_dq.noalias() += f_partial_da * d->impulses->dv0_dq;
    impulses_->updateVelocityDiff(d->impulses, d->Fx.bottomRows(nv));
    impulses_->updateForceDiff(d->impulses, d->df_dq);
  }
  costs_->calcDiff(d->costs, x, u, false);
}

boost::shared_ptr<ActionDataAbstract> ActionModelImpulseFwdDynamics::createData() {
  return boost::make_shared<ActionDataImpulseFwdDynamics>(this);
}

pinocchio::Model& ActionModelImpulseFwdDynamics::get_pinocchio() const { return pinocchio_; }

const boost::shared_ptr<ImpulseModelMultiple>& ActionModelImpulseFwdDynamics::get_impulses() const {
  return impulses_;
}

const boost::shared_ptr<CostModelSum>& ActionModelImpulseFwdDynamics::get_costs() const { return costs_; }

const Eigen::VectorXd& ActionModelImpulseFwdDynamics::get_armature() const { return armature_; }

const double& ActionModelImpulseFwdDynamics::get_restitution_coefficient() const { return r_coeff_; }

const double& ActionModelImpulseFwdDynamics::get_damping_factor() const { return JMinvJt_damping_; }

void ActionModelImpulseFwdDynamics::set_armature(const Eigen::VectorXd& armature) {
  assert(static_cast<std::size_t>(armature.size()) == state_->get_nv() &&
         "The armature dimension is wrong, we cannot set it.");
  if (static_cast<std::size_t>(armature.size()) != state_->get_nv()) {
    std::cout << "The armature dimension is wrong, we cannot set it." << std::endl;
  } else {
    armature_ = armature;
    with_armature_ = false;
  }
}

void ActionModelImpulseFwdDynamics::set_restitution_coefficient(const double& r_coeff) { r_coeff_ = r_coeff; }

void ActionModelImpulseFwdDynamics::set_damping_factor(const double& damping) { JMinvJt_damping_ = damping; }

}  // namespace crocoddyl
