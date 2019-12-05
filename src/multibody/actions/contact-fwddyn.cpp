///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/actions/contact-fwddyn.hpp"
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>

namespace crocoddyl {

DifferentialActionModelContactFwdDynamics::DifferentialActionModelContactFwdDynamics(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActuationModelFloatingBase> actuation,
    boost::shared_ptr<ContactModelMultiple> contacts, boost::shared_ptr<CostModelSum> costs,
    const double& JMinvJt_damping, const bool& enable_force)
    : DifferentialActionModelAbstract(state, actuation->get_nu(), costs->get_nr()),
      actuation_(actuation),
      contacts_(contacts),
      costs_(costs),
      pinocchio_(state->get_pinocchio()),
      with_armature_(true),
      armature_(Eigen::VectorXd::Zero(state->get_nv())),
      JMinvJt_damping_(fabs(JMinvJt_damping)),
      enable_force_(enable_force) {
  if (JMinvJt_damping_ < 0.) {
    JMinvJt_damping_ = 0.;
    throw std::invalid_argument("The damping factor has to be positive, set to 0");
  }
  if (contacts_->get_nu() != nu_) {
    throw std::invalid_argument("Contacts doesn't have the same control dimension (it should be " +
                                std::to_string(nu_) + ")");
  }
  if (costs_->get_nu() != nu_) {
    throw std::invalid_argument("Costs doesn't have the same control dimension (it should be " + std::to_string(nu_) +
                                ")");
  }

  set_u_lb(-1. * pinocchio_.effortLimit.tail(nu_));
  set_u_ub(+1. * pinocchio_.effortLimit.tail(nu_));
}

DifferentialActionModelContactFwdDynamics::~DifferentialActionModelContactFwdDynamics() {}

void DifferentialActionModelContactFwdDynamics::calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                                     const Eigen::Ref<const Eigen::VectorXd>& x,
                                                     const Eigen::Ref<const Eigen::VectorXd>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw std::invalid_argument("x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw std::invalid_argument("u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }

  DifferentialActionDataContactFwdDynamics* d = static_cast<DifferentialActionDataContactFwdDynamics*>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const Eigen::VectorXd>, Eigen::Dynamic> q = x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const Eigen::VectorXd>, Eigen::Dynamic> v = x.tail(state_->get_nv());

  // Computing the forward dynamics with the holonomic constraints defined by the contact model
  pinocchio::computeAllTerms(pinocchio_, d->pinocchio, q, v);
  pinocchio::updateFramePlacements(pinocchio_, d->pinocchio);
  pinocchio::computeCentroidalDynamics(pinocchio_, d->pinocchio, q, v);

  if (!with_armature_) {
    d->pinocchio.M.diagonal() += armature_;
  }
  actuation_->calc(d->actuation, x, u);
  contacts_->calc(d->multibody.contacts, x);

#ifndef NDEBUG
  Eigen::FullPivLU<Eigen::MatrixXd> Jc_lu(d->multibody.contacts->Jc);

  if (Jc_lu.rank() < d->multibody.contacts->Jc.rows()) {
    assert(JMinvJt_damping_ > 0. && "A damping factor is needed as the contact Jacobian is not full-rank");
  }
#endif

  pinocchio::forwardDynamics(pinocchio_, d->pinocchio, d->actuation->tau, d->multibody.contacts->Jc,
                             d->multibody.contacts->a0, JMinvJt_damping_);
  d->xout = d->pinocchio.ddq;
  contacts_->updateAcceleration(d->multibody.contacts, d->pinocchio.ddq);
  contacts_->updateForce(d->multibody.contacts, d->pinocchio.lambda_c);

  // Computing the cost value and residuals
  costs_->calc(d->costs, x, u);
  d->cost = d->costs->cost;
}

void DifferentialActionModelContactFwdDynamics::calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                                         const Eigen::Ref<const Eigen::VectorXd>& x,
                                                         const Eigen::Ref<const Eigen::VectorXd>& u,
                                                         const bool& recalc) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw std::invalid_argument("x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw std::invalid_argument("u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }

  const std::size_t& nv = state_->get_nv();
  const std::size_t& nc = contacts_->get_nc();
  const Eigen::VectorBlock<const Eigen::Ref<const Eigen::VectorXd>, Eigen::Dynamic> q = x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const Eigen::VectorXd>, Eigen::Dynamic> v = x.tail(nv);

  DifferentialActionDataContactFwdDynamics* d = static_cast<DifferentialActionDataContactFwdDynamics*>(data.get());
  if (recalc) {
    calc(data, x, u);
  }

  // Computing the dynamics derivatives
  pinocchio::computeRNEADerivatives(pinocchio_, d->pinocchio, q, v, d->xout, d->multibody.contacts->fext);
  pinocchio::getKKTContactDynamicMatrixInverse(pinocchio_, d->pinocchio, d->multibody.contacts->Jc, d->Kinv);

  actuation_->calcDiff(d->actuation, x, u, false);
  contacts_->calcDiff(d->multibody.contacts, x, false);

  Eigen::Block<Eigen::MatrixXd> a_partial_dtau = d->Kinv.topLeftCorner(nv, nv);
  Eigen::Block<Eigen::MatrixXd> a_partial_da = d->Kinv.topRightCorner(nv, nc);
  Eigen::Block<Eigen::MatrixXd> f_partial_dtau = d->Kinv.bottomLeftCorner(nc, nv);
  Eigen::Block<Eigen::MatrixXd> f_partial_da = d->Kinv.bottomRightCorner(nc, nc);

  d->Fx.leftCols(nv).noalias() = -a_partial_dtau * d->pinocchio.dtau_dq;
  d->Fx.rightCols(nv).noalias() = -a_partial_dtau * d->pinocchio.dtau_dv;
  d->Fx.noalias() -= a_partial_da * d->multibody.contacts->da0_dx;
  d->Fx.noalias() += a_partial_dtau * d->actuation->dtau_dx;
  d->Fu.noalias() = a_partial_dtau * d->actuation->dtau_du;

  // Computing the cost derivatives
  if (enable_force_) {
    d->df_dx.leftCols(nv).noalias() = f_partial_dtau * d->pinocchio.dtau_dq;
    d->df_dx.rightCols(nv).noalias() = f_partial_dtau * d->pinocchio.dtau_dv;
    d->df_dx.noalias() += f_partial_da * d->multibody.contacts->da0_dx;
    d->df_dx.noalias() -= f_partial_dtau * d->actuation->dtau_dx;
    d->df_du.noalias() = -f_partial_dtau * d->actuation->dtau_du;
    contacts_->updateAccelerationDiff(d->multibody.contacts, d->Fx.bottomRows(nv));
    contacts_->updateForceDiff(d->multibody.contacts, d->df_dx, d->df_du);
  }
  costs_->calcDiff(d->costs, x, u, false);
}

boost::shared_ptr<DifferentialActionDataAbstract> DifferentialActionModelContactFwdDynamics::createData() {
  return boost::make_shared<DifferentialActionDataContactFwdDynamics>(this);
}

pinocchio::Model& DifferentialActionModelContactFwdDynamics::get_pinocchio() const { return pinocchio_; }

const boost::shared_ptr<ActuationModelFloatingBase>& DifferentialActionModelContactFwdDynamics::get_actuation() const {
  return actuation_;
}

const boost::shared_ptr<ContactModelMultiple>& DifferentialActionModelContactFwdDynamics::get_contacts() const {
  return contacts_;
}

const boost::shared_ptr<CostModelSum>& DifferentialActionModelContactFwdDynamics::get_costs() const { return costs_; }

const Eigen::VectorXd& DifferentialActionModelContactFwdDynamics::get_armature() const { return armature_; }

const double& DifferentialActionModelContactFwdDynamics::get_damping_factor() const { return JMinvJt_damping_; }

void DifferentialActionModelContactFwdDynamics::set_armature(const Eigen::VectorXd& armature) {
  if (static_cast<std::size_t>(armature.size()) != state_->get_nv()) {
    throw std::invalid_argument("The armature dimension is wrong (it should be " + std::to_string(state_->get_nv()) +
                                ")");
  }
  armature_ = armature;
  with_armature_ = false;
}

void DifferentialActionModelContactFwdDynamics::set_damping_factor(const double& damping) {
  if (damping < 0.) {
    throw std::invalid_argument("The damping factor has to be positive");
  }
  JMinvJt_damping_ = damping;
}

}  // namespace crocoddyl
