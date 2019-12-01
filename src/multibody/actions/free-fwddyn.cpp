///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/actions/free-fwddyn.hpp"
#include <pinocchio/algorithm/aba.hpp>
#include <pinocchio/algorithm/aba-derivatives.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/cholesky.hpp>

namespace crocoddyl {

DifferentialActionModelFreeFwdDynamics::DifferentialActionModelFreeFwdDynamics(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActuationModelAbstract> actuation,
    boost::shared_ptr<CostModelSum> costs)
    : DifferentialActionModelAbstract(state, actuation->get_nu(), costs->get_nr()),
      actuation_(actuation),
      costs_(costs),
      pinocchio_(state->get_pinocchio()),
      with_armature_(true),
      armature_(Eigen::VectorXd::Zero(state->get_nv())) {
  if (costs_->get_nu() != nu_) {
    throw std::invalid_argument("Costs doesn't have the same control dimension (it should be " + std::to_string(nu_) +
                                ")");
  }
}

DifferentialActionModelFreeFwdDynamics::~DifferentialActionModelFreeFwdDynamics() {}

void DifferentialActionModelFreeFwdDynamics::calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                                  const Eigen::Ref<const Eigen::VectorXd>& x,
                                                  const Eigen::Ref<const Eigen::VectorXd>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw std::invalid_argument("x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw std::invalid_argument("u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }

  DifferentialActionDataFreeFwdDynamics* d = static_cast<DifferentialActionDataFreeFwdDynamics*>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const Eigen::VectorXd>, Eigen::Dynamic> q = x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const Eigen::VectorXd>, Eigen::Dynamic> v = x.tail(state_->get_nv());

  actuation_->calc(d->actuation, x, u);

  // Computing the dynamics using ABA or manually for armature case
  if (with_armature_) {
    d->xout = pinocchio::aba(pinocchio_, d->pinocchio, q, v, d->actuation->tau);
  } else {
    pinocchio::computeAllTerms(pinocchio_, d->pinocchio, q, v);
    d->pinocchio.M.diagonal() += armature_;
    pinocchio::cholesky::decompose(pinocchio_, d->pinocchio);
    d->Minv.setZero();
    pinocchio::cholesky::computeMinv(pinocchio_, d->pinocchio, d->Minv);
    d->u_drift = d->actuation->tau - d->pinocchio.nle;
    d->xout.noalias() = d->Minv * d->u_drift;
  }

  // Computing the cost value and residuals
  pinocchio::forwardKinematics(pinocchio_, d->pinocchio, q, v);
  pinocchio::updateFramePlacements(pinocchio_, d->pinocchio);
  costs_->calc(d->costs, x, u);
  d->cost = d->costs->cost;
}

void DifferentialActionModelFreeFwdDynamics::calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                                      const Eigen::Ref<const Eigen::VectorXd>& x,
                                                      const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw std::invalid_argument("x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw std::invalid_argument("u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }

  const std::size_t& nv = state_->get_nv();
  const Eigen::VectorBlock<const Eigen::Ref<const Eigen::VectorXd>, Eigen::Dynamic> q = x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const Eigen::VectorXd>, Eigen::Dynamic> v = x.tail(nv);

  DifferentialActionDataFreeFwdDynamics* d = static_cast<DifferentialActionDataFreeFwdDynamics*>(data.get());
  if (recalc) {
    calc(data, x, u);
    pinocchio::computeJointJacobians(pinocchio_, d->pinocchio, q);
  }

  actuation_->calcDiff(d->actuation, x, u, false);

  // Computing the dynamics derivatives
  if (with_armature_) {
    pinocchio::computeABADerivatives(pinocchio_, d->pinocchio, q, v, d->actuation->tau);
    d->Fx.leftCols(nv) = d->pinocchio.ddq_dq;
    d->Fx.rightCols(nv) = d->pinocchio.ddq_dv;
    d->Fx += d->pinocchio.Minv * d->actuation->dtau_dx;
    d->Fu.noalias() = d->pinocchio.Minv * d->actuation->dtau_du;
  } else {
    pinocchio::computeRNEADerivatives(pinocchio_, d->pinocchio, q, v, d->xout);
    d->dtau_dx.leftCols(nv) = d->actuation->dtau_dx.leftCols(nv) - d->pinocchio.dtau_dq;
    d->dtau_dx.rightCols(nv) = d->actuation->dtau_dx.rightCols(nv) - d->pinocchio.dtau_dv;
    d->Fx.noalias() = d->Minv * d->dtau_dx;
    d->Fu.noalias() = d->Minv * d->actuation->dtau_du;
  }

  // Computing the cost derivatives
  costs_->calcDiff(d->costs, x, u, false);
}

boost::shared_ptr<DifferentialActionDataAbstract> DifferentialActionModelFreeFwdDynamics::createData() {
  return boost::make_shared<DifferentialActionDataFreeFwdDynamics>(this);
}

pinocchio::Model& DifferentialActionModelFreeFwdDynamics::get_pinocchio() const { return pinocchio_; }

const boost::shared_ptr<ActuationModelAbstract>& DifferentialActionModelFreeFwdDynamics::get_actuation() const {
  return actuation_;
}

const boost::shared_ptr<CostModelSum>& DifferentialActionModelFreeFwdDynamics::get_costs() const { return costs_; }

const Eigen::VectorXd& DifferentialActionModelFreeFwdDynamics::get_armature() const { return armature_; }

void DifferentialActionModelFreeFwdDynamics::set_armature(const Eigen::VectorXd& armature) {
  if (static_cast<std::size_t>(armature.size()) != state_->get_nv()) {
    throw std::invalid_argument("The armature dimension is wrong (it should be " + std::to_string(state_->get_nv()) +
                                ")");
  }

  armature_ = armature;
  with_armature_ = false;
}

}  // namespace crocoddyl
