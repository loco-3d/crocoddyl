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
// #include <pinocchio/algorithm/aba.hpp>
// #include <pinocchio/algorithm/aba-derivatives.hpp>
// #include <pinocchio/algorithm/kinematics.hpp>
// #include <pinocchio/algorithm/jacobian.hpp>
// #include <pinocchio/algorithm/cholesky.hpp>

namespace crocoddyl {

DifferentialActionModelContactFwdDynamics::DifferentialActionModelContactFwdDynamics(
    StateMultibody& state, ActuationModelFloatingBase& actuation, ContactModelMultiple& contacts, CostModelSum& costs)
    : DifferentialActionModelAbstract(state, actuation.get_nu(), costs.get_nr()),
      actuation_(actuation),
      contacts_(contacts),
      costs_(costs),
      pinocchio_(state.get_pinocchio()),
      force_aba_(true),
      armature_(Eigen::VectorXd::Zero(state.get_nv())),
      JMinvJt_damping_(0.) {}

DifferentialActionModelContactFwdDynamics::~DifferentialActionModelContactFwdDynamics() {}

void DifferentialActionModelContactFwdDynamics::calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                                     const Eigen::Ref<const Eigen::VectorXd>& x,
                                                     const Eigen::Ref<const Eigen::VectorXd>& u) {
  assert(x.size() == state_.get_nx() && "DifferentialActionModelContactFwdDynamics::calc: x has wrong dimension");
  assert(u.size() == nu_ && "DifferentialActionModelContactFwdDynamics::calc: u has wrong dimension");

  DifferentialActionDataContactFwdDynamics* d = static_cast<DifferentialActionDataContactFwdDynamics*>(data.get());
  d->qcur = x.head(state_.get_nq());
  d->vcur = x.tail(state_.get_nv());

  // Computing the forward dynamics with the holonomic constraints defined by the contact model
  pinocchio::computeAllTerms(pinocchio_, d->pinocchio, d->qcur, d->vcur);
  pinocchio::updateFramePlacements(pinocchio_, d->pinocchio);

  if (!force_aba_) {
    d->pinocchio.M.diagonal() += armature_;
  }
  actuation_.calc(d->actuation, x, u);
  contacts_.calc(d->contacts, x);
  pinocchio::forwardDynamics(pinocchio_, d->pinocchio, d->qcur, d->vcur, d->actuation->a, d->contacts->Jc,
                             d->contacts->a0, JMinvJt_damping_, false);
  d->xout = d->pinocchio.ddq;
  contacts_.updateLagrangian(d->contacts, d->pinocchio.lambda_c);

  // Computing the cost value and residuals
  costs_.calc(d->costs, x, u);
  d->cost = d->costs->cost;
}

void DifferentialActionModelContactFwdDynamics::calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                                         const Eigen::Ref<const Eigen::VectorXd>& x,
                                                         const Eigen::Ref<const Eigen::VectorXd>& u,
                                                         const bool& recalc) {
  assert(x.size() == state_.get_nx() && "DifferentialActionModelContactFwdDynamics::calcDiff: x has wrong dimension");
  assert(u.size() == nu_ && "DifferentialActionModelContactFwdDynamics::calcDiff: u has wrong dimension");

  DifferentialActionDataContactFwdDynamics* d = static_cast<DifferentialActionDataContactFwdDynamics*>(data.get());
  const unsigned int& nv = state_.get_nv();
  if (recalc) {
    calc(data, x, u);
  } else {
    d->qcur = x.head(state_.get_nq());
    d->vcur = x.tail(nv);
  }

  // Computing the dynamics derivatives
  pinocchio::computeRNEADerivatives(pinocchio_, d->pinocchio, d->qcur, d->vcur, d->xout, d->contacts->fext);
  pinocchio::computeForwardKinematicsDerivatives(pinocchio_, d->pinocchio, d->qcur, d->vcur, d->xout);
  pinocchio::updateFramePlacements(pinocchio_, d->pinocchio);

  actuation_.calcDiff(d->actuation, x, u, false);
  contacts_.calcDiff(d->contacts, x, false);

  // Kinv = [ [Minv - Minv*Jt*Jtinv, Jinv], [Jtinv, -JtinvMJinv] ]
  // Kinv = [ [a_partial_dtau, a_partial_da], [f_partial_dtau, f_partial_da] ]
  pinocchio::cholesky::decompose(pinocchio_, d->pinocchio);
  d->Minv.setZero();
  pinocchio::cholesky::computeMinv(pinocchio_, d->pinocchio, d->Minv);
  // TODO(cmastalli): use cholesky
  d->f_partial_da = -d->pinocchio.JMinvJt.inverse();
  d->JtinvM.noalias() = -d->f_partial_da * d->contacts->Jc;
  d->f_partial_dtau.noalias() = d->JtinvM * d->Minv;
  d->a_partial_da = d->f_partial_dtau.transpose();
  d->JtJtinv.noalias() = d->contacts->Jc.transpose() * d->f_partial_dtau;
  d->a_partial_dtau.noalias() = d->Minv - d->Minv * d->JtJtinv;

  d->Fx.leftCols(nv).noalias() = -d->a_partial_dtau * d->pinocchio.dtau_dq;
  d->Fx.rightCols(nv).noalias() = -d->a_partial_dtau * d->pinocchio.dtau_dv;
  d->Fx.noalias() -= d->a_partial_da * d->contacts->Ax;
  d->Fx.noalias() += d->a_partial_dtau * d->actuation->Ax;
  d->Fu.noalias() = d->a_partial_dtau * d->actuation->Au;

  // Computing the cost derivatives
  // TODO(cmastalli): Set the derivatives of the Lagrangian, i.e. compute f_partial_dx and f_partial_du
  // contacts_.updateLagrangian(d->contacts, -d->pinocchio.lambda_c);
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

void DifferentialActionModelContactFwdDynamics::set_armature(const Eigen::VectorXd& armature) {
  if (armature.size() != state_.get_nv()) {
    std::cout << "The armature dimension is wrong, we cannot set it." << std::endl;
  } else {
    armature_ = armature;
    force_aba_ = false;
  }
}

}  // namespace crocoddyl
