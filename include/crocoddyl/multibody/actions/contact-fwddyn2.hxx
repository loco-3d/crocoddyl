///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, LAAS-CNRS, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/actions/contact-fwddyn2.hpp"

#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>

namespace crocoddyl {

template <typename Scalar>
DifferentialActionModelContactFwdDynamics2Tpl<Scalar>::DifferentialActionModelContactFwdDynamics2Tpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActuationModelAbstract> actuation,
    const PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidContactModel) & contacts,
    boost::shared_ptr<CostModelSum> costs, const Scalar mu_contacts)
    : Base(state, actuation->get_nu(), costs->get_nr()),
      actuation_(actuation),
      contacts_(contacts),
      costs_(costs),
      pinocchio_(*state->get_pinocchio().get()),
      mu_contacts(mu_contacts) {
  if (costs_->get_nu() != nu_) {
    throw_pretty("Invalid argument: "
                 << "Costs doesn't have the same control dimension (it should be " + std::to_string(nu_) + ")");
  }

  Base::set_u_lb(Scalar(-1.) * pinocchio_.effortLimit.tail(nu_));
  Base::set_u_ub(Scalar(+1.) * pinocchio_.effortLimit.tail(nu_));
}

template <typename Scalar>
DifferentialActionModelContactFwdDynamics2Tpl<Scalar>::~DifferentialActionModelContactFwdDynamics2Tpl() {}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamics2Tpl<Scalar>::calc(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
    const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }

  DifferentialActionDataContactFwdDynamics2Tpl<Scalar>* d =
      static_cast<DifferentialActionDataContactFwdDynamics2Tpl<Scalar>*>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.tail(state_->get_nv());

  // Computing the forward dynamics with the holonomic constraints defined by the contact model

  actuation_->calc(d->multibody.actuation, x, u);
  d->xout = pinocchio::contactDynamics(pinocchio_, d->pinocchio, q, v, d->multibody.actuation->tau, contacts_,
                                       d->multibody.contacts, mu_contacts);
  pinocchio::jacobianCenterOfMass(pinocchio_, d->pinocchio, false);
  // Computing the cost value and residuals
  costs_->calc(d->costs, x, u);
  d->cost = d->costs->cost;
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamics2Tpl<Scalar>::calcDiff(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
    const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }

  const std::size_t& nv = state_->get_nv();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.tail(nv);

  DifferentialActionDataContactFwdDynamics2Tpl<Scalar>* d =
      static_cast<DifferentialActionDataContactFwdDynamics2Tpl<Scalar>*>(data.get());

  // Computing the dynamics derivatives
  actuation_->calcDiff(d->multibody.actuation, x, u);
  pinocchio::computeContactDynamicsDerivatives(
      pinocchio_, d->pinocchio, contacts_, d->multibody.contacts, Scalar(0.), d->Fx.leftCols(nv), d->Fx.rightCols(nv),
      d->pinocchio.ddq_dtau, d->pinocchio.dlambda_dq, d->pinocchio.dlambda_dv, d->pinocchio.dlambda_dtau);

  d->Fu.noalias() = d->pinocchio.ddq_dtau * d->multibody.actuation->dtau_du;
  costs_->calcDiff(d->costs, x, u);
}

template <typename Scalar>
boost::shared_ptr<DifferentialActionDataAbstractTpl<Scalar> >
DifferentialActionModelContactFwdDynamics2Tpl<Scalar>::createData() {
  boost::shared_ptr<DifferentialActionDataAbstractTpl<Scalar> > data =
      boost::make_shared<DifferentialActionDataContactFwdDynamics2>(this);
  DifferentialActionDataContactFwdDynamics2* d = static_cast<DifferentialActionDataContactFwdDynamics2*>(data.get());
  pinocchio::initContactDynamics(pinocchio_, d->pinocchio, contacts_);

  return data;
}
/*
template <typename Scalar>
void DifferentialActionModelContactFwdDynamics2Tpl<Scalar>::quasiStatic(const
boost::shared_ptr<DifferentialActionDataAbstract>& data, Eigen::Ref<VectorXs> u, const Eigen::Ref<const VectorXs>& x,
                                                                     const std::size_t& maxiter, const Scalar& tol) {
if (static_cast<std::size_t>(u.size()) != nu_) {
  throw_pretty("Invalid argument: "
               << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
}
if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
  throw_pretty("Invalid argument: "
               << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
}
// Static casting the data
DifferentialActionDataContactFwdDynamics2Tpl<Scalar>* d =
    static_cast<DifferentialActionDataContactFwdDynamics2Tpl<Scalar>*>(data.get());
const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.tail(state_->get_nv());

d->multibody.actuation->tau = pinocchio::rnea(pinocchio_, d->pinocchio, q, v, VectorXs::Zero(state_->get_nv()));
actuation_->get_actuated(d->multibody.actuation, u);
}
*/

template <typename Scalar>
pinocchio::ModelTpl<Scalar>& DifferentialActionModelContactFwdDynamics2Tpl<Scalar>::get_pinocchio() const {
  return pinocchio_;
}

template <typename Scalar>
const boost::shared_ptr<ActuationModelAbstractTpl<Scalar> >&
DifferentialActionModelContactFwdDynamics2Tpl<Scalar>::get_actuation() const {
  return actuation_;
}

template <typename Scalar>
const std::vector<pinocchio::RigidContactModelTpl<Scalar, 0>,
                  Eigen::aligned_allocator<pinocchio::RigidContactModelTpl<Scalar, 0> > >&
DifferentialActionModelContactFwdDynamics2Tpl<Scalar>::get_contacts() const {
  return contacts_;
}

template <typename Scalar>
const boost::shared_ptr<CostModelSumTpl<Scalar> >& DifferentialActionModelContactFwdDynamics2Tpl<Scalar>::get_costs()
    const {
  return costs_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& DifferentialActionModelContactFwdDynamics2Tpl<Scalar>::get_armature()
    const {
  return pinocchio_.armature;
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamics2Tpl<Scalar>::set_armature(const VectorXs& armature) {
  if (static_cast<std::size_t>(armature.size()) != state_->get_nv()) {
    throw_pretty("Invalid argument: "
                 << "The armature dimension is wrong (it should be " + std::to_string(state_->get_nv()) + ")");
  }
  pinocchio_.armature = armature;
}

template <typename Scalar>
const Scalar& DifferentialActionModelContactFwdDynamics2Tpl<Scalar>::get_mu() const {
  return mu_contacts;
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamics2Tpl<Scalar>::set_mu(const Scalar mu_contacts_) {
  mu_contacts = mu_contacts_;
}

}  // namespace crocoddyl
