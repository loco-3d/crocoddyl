///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/actions/contact-fwddyn.hpp"

#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>

namespace crocoddyl {

template <typename Scalar>
DifferentialActionModelContactFwdDynamicsTpl<Scalar>::DifferentialActionModelContactFwdDynamicsTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActuationModelFloatingBase> actuation,
    boost::shared_ptr<ContactModelMultiple> contacts, boost::shared_ptr<CostModelSum> costs,
    const Scalar& JMinvJt_damping, const bool& enable_force)
    : Base(state, actuation->get_nu(), costs->get_nr()),
      actuation_(actuation),
      contacts_(contacts),
      costs_(costs),
      pinocchio_(*state->get_pinocchio().get()),
      with_armature_(true),
      armature_(VectorXs::Zero(state->get_nv())),
      JMinvJt_damping_(fabs(JMinvJt_damping)),
      enable_force_(enable_force) {
  if (JMinvJt_damping_ < Scalar(0.)) {
    JMinvJt_damping_ = Scalar(0.);
    throw_pretty("Invalid argument: "
                 << "The damping factor has to be positive, set to 0");
  }
  if (contacts_->get_nu() != nu_) {
    throw_pretty("Invalid argument: "
                 << "Contacts doesn't have the same control dimension (it should be " + std::to_string(nu_) + ")");
  }
  if (costs_->get_nu() != nu_) {
    throw_pretty("Invalid argument: "
                 << "Costs doesn't have the same control dimension (it should be " + std::to_string(nu_) + ")");
  }

  Base::set_u_lb(Scalar(-1.) * pinocchio_.effortLimit.tail(nu_));
  Base::set_u_ub(Scalar(+1.) * pinocchio_.effortLimit.tail(nu_));
}

template <typename Scalar>
DifferentialActionModelContactFwdDynamicsTpl<Scalar>::~DifferentialActionModelContactFwdDynamicsTpl() {}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamicsTpl<Scalar>::calc(
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

  const std::size_t& nc = contacts_->get_nc();
  Data* d = static_cast<Data*>(data.get());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.tail(state_->get_nv());

  // Computing the forward dynamics with the holonomic constraints defined by the contact model
  pinocchio::computeAllTerms(pinocchio_, d->pinocchio, q, v);
  pinocchio::computeCentroidalMomentum(pinocchio_, d->pinocchio);

  if (!with_armature_) {
    d->pinocchio.M.diagonal() += armature_;
  }
  actuation_->calc(d->multibody.actuation, x, u);
  contacts_->calc(d->multibody.contacts, x);

#ifndef NDEBUG
  Eigen::FullPivLU<MatrixXs> Jc_lu(d->multibody.contacts->Jc.topRows(nc));

  if (Jc_lu.rank() < d->multibody.contacts->Jc.topRows(nc).rows() && JMinvJt_damping_ == Scalar(0.)) {
    throw_pretty("A damping factor is needed as the contact Jacobian is not full-rank");
  }
#endif

  pinocchio::forwardDynamics(pinocchio_, d->pinocchio, d->multibody.actuation->tau,
                             d->multibody.contacts->Jc.topRows(nc), d->multibody.contacts->a0.head(nc),
                             JMinvJt_damping_);
  d->xout = d->pinocchio.ddq;
  contacts_->updateAcceleration(d->multibody.contacts, d->pinocchio.ddq);
  contacts_->updateForce(d->multibody.contacts, d->pinocchio.lambda_c);

  // Computing the cost value and residuals
  costs_->calc(d->costs, x, u);
  d->cost = d->costs->cost;
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamicsTpl<Scalar>::calcDiff(
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
  const std::size_t& nc = contacts_->get_nc();
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q = x.head(state_->get_nq());
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> v = x.tail(nv);

  Data* d = static_cast<Data*>(data.get());

  // Computing the dynamics derivatives
  pinocchio::computeRNEADerivatives(pinocchio_, d->pinocchio, q, v, d->xout, d->multibody.contacts->fext);
  pinocchio::getKKTContactDynamicMatrixInverse(pinocchio_, d->pinocchio, d->multibody.contacts->Jc.topRows(nc),
                                               d->Kinv);

  actuation_->calcDiff(d->multibody.actuation, x, u);
  contacts_->calcDiff(d->multibody.contacts, x);

  Eigen::Block<MatrixXs> a_partial_dtau = d->Kinv.topLeftCorner(nv, nv);
  Eigen::Block<MatrixXs> a_partial_da = d->Kinv.topRightCorner(nv, nc);
  Eigen::Block<MatrixXs> f_partial_dtau = d->Kinv.bottomLeftCorner(nc, nv);
  Eigen::Block<MatrixXs> f_partial_da = d->Kinv.bottomRightCorner(nc, nc);

  d->Fx.leftCols(nv).noalias() = -a_partial_dtau * d->pinocchio.dtau_dq;
  d->Fx.rightCols(nv).noalias() = -a_partial_dtau * d->pinocchio.dtau_dv;
  d->Fx.noalias() -= a_partial_da * d->multibody.contacts->da0_dx.topRows(nc);
  d->Fx.noalias() += a_partial_dtau * d->multibody.actuation->dtau_dx;
  d->Fu.noalias() = a_partial_dtau * d->multibody.actuation->dtau_du;

  // Computing the cost derivatives
  if (enable_force_) {
    d->df_dx.leftCols(nv).noalias() = f_partial_dtau * d->pinocchio.dtau_dq;
    d->df_dx.rightCols(nv).noalias() = f_partial_dtau * d->pinocchio.dtau_dv;
    d->df_dx.noalias() += f_partial_da * d->multibody.contacts->da0_dx.topRows(nc);
    d->df_dx.noalias() -= f_partial_dtau * d->multibody.actuation->dtau_dx;
    d->df_du.noalias() = -f_partial_dtau * d->multibody.actuation->dtau_du;
    contacts_->updateAccelerationDiff(d->multibody.contacts, d->Fx.bottomRows(nv));
    contacts_->updateForceDiff(d->multibody.contacts, d->df_dx, d->df_du);
  }
  costs_->calcDiff(d->costs, x, u);
}

template <typename Scalar>
boost::shared_ptr<DifferentialActionDataAbstractTpl<Scalar> >
DifferentialActionModelContactFwdDynamicsTpl<Scalar>::createData() {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
}

template <typename Scalar>
bool DifferentialActionModelContactFwdDynamicsTpl<Scalar>::checkData(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data) {
  boost::shared_ptr<Data> d = boost::dynamic_pointer_cast<Data>(data);
  if (d != NULL) {
    return true;
  } else {
    return false;
  }
}

template <typename Scalar>
pinocchio::ModelTpl<Scalar>& DifferentialActionModelContactFwdDynamicsTpl<Scalar>::get_pinocchio() const {
  return pinocchio_;
}

template <typename Scalar>
const boost::shared_ptr<ActuationModelFloatingBaseTpl<Scalar> >&
DifferentialActionModelContactFwdDynamicsTpl<Scalar>::get_actuation() const {
  return actuation_;
}

template <typename Scalar>
const boost::shared_ptr<ContactModelMultipleTpl<Scalar> >&
DifferentialActionModelContactFwdDynamicsTpl<Scalar>::get_contacts() const {
  return contacts_;
}

template <typename Scalar>
const boost::shared_ptr<CostModelSumTpl<Scalar> >& DifferentialActionModelContactFwdDynamicsTpl<Scalar>::get_costs()
    const {
  return costs_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& DifferentialActionModelContactFwdDynamicsTpl<Scalar>::get_armature()
    const {
  return armature_;
}

template <typename Scalar>
const Scalar& DifferentialActionModelContactFwdDynamicsTpl<Scalar>::get_damping_factor() const {
  return JMinvJt_damping_;
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamicsTpl<Scalar>::set_armature(const VectorXs& armature) {
  if (static_cast<std::size_t>(armature.size()) != state_->get_nv()) {
    throw_pretty("Invalid argument: "
                 << "The armature dimension is wrong (it should be " + std::to_string(state_->get_nv()) + ")");
  }
  armature_ = armature;
  with_armature_ = false;
}

template <typename Scalar>
void DifferentialActionModelContactFwdDynamicsTpl<Scalar>::set_damping_factor(const Scalar& damping) {
  if (damping < 0.) {
    throw_pretty("Invalid argument: "
                 << "The damping factor has to be positive");
  }
  JMinvJt_damping_ = damping;
}

}  // namespace crocoddyl
