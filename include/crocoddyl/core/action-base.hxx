///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <iostream>                 // std::ostream
#include <typeinfo>                 // typeid()
#include <boost/core/demangle.hpp>  // boost::core::demangle

#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename Scalar>
ActionModelAbstractTpl<Scalar>::ActionModelAbstractTpl(boost::shared_ptr<StateAbstractTpl<Scalar> > state,
                                                       const std::size_t nu, const std::size_t nr)
    : nu_(nu),
      nr_(nr),
      state_(state),
      unone_(MathBase::VectorXs::Zero(nu)),
      u_lb_(MathBase::VectorXs::Constant(nu, -std::numeric_limits<Scalar>::infinity())),
      u_ub_(MathBase::VectorXs::Constant(nu, std::numeric_limits<Scalar>::infinity())),
      has_control_limits_(false) {}

template <typename Scalar>
ActionModelAbstractTpl<Scalar>::~ActionModelAbstractTpl() {}

template <typename Scalar>
void ActionModelAbstractTpl<Scalar>::calc(const boost::shared_ptr<ActionDataAbstract>& data,
                                          const Eigen::Ref<const VectorXs>& x) {
  calc(data, x, unone_);
}

template <typename Scalar>
void ActionModelAbstractTpl<Scalar>::calcDiff(const boost::shared_ptr<ActionDataAbstract>& data,
                                              const Eigen::Ref<const VectorXs>& x) {
  calcDiff(data, x, unone_);
}

template <typename Scalar>
void ActionModelAbstractTpl<Scalar>::quasiStatic(const boost::shared_ptr<ActionDataAbstract>& data,
                                                 Eigen::Ref<VectorXs> u, const Eigen::Ref<const VectorXs>& x,
                                                 const std::size_t maxiter, const Scalar tol) {
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }
  // Check the velocity input is zero
  assert_pretty(x.tail(state_->get_nv()).isZero(), "The velocity input should be zero for quasi-static to work.");

  const std::size_t ndx = state_->get_ndx();
  VectorXs dx = VectorXs::Zero(ndx);
  if (nu_ == 0) {
    // TODO(cmastalli): create a method for autonomous systems
  } else {
    VectorXs du = VectorXs::Zero(nu_);
    for (std::size_t i = 0; i < maxiter; ++i) {
      calc(data, x, u);
      calcDiff(data, x, u);
      state_->diff(x, data->xnext, dx);
      du.noalias() = -pseudoInverse(data->Fu) * dx;
      u += du;
      if (du.norm() <= tol) {
        break;
      }
    }
  }
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs ActionModelAbstractTpl<Scalar>::quasiStatic_x(
    const boost::shared_ptr<ActionDataAbstract>& data, const VectorXs& x, const std::size_t maxiter,
    const Scalar tol) {
  VectorXs u(nu_);
  u.setZero();
  quasiStatic(data, u, x, maxiter, tol);
  return u;
}

template <typename Scalar>
boost::shared_ptr<ActionDataAbstractTpl<Scalar> > ActionModelAbstractTpl<Scalar>::createData() {
  return boost::allocate_shared<ActionDataAbstract>(Eigen::aligned_allocator<ActionDataAbstract>(), this);
}

template <typename Scalar>
bool ActionModelAbstractTpl<Scalar>::checkData(const boost::shared_ptr<ActionDataAbstract>&) {
  return false;
}

template <typename Scalar>
std::size_t ActionModelAbstractTpl<Scalar>::get_nu() const {
  return nu_;
}

template <typename Scalar>
std::size_t ActionModelAbstractTpl<Scalar>::get_nr() const {
  return nr_;
}

template <typename Scalar>
const boost::shared_ptr<StateAbstractTpl<Scalar> >& ActionModelAbstractTpl<Scalar>::get_state() const {
  return state_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& ActionModelAbstractTpl<Scalar>::get_u_lb() const {
  return u_lb_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& ActionModelAbstractTpl<Scalar>::get_u_ub() const {
  return u_ub_;
}

template <typename Scalar>
bool ActionModelAbstractTpl<Scalar>::get_has_control_limits() const {
  return has_control_limits_;
}

template <typename Scalar>
void ActionModelAbstractTpl<Scalar>::set_u_lb(const VectorXs& u_lb) {
  if (static_cast<std::size_t>(u_lb.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "lower bound has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  u_lb_ = u_lb;
  update_has_control_limits();
}

template <typename Scalar>
void ActionModelAbstractTpl<Scalar>::set_u_ub(const VectorXs& u_ub) {
  if (static_cast<std::size_t>(u_ub.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "upper bound has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  u_ub_ = u_ub;
  update_has_control_limits();
}

template <typename Scalar>
void ActionModelAbstractTpl<Scalar>::update_has_control_limits() {
  has_control_limits_ = isfinite(u_lb_.array()).any() && isfinite(u_ub_.array()).any();
}

template <typename Scalar>
std::ostream& operator<<(std::ostream& os, const ActionModelAbstractTpl<Scalar>& action_model) {
  os << "ActionModel type '" << boost::core::demangle(typeid(action_model).name()) << "'";
  return os;
}

}  // namespace crocoddyl
