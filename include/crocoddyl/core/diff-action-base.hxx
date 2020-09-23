///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, The University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename Scalar>
DifferentialActionModelAbstractTpl<Scalar>::DifferentialActionModelAbstractTpl(boost::shared_ptr<StateAbstract> state,
                                                                               const std::size_t& nu,
                                                                               const std::size_t& nr,
                                                                               const std::size_t& ng)
    : nu_(nu),
      nr_(nr),
      ng_(ng),
      state_(state),
      unone_(VectorXs::Zero(nu)),
      u_lb_(VectorXs::Constant(nu, -std::numeric_limits<Scalar>::infinity())),
      u_ub_(VectorXs::Constant(nu, std::numeric_limits<Scalar>::infinity())),
      has_control_limits_(false) {}

template <typename Scalar>
DifferentialActionModelAbstractTpl<Scalar>::~DifferentialActionModelAbstractTpl() {}

template <typename Scalar>
void DifferentialActionModelAbstractTpl<Scalar>::calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                                                      const Eigen::Ref<const VectorXs>& x) {
  calc(data, x, unone_);
}

template <typename Scalar>
void DifferentialActionModelAbstractTpl<Scalar>::quasiStatic(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data, Eigen::Ref<VectorXs> u,
    const Eigen::Ref<const VectorXs>& x, const std::size_t& maxiter, const Scalar& tol) {
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty("Invalid argument: "
                 << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
  }

  const std::size_t& ndx = state_->get_ndx();
  VectorXs dx = VectorXs::Zero(ndx);
  if (nu_ == 0) {
    // TODO(cmastalli): create a method for autonomous systems
  } else {
    VectorXs du = VectorXs::Zero(nu_);
    for (std::size_t i = 0; i < maxiter; ++i) {
      calc(data, x, u);
      calcDiff(data, x, u);
      du.noalias() = -pseudoInverse(data->Fu) * data->xout;
      u += du;
      if (du.norm() <= tol) {
        break;
      }
    }
  }
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs DifferentialActionModelAbstractTpl<Scalar>::quasiStatic_x(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data, const VectorXs& x, const std::size_t& maxiter,
    const Scalar& tol) {
  VectorXs u(nu_);
  u.setZero();
  quasiStatic(data, u, x, maxiter, tol);
  return u;
}

template <typename Scalar>
void DifferentialActionModelAbstractTpl<Scalar>::calcDiff(
    const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x) {
  calcDiff(data, x, unone_);
}

template <typename Scalar>
boost::shared_ptr<DifferentialActionDataAbstractTpl<Scalar> >
DifferentialActionModelAbstractTpl<Scalar>::createData() {
  return boost::allocate_shared<DifferentialActionDataAbstract>(
      Eigen::aligned_allocator<DifferentialActionDataAbstract>(), this);
}

template <typename Scalar>
bool DifferentialActionModelAbstractTpl<Scalar>::checkData(const boost::shared_ptr<DifferentialActionDataAbstract>&) {
  return false;
}

template <typename Scalar>
const std::size_t& DifferentialActionModelAbstractTpl<Scalar>::get_nu() const {
  return nu_;
}

template <typename Scalar>
const std::size_t& DifferentialActionModelAbstractTpl<Scalar>::get_nr() const {
  return nr_;
}

template <typename Scalar>
const std::size_t& DifferentialActionModelAbstractTpl<Scalar>::get_ng() const {
  return ng_;
}

template <typename Scalar>
const boost::shared_ptr<StateAbstractTpl<Scalar> >& DifferentialActionModelAbstractTpl<Scalar>::get_state() const {
  return state_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& DifferentialActionModelAbstractTpl<Scalar>::get_u_lb() const {
  return u_lb_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& DifferentialActionModelAbstractTpl<Scalar>::get_u_ub() const {
  return u_ub_;
}

template <typename Scalar>
bool const& DifferentialActionModelAbstractTpl<Scalar>::get_has_control_limits() const {
  return has_control_limits_;
}

template <typename Scalar>
void DifferentialActionModelAbstractTpl<Scalar>::set_u_lb(const VectorXs& u_lb) {
  if (static_cast<std::size_t>(u_lb.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "lower bound has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  u_lb_ = u_lb;
  update_has_control_limits();
}

template <typename Scalar>
void DifferentialActionModelAbstractTpl<Scalar>::set_u_ub(const VectorXs& u_ub) {
  if (static_cast<std::size_t>(u_ub.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "upper bound has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  u_ub_ = u_ub;
  update_has_control_limits();
}

template <typename Scalar>
void DifferentialActionModelAbstractTpl<Scalar>::update_has_control_limits() {
  has_control_limits_ = isfinite(u_lb_.array()).any() && isfinite(u_ub_.array()).any();
}

}  // namespace crocoddyl
