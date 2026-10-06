///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/math.hpp"

namespace crocoddyl {

template <typename Scalar>
DynamicsModelAbstractTpl<Scalar>::DynamicsModelAbstractTpl(
    std::shared_ptr<StateAbstractTpl<Scalar>> state,
    const DynamicsType dyn_type, const std::size_t np, const std::size_t nu,
    const std::size_t ng, const std::size_t nh)
    : nu_(nu),
      np_(np),
      ng_(ng),
      nh_(nh),
      state_(state),
      unone_(VectorXs::Zero(nu)),
      tau_meas_(VectorXs::Zero(nu)),
      p_lb_(VectorXs::Constant(np, -std::numeric_limits<Scalar>::infinity())),
      p_ub_(VectorXs::Constant(np, std::numeric_limits<Scalar>::infinity())),
      dyn_type_(dyn_type) {}

template <typename Scalar>
void DynamicsModelAbstractTpl<Scalar>::calc(
    const std::shared_ptr<DynamicsDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  calc(data, x, unone_);
}

template <typename Scalar>
void DynamicsModelAbstractTpl<Scalar>::calcDiff(
    const std::shared_ptr<DynamicsDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }
  calcDiff_xu(data, x, u);
  if (np_ != 0) {
    calcDiff_p(data, x, u);
  }
}

template <typename Scalar>
void DynamicsModelAbstractTpl<Scalar>::calcDiff(
    const std::shared_ptr<DynamicsDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  calcDiff_xu(data, x);
}

template <typename Scalar>
void DynamicsModelAbstractTpl<Scalar>::calcDiff_xu(
    const std::shared_ptr<DynamicsDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  calcDiff_xu(data, x, unone_);
}

template <typename Scalar>
void DynamicsModelAbstractTpl<Scalar>::calcDiff_p(
    const std::shared_ptr<DynamicsDataAbstract>&,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  throw_pretty(
      "Invalid call: "
      "calcDiff_p must be implemented in derived classes when np != 0");
}

template <typename Scalar>
std::shared_ptr<DynamicsDataAbstractTpl<Scalar>>
DynamicsModelAbstractTpl<Scalar>::createData() {
  return std::allocate_shared<DynamicsDataAbstract>(
      Eigen::aligned_allocator<DynamicsDataAbstract>(), this);
}

template <typename Scalar>
std::shared_ptr<DynamicsDataAbstractTpl<Scalar>>
DynamicsModelAbstractTpl<Scalar>::createData(
    const std::shared_ptr<ParameterDataManager>&) {
  return createData();
}

template <typename Scalar>
void DynamicsModelAbstractTpl<Scalar>::set_params(
    const std::shared_ptr<DynamicsDataAbstract>&,
    std::shared_ptr<ParameterManager>) {
  throw_pretty(
      "Invalid call: set_params is not supported for this dynamics "
      "model");
}

template <typename Scalar>
void DynamicsModelAbstractTpl<Scalar>::update_p(
    const std::shared_ptr<DynamicsDataAbstract>&,
    const Eigen::Ref<const VectorXs>&) {
  throw_pretty(
      "Invalid call: update_p is not supported for this dynamics "
      "model");
}

template <typename Scalar>
bool DynamicsModelAbstractTpl<Scalar>::checkData(
    const std::shared_ptr<DynamicsDataAbstract>&) {
  return false;
}

template <typename Scalar>
void DynamicsModelAbstractTpl<Scalar>::quasiStatic(
    const std::shared_ptr<DynamicsDataAbstract>& data, Eigen::Ref<VectorXs> u,
    const Eigen::Ref<const VectorXs>& x, const std::size_t maxiter,
    const Scalar tol) {
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "u has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }
  if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
    throw_pretty(
        "Invalid argument: " << "x has wrong dimension (it should be " +
                                    std::to_string(state_->get_nx()) + ")");
  }
  assert_pretty(x.tail(state_->get_nv()).isZero(),
                "The velocity input should be zero for quasi-static to work.");

  if (nu_ != 0) {
    VectorXs du = VectorXs::Zero(nu_);
    for (std::size_t i = 0; i < maxiter; ++i) {
      calc(data, x, u);
      calcDiff_xu(data, x, u);
      du.noalias() = -pseudoInverse(data->Fu) * data->vdot;
      u += du;
      if (du.norm() <= tol) {
        break;
      }
    }
  }
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs
DynamicsModelAbstractTpl<Scalar>::quasiStatic_x(
    const std::shared_ptr<DynamicsDataAbstract>& data, const VectorXs& x,
    const std::size_t maxiter, const Scalar tol) {
  VectorXs u(nu_);
  u.setZero();
  quasiStatic(data, u, x, maxiter, tol);
  return u;
}

template <typename Scalar>
void DynamicsModelAbstractTpl<Scalar>::update_tau(
    const Eigen::Ref<const VectorXs>& tau_meas) {
  if (static_cast<std::size_t>(tau_meas.size()) != nu_) {
    throw_pretty(
        "Invalid argument: " << "tau_meas has wrong dimension (it should be " +
                                    std::to_string(nu_) + ")");
  }
  tau_meas_ = tau_meas;
}

template <typename Scalar>
void DynamicsModelAbstractTpl<Scalar>::set_dyn_type(
    const DynamicsType dyn_type) {
  dyn_type_ = dyn_type;
}

template <typename Scalar>
std::size_t DynamicsModelAbstractTpl<Scalar>::get_nu() const {
  return nu_;
}

template <typename Scalar>
std::size_t DynamicsModelAbstractTpl<Scalar>::get_np() const {
  return np_;
}

template <typename Scalar>
std::size_t DynamicsModelAbstractTpl<Scalar>::get_ng() const {
  return ng_;
}

template <typename Scalar>
std::size_t DynamicsModelAbstractTpl<Scalar>::get_nh() const {
  return nh_;
}

template <typename Scalar>
const std::shared_ptr<StateAbstractTpl<Scalar>>&
DynamicsModelAbstractTpl<Scalar>::get_state() const {
  return state_;
}

template <typename Scalar>
DynamicsType DynamicsModelAbstractTpl<Scalar>::get_dyn_type() const {
  return dyn_type_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DynamicsModelAbstractTpl<Scalar>::get_tau_meas() const {
  return tau_meas_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DynamicsModelAbstractTpl<Scalar>::get_p_lb() const {
  return p_lb_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
DynamicsModelAbstractTpl<Scalar>::get_p_ub() const {
  return p_ub_;
}

template <typename Scalar>
void DynamicsModelAbstractTpl<Scalar>::set_p_lb(const VectorXs& p_lb) {
  if (static_cast<std::size_t>(p_lb.size()) != np_) {
    throw_pretty(
        "Invalid argument: " << "p_lb has wrong dimension (it should be " +
                                    std::to_string(np_) + ")");
  }
  p_lb_ = p_lb;
}

template <typename Scalar>
void DynamicsModelAbstractTpl<Scalar>::set_p_ub(const VectorXs& p_ub) {
  if (static_cast<std::size_t>(p_ub.size()) != np_) {
    throw_pretty(
        "Invalid argument: " << "p_ub has wrong dimension (it should be " +
                                    std::to_string(np_) + ")");
  }
  p_ub_ = p_ub;
}

template <typename Scalar>
void DynamicsModelAbstractTpl<Scalar>::print(std::ostream& os) const {
  os << boost::core::demangle(typeid(*this).name());
}

template <typename Scalar>
std::ostream& operator<<(std::ostream& os,
                         const DynamicsModelAbstractTpl<Scalar>& model) {
  model.print(os);
  return os;
}

}  // namespace crocoddyl
