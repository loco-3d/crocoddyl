///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#include "crocoddyl/core/mathbase.hpp"

namespace crocoddyl {
template <typename Scalar>
StateAbstractTpl<Scalar>::StateAbstractTpl(const std::size_t& nx, const std::size_t& ndx)
    : nx_(nx),
      ndx_(ndx),
      lb_(VectorXs::Constant(nx_, -std::numeric_limits<Scalar>::infinity())),
      ub_(VectorXs::Constant(nx_, std::numeric_limits<Scalar>::infinity())),
      has_limits_(false) {
  nv_ = ndx / 2;
  nq_ = nx_ - nv_;
}

template <typename Scalar>
StateAbstractTpl<Scalar>::StateAbstractTpl()
    : nx_(0),
      ndx_(0),
      lb_(MathBase::VectorXs::Constant(nx_, -std::numeric_limits<Scalar>::infinity())),
      ub_(MathBase::VectorXs::Constant(nx_, std::numeric_limits<Scalar>::infinity())),
      has_limits_(false) {}

template <typename Scalar>
StateAbstractTpl<Scalar>::~StateAbstractTpl() {}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs StateAbstractTpl<Scalar>::diff_dx(const Eigen::Ref<const VectorXs>& x0,
                                                                         const Eigen::Ref<const VectorXs>& x1) {
  VectorXs dxout = VectorXs::Zero(ndx_);
  diff(x0, x1, dxout);
  return dxout;
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::VectorXs StateAbstractTpl<Scalar>::integrate_x(const Eigen::Ref<const VectorXs>& x,
                                                                             const Eigen::Ref<const VectorXs>& dx) {
  VectorXs xout = VectorXs::Zero(nx_);
  integrate(x, dx, xout);
  return xout;
}

template <typename Scalar>
std::vector<typename MathBaseTpl<Scalar>::MatrixXs> StateAbstractTpl<Scalar>::Jdiff_Js(
    const Eigen::Ref<const VectorXs>& x0, const Eigen::Ref<const VectorXs>& x1, Jcomponent firstsecond) {
  MatrixXs Jfirst(ndx_, ndx_), Jsecond(ndx_, ndx_);
  Jfirst.setZero();
  Jsecond.setZero();
  std::vector<MatrixXs> Jacs;
  Jdiff(x0, x1, Jfirst, Jsecond, firstsecond);
  switch (firstsecond) {
    case both:
      Jacs.push_back(Jfirst);
      Jacs.push_back(Jsecond);
      break;
    case first:
      Jacs.push_back(Jfirst);
      break;
    case second:
      Jacs.push_back(Jsecond);
      break;
    default:
      Jacs.push_back(Jfirst);
      Jacs.push_back(Jsecond);
      break;
  }
  return Jacs;
}

template <typename Scalar>
std::vector<typename MathBaseTpl<Scalar>::MatrixXs> StateAbstractTpl<Scalar>::Jintegrate_Js(
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& dx, Jcomponent firstsecond) {
  MatrixXs Jfirst(ndx_, ndx_), Jsecond(ndx_, ndx_);
  std::vector<MatrixXs> Jacs;
  Jintegrate(x, dx, Jfirst, Jsecond, firstsecond);
  switch (firstsecond) {
    case both:
      Jacs.push_back(Jfirst);
      Jacs.push_back(Jsecond);
      break;
    case first:
      Jacs.push_back(Jfirst);
      break;
    case second:
      Jacs.push_back(Jsecond);
      break;
    default:
      Jacs.push_back(Jfirst);
      Jacs.push_back(Jsecond);
      break;
  }
  return Jacs;
}

template <typename Scalar>
const std::size_t& StateAbstractTpl<Scalar>::get_nx() const {
  return nx_;
}

template <typename Scalar>
const std::size_t& StateAbstractTpl<Scalar>::get_ndx() const {
  return ndx_;
}

template <typename Scalar>
const std::size_t& StateAbstractTpl<Scalar>::get_nq() const {
  return nq_;
}

template <typename Scalar>
const std::size_t& StateAbstractTpl<Scalar>::get_nv() const {
  return nv_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& StateAbstractTpl<Scalar>::get_lb() const {
  return lb_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& StateAbstractTpl<Scalar>::get_ub() const {
  return ub_;
}

template <typename Scalar>
bool const& StateAbstractTpl<Scalar>::get_has_limits() const {
  return has_limits_;
}

template <typename Scalar>
void StateAbstractTpl<Scalar>::set_lb(const VectorXs& lb) {
  if (static_cast<std::size_t>(lb.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "lower bound has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }
  lb_ = lb;
  update_has_limits();
}

template <typename Scalar>
void StateAbstractTpl<Scalar>::set_ub(const VectorXs& ub) {
  if (static_cast<std::size_t>(ub.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "upper bound has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }
  ub_ = ub;
  update_has_limits();
}

template <typename Scalar>
void StateAbstractTpl<Scalar>::update_has_limits() {
  has_limits_ = isfinite(lb_.array()).any() || isfinite(ub_.array()).any();
}

}  // namespace crocoddyl
