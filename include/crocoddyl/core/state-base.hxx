///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////
#include <crocoddyl/core/mathbase.hpp>

namespace crocoddyl {
template <typename Scalar>
StateAbstractTpl<Scalar>::StateAbstractTpl(const std::size_t& nx, const std::size_t& ndx)
    : nx_(nx),
      ndx_(ndx),
      lb_(MathBase::VectorXs::Constant(nx_, -std::numeric_limits<Scalar>::infinity())),
      ub_(MathBase::VectorXs::Constant(nx_, std::numeric_limits<Scalar>::infinity())),
      has_limits_(false) {
  nv_ = ndx / 2;
  nq_ = nx_ - nv_;
}

template <typename Scalar>
StateAbstractTpl<Scalar>::~StateAbstractTpl() {}

template <typename Scalar>
const std::size_t& StateAbstractTpl<Scalar>::get_nx() const { return nx_; }
template <typename Scalar>
const std::size_t& StateAbstractTpl<Scalar>::get_ndx() const { return ndx_; }
template <typename Scalar>
const std::size_t& StateAbstractTpl<Scalar>::get_nq() const { return nq_; }
template <typename Scalar>
const std::size_t& StateAbstractTpl<Scalar>::get_nv() const { return nv_; }
template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& StateAbstractTpl<Scalar>::get_lb() const { return lb_; }
template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& StateAbstractTpl<Scalar>::get_ub() const { return ub_; }
template <typename Scalar>
bool const& StateAbstractTpl<Scalar>::get_has_limits() const { return has_limits_; }
template <typename Scalar>
void StateAbstractTpl<Scalar>::set_lb(const typename MathBase::VectorXs& lb) {
  if (static_cast<std::size_t>(lb.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "lower bound has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }
  lb_ = lb;
  update_has_limits();
}
template <typename Scalar>
void StateAbstractTpl<Scalar>::set_ub(const typename MathBase::VectorXs& ub) {
  if (static_cast<std::size_t>(ub.size()) != nx_) {
    throw_pretty("Invalid argument: "
                 << "upper bound has wrong dimension (it should be " + std::to_string(nx_) + ")");
  }
  ub_ = ub;
  update_has_limits();
}
template <typename Scalar>
void StateAbstractTpl<Scalar>::update_has_limits() { has_limits_ = isfinite(lb_.array()).any() && isfinite(ub_.array()).any(); }

}  // namespace crocoddyl
