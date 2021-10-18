///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <typeinfo>
#include <boost/core/demangle.hpp>

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
void ActionModelAbstractTpl<Scalar>::multiplyByFx(const Eigen::Ref<const MatrixXs>& Fx,
                                                  const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXs> out,
                                                  const AssignmentOp op) const {
  assert_pretty(is_a_AssignmentOp(op), ("op must be one of the AssignmentOp {settop, addto, rmfrom}"));
  if (static_cast<std::size_t>(A.cols()) != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "number of columns of A is wrong, it should be " + std::to_string(state_->get_ndx()) +
                        " instead of " + std::to_string(A.cols()));
  }
  if (A.rows() != out.rows()) {
    throw_pretty("Invalid argument: "
                 << "A and out have different number of rows: " + std::to_string(A.rows()) + " and " +
                        std::to_string(out.rows()));
  }
  if (static_cast<std::size_t>(out.cols()) != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "number of columns of out is wrong, it should be " + std::to_string(state_->get_ndx()) +
                        " instead of " + std::to_string(out.cols()));
  }

  switch (op) {
    case setto:
      out.noalias() = A * Fx;
      break;
    case addto:
      out.noalias() += A * Fx;
      break;
    case rmfrom:
      out.noalias() -= A * Fx;
      break;
    default:
      throw_pretty("Invalid argument: allowed operators: setto, addto, rmfrom");
  }
}

template <typename Scalar>
void ActionModelAbstractTpl<Scalar>::multiplyFxTransposeBy(const Eigen::Ref<const MatrixXs>& FxTranspose,
                                                           const Eigen::Ref<const MatrixXs>& A,
                                                           Eigen::Ref<MatrixXsRowMajor> out,
                                                           const AssignmentOp op) const {
  assert_pretty(is_a_AssignmentOp(op), ("op must be one of the AssignmentOp {settop, addto, rmfrom}"));
  if (static_cast<std::size_t>(A.rows()) != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "number of rows of A is wrong, it should be " + std::to_string(state_->get_ndx()) +
                        " instead of " + std::to_string(A.rows()));
  }
  if (A.cols() != out.cols()) {
    throw_pretty("Invalid argument: "
                 << "A and out have different number of columns: " + std::to_string(A.cols()) + " and " +
                        std::to_string(out.cols()));
  }
  if (static_cast<std::size_t>(out.rows()) != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "number of rows of out is wrong, it should be " + std::to_string(state_->get_ndx()) +
                        " instead of " + std::to_string(out.cols()));
  }
  switch (op) {
    case setto:
      out.noalias() = FxTranspose * A;
      break;
    case addto:
      out.noalias() += FxTranspose * A;
      break;
    case rmfrom:
      out.noalias() -= FxTranspose * A;
      break;
    default:
      throw_pretty("Invalid argument: allowed operators: setto, addto, rmfrom");
  }
}

template <typename Scalar>
void ActionModelAbstractTpl<Scalar>::multiplyByFu(const Eigen::Ref<const MatrixXs>& Fu,
                                                  const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXs> out,
                                                  const AssignmentOp op) const {
  assert_pretty(is_a_AssignmentOp(op), ("op must be one of the AssignmentOp {settop, addto, rmfrom}"));
  if (static_cast<std::size_t>(A.cols()) != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "number of columns of A is wrong, it should be " + std::to_string(state_->get_ndx()) +
                        " instead of " + std::to_string(A.cols()));
  }
  if (A.rows() != out.rows()) {
    throw_pretty("Invalid argument: "
                 << "A and out have different number of rows: " + std::to_string(A.rows()) + " and " +
                        std::to_string(out.rows()));
  }
  if (static_cast<std::size_t>(out.cols()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "number of columns of out is wrong, it should be " + std::to_string(nu_) + " instead of " +
                        std::to_string(out.cols()));
  }
  switch (op) {
    case setto:
      out.noalias() = A * Fu;
      break;
    case addto:
      out.noalias() += A * Fu;
      break;
    case rmfrom:
      out.noalias() -= A * Fu;
      break;
    default:
      throw_pretty("Invalid argument: allowed operators: setto, addto, rmfrom");
  }
}

template <typename Scalar>
void ActionModelAbstractTpl<Scalar>::multiplyFuTransposeBy(const Eigen::Ref<const MatrixXs>& FuTranspose,
                                                           const Eigen::Ref<const MatrixXs>& A,
                                                           Eigen::Ref<MatrixXsRowMajor> out,
                                                           const AssignmentOp op) const {
  assert_pretty(is_a_AssignmentOp(op), ("op must be one of the AssignmentOp {settop, addto, rmfrom}"));
  if (static_cast<std::size_t>(A.rows()) != state_->get_ndx()) {
    throw_pretty("Invalid argument: "
                 << "number of rows of A is wrong, it should be " + std::to_string(state_->get_ndx()) +
                        " instead of " + std::to_string(A.rows()));
  }
  if (A.cols() != out.cols()) {
    throw_pretty("Invalid argument: "
                 << "A and out have different number of columns: " + std::to_string(A.cols()) + " and " +
                        std::to_string(out.cols()));
  }
  if (static_cast<std::size_t>(out.rows()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "number of rows of out is wrong, it should be " + std::to_string(nu_) + " instead of " +
                        std::to_string(out.cols()));
  }
  switch (op) {
    case setto:
      out.noalias() = FuTranspose * A;
      break;
    case addto:
      out.noalias() += FuTranspose * A;
      break;
    case rmfrom:
      out.noalias() -= FuTranspose * A;
      break;
    default:
      throw_pretty("Invalid argument: allowed operators: setto, addto, rmfrom");
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
typename MathBaseTpl<Scalar>::MatrixXs ActionModelAbstractTpl<Scalar>::multiplyByFx_A(
    const Eigen::Ref<const MatrixXs>& Fx, const Eigen::Ref<const MatrixXs>& A) {
  MatrixXs out(A.rows(), state_->get_ndx());
  out.setZero();
  multiplyByFx(Fx, A, out);
  return out;
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::MatrixXsRowMajor ActionModelAbstractTpl<Scalar>::multiplyFxTransposeBy_A(
    const Eigen::Ref<const MatrixXs>& FxTranspose, const Eigen::Ref<const MatrixXs>& A) {
  MatrixXsRowMajor out(state_->get_ndx(), A.cols());
  out.setZero();
  multiplyFxTransposeBy(FxTranspose, A, out);
  return out;
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::MatrixXs ActionModelAbstractTpl<Scalar>::multiplyByFu_A(
    const Eigen::Ref<const MatrixXs>& Fu, const Eigen::Ref<const MatrixXs>& A) {
  MatrixXs out(A.rows(), nu_);
  out.setZero();
  multiplyByFu(Fu, A, out);
  return out;
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::MatrixXsRowMajor ActionModelAbstractTpl<Scalar>::multiplyFuTransposeBy_A(
    const Eigen::Ref<const MatrixXs>& FuTranspose, const Eigen::Ref<const MatrixXs>& A) {
  MatrixXsRowMajor out(nu_, A.cols());
  out.setZero();
  multiplyFuTransposeBy(FuTranspose, A, out);
  return out;
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
void ActionModelAbstractTpl<Scalar>::print(std::ostream& os) const {
  os << boost::core::demangle(typeid(*this).name());
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
std::ostream& operator<<(std::ostream& os, const ActionModelAbstractTpl<Scalar>& model) {
  model.print(os);
  return os;
}

}  // namespace crocoddyl
