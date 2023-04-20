///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2022, Heriot-Watt University, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

using std::isfinite;

template <typename Scalar>
ConstraintModelResidualTpl<Scalar>::ConstraintModelResidualTpl(
    boost::shared_ptr<typename Base::StateAbstract> state,
    boost::shared_ptr<ResidualModelAbstract> residual, const VectorXs& lower,
    const VectorXs& upper)
    : Base(state, residual, residual->get_nr(), 0) {
  lb_ = lower;
  ub_ = upper;
  for (std::size_t i = 0; i < residual_->get_nr(); ++i) {
    if (isfinite(lb_(i)) && isfinite(ub_(i))) {
      if (lb_(i) - ub_(i) > 0) {
        throw_pretty(
            "Invalid argument: the upper bound is not equal to / higher than "
            "the lower bound.")
      }
    }
  }
  if ((lb_.array() == std::numeric_limits<Scalar>::infinity()).any() ||
      (lb_.array() == std::numeric_limits<Scalar>::max()).any()) {
    throw_pretty(
        "Invalid argument: the lower bound cannot contain a positive "
        "infinity/max value");
  }
  if ((ub_.array() == -std::numeric_limits<Scalar>::infinity()).any() ||
      (ub_.array() == -std::numeric_limits<Scalar>::infinity()).any()) {
    throw_pretty(
        "Invalid argument: the lower bound cannot contain a negative "
        "infinity/max value");
  }
}

template <typename Scalar>
ConstraintModelResidualTpl<Scalar>::ConstraintModelResidualTpl(
    boost::shared_ptr<typename Base::StateAbstract> state,
    boost::shared_ptr<ResidualModelAbstract> residual)
    : Base(state, residual, 0, residual->get_nr()) {}

template <typename Scalar>
ConstraintModelResidualTpl<Scalar>::~ConstraintModelResidualTpl() {}

template <typename Scalar>
void ConstraintModelResidualTpl<Scalar>::calc(
    const boost::shared_ptr<ConstraintDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  // Compute the constraint residual
  residual_->calc(data->residual, x, u);

  // Fill the residual values for its corresponding type of constraint
  updateCalc(data);
}

template <typename Scalar>
void ConstraintModelResidualTpl<Scalar>::calc(
    const boost::shared_ptr<ConstraintDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  // Compute the constraint residual
  residual_->calc(data->residual, x);

  // Fill the residual values for its corresponding type of constraint
  updateCalc(data);
}

template <typename Scalar>
void ConstraintModelResidualTpl<Scalar>::calcDiff(
    const boost::shared_ptr<ConstraintDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  // Compute the derivatives of the residual function
  residual_->calcDiff(data->residual, x, u);

  // Fill the residual values for its corresponding type of constraint
  updateCalcDiff(data);
}

template <typename Scalar>
void ConstraintModelResidualTpl<Scalar>::calcDiff(
    const boost::shared_ptr<ConstraintDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  // Compute the derivatives of the residual function
  residual_->calcDiff(data->residual, x);

  // Fill the residual values for its corresponding type of constraint
  updateCalcDiff(data);
}

template <typename Scalar>
boost::shared_ptr<ConstraintDataAbstractTpl<Scalar> >
ConstraintModelResidualTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                      data);
}

template <typename Scalar>
void ConstraintModelResidualTpl<Scalar>::updateCalc(
    const boost::shared_ptr<ConstraintDataAbstract>& data) {
  switch (type_) {
    case ConstraintType::Inequality:
      data->g = data->residual->r;
      break;
    case ConstraintType::Equality:
      data->h = data->residual->r;
      break;
    case ConstraintType::Both:  // this condition is not supported and possible
      break;
  }
}

template <typename Scalar>
void ConstraintModelResidualTpl<Scalar>::updateCalcDiff(
    const boost::shared_ptr<ConstraintDataAbstract>& data) {
  const bool is_rq = residual_->get_q_dependent();
  const bool is_rv = residual_->get_v_dependent();
  const bool is_ru = residual_->get_u_dependent() || nu_ == 0;
  switch (type_) {
    case ConstraintType::Inequality:
      if (is_rq && is_rv) {
        data->Gx = data->residual->Rx;
      } else if (is_rq) {
        const std::size_t nv = state_->get_nv();
        data->Gx.leftCols(nv) = data->residual->Rx.leftCols(nv);
        data->Gx.rightCols(nv).setZero();
      } else if (is_rv) {
        const std::size_t nv = state_->get_nv();
        data->Gx.leftCols(nv).setZero();
        data->Gx.rightCols(nv) = data->residual->Rx.rightCols(nv);
      }
      if (is_ru) {
        data->Gu = data->residual->Ru;
      }
      break;
    case ConstraintType::Equality:
      if (is_rq && is_rv) {
        data->Hx = data->residual->Rx;
      } else if (is_rq) {
        const std::size_t nv = state_->get_nv();
        data->Hx.leftCols(nv) = data->residual->Rx.leftCols(nv);
        data->Hx.rightCols(nv).setZero();
      } else if (is_rv) {
        const std::size_t nv = state_->get_nv();
        data->Hx.leftCols(nv).setZero();
        data->Hx.rightCols(nv) = data->residual->Rx.rightCols(nv);
      }
      if (is_ru) {
        data->Hu = data->residual->Ru;
      }
      break;
    case ConstraintType::Both:  // this condition is not supported and possible
      break;
  }
}

template <typename Scalar>
void ConstraintModelResidualTpl<Scalar>::print(std::ostream& os) const {
  os << "ConstraintModelResidual {" << *residual_ << "}";
}

}  // namespace crocoddyl
