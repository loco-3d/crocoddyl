///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename Scalar>
ConstraintModelResidualTpl<Scalar>::ConstraintModelResidualTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                                               boost::shared_ptr<ResidualModelAbstract> residual,
                                                               const VectorXs& lower, const VectorXs& upper)
    : Base(state, residual,
           2 * ((upper - lower).array() != 0.).count() -
               (upper.array() == std::numeric_limits<Scalar>::infinity()).count() -
               (lower.array() == -std::numeric_limits<Scalar>::infinity()).count(),
           ((upper - lower).array() == 0.).count()),
      lb_(lower),
      ub_(upper),
      constraint_type_(residual->get_nr()) {
  for (std::size_t i = 0; i < residual_->get_nr(); ++i) {
    if (isfinite(lb_(i)) && isfinite(ub_(i))) {
      if (lb_(i) - ub_(i) > 0) {
        throw_pretty("Invalid argument: the upper bound is not equals / higher than the lower bound.")
      }
    }
  }
  if ((lb_.array() == std::numeric_limits<Scalar>::infinity()).any()) {
    throw_pretty("Invalid argument: the lower bound cannot contain a positive infinity value");
  }
  if ((ub_.array() == -std::numeric_limits<Scalar>::infinity()).any()) {
    throw_pretty("Invalid argument: the lower bound cannot contain a negative infinity value");
  }
  updateConstraintType();
}

template <typename Scalar>
ConstraintModelResidualTpl<Scalar>::ConstraintModelResidualTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                                               boost::shared_ptr<ResidualModelAbstract> residual)
    : Base(state, residual, 0, residual->get_nr()),
      lb_(VectorXs::Zero(nh_)),
      ub_(VectorXs::Zero(nh_)),
      constraint_type_(residual->get_nr()) {
  constraint_type_ = 0;
}

template <typename Scalar>
ConstraintModelResidualTpl<Scalar>::~ConstraintModelResidualTpl() {}

template <typename Scalar>
void ConstraintModelResidualTpl<Scalar>::calc(const boost::shared_ptr<ConstraintDataAbstract>& data,
                                              const Eigen::Ref<const VectorXs>& x,
                                              const Eigen::Ref<const VectorXs>& u) {
  // Compute the constraint residual
  residual_->calc(data->residual, x, u);

  // Fill the residual values for its corresponding type of constraint
  std::size_t nh_i = 0;
  std::size_t ng_i = 0;
  const std::size_t nr = residual_->get_nr();
  for (std::size_t i = 0; i < nr; ++i) {
    if (constraint_type_(i) == 0) {  // equality constraint
      data->h(nh_i) = data->residual->r(i) - ub_(i);
      ++nh_i;
    } else if (constraint_type_(i) == 1) {  // inequality constraint
      data->g(ng_i) = data->residual->r(i) - ub_(i);
      data->g(ng_i + 1) = lb_(i) - data->residual->r(i);
      ng_i += 2;
    } else if (constraint_type_(i) == 2) {  // lower inequality constraint
      data->g(ng_i) = lb_(i) - data->residual->r(i);
      ++ng_i;
    } else if (constraint_type_(i) == 3) {
      data->g(ng_i) = data->residual->r(i) - ub_(i);
      ++ng_i;
    }
  }
}

template <typename Scalar>
void ConstraintModelResidualTpl<Scalar>::calcDiff(const boost::shared_ptr<ConstraintDataAbstract>& data,
                                                  const Eigen::Ref<const VectorXs>& x,
                                                  const Eigen::Ref<const VectorXs>& u) {
  // Compute the derivatives of the residual function
  // Data* d = static_cast<Data*>(data.get());
  residual_->calcDiff(data->residual, x, u);

  // Fill the residual values for its corresponding type of constraint
  std::size_t nh_i = 0;
  std::size_t ng_i = 0;
  const std::size_t nv = state_->get_nv();
  const std::size_t nr = residual_->get_nr();
  const bool is_rq = residual_->get_q_dependent();
  const bool is_rv = residual_->get_v_dependent();
  const bool is_ru = residual_->get_u_dependent() || nu_ == 0;
  for (std::size_t i = 0; i < nr; ++i) {
    if (constraint_type_(i) == 0) {  // equality constraint
      if (is_rq && is_rv) {
        data->Hx.row(nh_i) = data->residual->Rx.row(i);
      } else if (is_rq) {
        data->Hx.row(nh_i).head(nv) = data->residual->Rx.row(i).head(nv);
      } else if (is_rv) {
        data->Hx.row(nh_i).tail(nv) = data->residual->Rx.row(i).tail(nv);
      }
      if (is_ru) {
        data->Hu.row(nh_i) = data->residual->Ru.row(i);
      }
      ++nh_i;
    } else if (constraint_type_(i) == 1) {  // inequality constraint
      if (is_rq && is_rv) {
        data->Gx.row(ng_i) = data->residual->Rx.row(i);
        data->Gx.row(ng_i + 1) = -data->residual->Rx.row(i);
      } else if (is_rq) {
        data->Gx.row(ng_i).head(nv) = data->residual->Rx.row(i).head(nv);
        data->Gx.row(ng_i + 1).head(nv) = -data->residual->Rx.row(i).head(nv);
      } else if (is_rv) {
        data->Gx.row(ng_i).tail(nv) = data->residual->Rx.row(i).tail(nv);
        data->Gx.row(ng_i + 1).tail(nv) = -data->residual->Rx.row(i).tail(nv);
      }
      if (is_ru) {
        data->Gu.row(ng_i) = data->residual->Ru.row(i);
        data->Gu.row(ng_i + 1) = -data->residual->Ru.row(i);
      }
      ng_i += 2;
    } else if (constraint_type_(i) == 2) {  // lower inequality constraint
      if (is_rq && is_rv) {
        data->Gx.row(ng_i) = -data->residual->Rx.row(i);
      } else if (is_rq) {
        data->Gx.row(ng_i).head(nv) = -data->residual->Rx.row(i).head(nv);
      } else if (is_rv) {
        data->Gx.row(ng_i).tail(nv) = -data->residual->Rx.row(i).tail(nv);
      }
      if (is_ru) {
        data->Gu.row(ng_i) = -data->residual->Ru.row(i);
      }
      ++ng_i;
    } else if (constraint_type_(i) == 3) {
      if (is_rq && is_rv) {
        data->Gx.row(ng_i) = data->residual->Rx.row(i);
      } else if (is_rq) {
        data->Gx.row(ng_i).head(nv) = data->residual->Rx.row(i).head(nv);
      } else if (is_rv) {
        data->Gx.row(ng_i).tail(nv) = data->residual->Rx.row(i).tail(nv);
      }
      if (is_ru) {
        data->Gu.row(ng_i) = data->residual->Ru.row(i);
      }
      ++ng_i;
    }
  }
}

template <typename Scalar>
boost::shared_ptr<ConstraintDataAbstractTpl<Scalar> > ConstraintModelResidualTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& ConstraintModelResidualTpl<Scalar>::get_ub() const {
  return ub_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& ConstraintModelResidualTpl<Scalar>::get_lb() const {
  return lb_;
}

template <typename Scalar>
void ConstraintModelResidualTpl<Scalar>::update_bounds(const VectorXs& lower, const VectorXs& upper) {
  if (((upper - lower).array() < 0.).any()) {
    throw_pretty("Invalid argument: the upper bound is not equals / higher than the lower bound.")
  }
  if ((lower.array() == std::numeric_limits<Scalar>::infinity()).any()) {
    throw_pretty("Invalid argument: the lower bound cannot containt a positive infinity value");
  }
  if ((upper.array() == -std::numeric_limits<Scalar>::infinity()).any()) {
    throw_pretty("Invalid argument: the lower bound cannot containt a negative infinity value");
  }

  // Update the information related to the bounds
  ng_ = 2 * ((upper - lower).array() != 0.).count() -
        (upper.array() == std::numeric_limits<Scalar>::infinity()).count() -
        (lower.array() == -std::numeric_limits<Scalar>::infinity()).count();
  nh_ = ((upper - lower).array() == 0.).count();
  lb_ = lower;
  ub_ = upper;
  updateConstraintType();
}

template <typename Scalar>
void ConstraintModelResidualTpl<Scalar>::updateConstraintType() {
  const std::size_t nr = residual_->get_nr();
  for (std::size_t i = 0; i < nr; ++i) {
    if (ub_(i) - lb_(i) == 0.) {
      constraint_type_(i) = 0;
    } else if (!std::isinf(lb_(i)) && !std::isinf(ub_(i))) {
      constraint_type_(i) = 1;
    } else if (std::isinf(lb_(i)) && std::isinf(ub_(i))) {
      constraint_type_(i) = 4;
    } else if (std::isinf(ub_(i))) {
      constraint_type_(i) = 2;
    } else {
      constraint_type_(i) = 3;
    }
  }
}

}  // namespace crocoddyl
