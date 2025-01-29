///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2023, University of Edinburgh, INRIA,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
template <typename Scalar>
CostModelResidualTpl<Scalar>::CostModelResidualTpl(
    std::shared_ptr<typename Base::StateAbstract> state,
    std::shared_ptr<ActivationModelAbstract> activation,
    std::shared_ptr<ResidualModelAbstract> residual)
    : Base(state, activation, residual) {}

template <typename Scalar>
CostModelResidualTpl<Scalar>::CostModelResidualTpl(
    std::shared_ptr<typename Base::StateAbstract> state,
    std::shared_ptr<ResidualModelAbstract> residual)
    : Base(state, residual) {}

template <typename Scalar>
CostModelResidualTpl<Scalar>::~CostModelResidualTpl() {}

template <typename Scalar>
void CostModelResidualTpl<Scalar>::calc(
    const std::shared_ptr<CostDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  // Compute the cost residual
  residual_->calc(data->residual, x, u);

  // Compute the cost
  activation_->calc(data->activation, data->residual->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelResidualTpl<Scalar>::calc(
    const std::shared_ptr<CostDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  const bool is_rq = residual_->get_q_dependent();
  const bool is_rv = residual_->get_v_dependent();
  if (!is_rq && !is_rv) {
    data->activation->a_value = 0.;
    data->cost = 0.;
    return;  // do nothing
  }

  // Compute the cost residual
  residual_->calc(data->residual, x);

  // Compute the cost
  activation_->calc(data->activation, data->residual->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelResidualTpl<Scalar>::calcDiff(
    const std::shared_ptr<CostDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  // Compute the derivatives of the activation and contact wrench cone residual
  // models
  residual_->calcDiff(data->residual, x, u);
  activation_->calcDiff(data->activation, data->residual->r);

  // Compute the derivatives of the cost function based on a Gauss-Newton
  // approximation
  residual_->calcCostDiff(data, data->residual, data->activation);
}

template <typename Scalar>
void CostModelResidualTpl<Scalar>::calcDiff(
    const std::shared_ptr<CostDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  // Compute the derivatives of the activation and contact wrench cone residual
  // models
  const bool is_rq = residual_->get_q_dependent();
  const bool is_rv = residual_->get_v_dependent();
  if (!is_rq && !is_rv) {
    data->Lx.setZero();
    data->Lxx.setZero();
    return;  // do nothing
  }
  residual_->calcDiff(data->residual, x);
  activation_->calcDiff(data->activation, data->residual->r);

  // Compute the derivatives of the cost function based on a Gauss-Newton
  // approximation
  residual_->calcCostDiff(data, data->residual, data->activation, false);
}

template <typename Scalar>
std::shared_ptr<CostDataAbstractTpl<Scalar> >
CostModelResidualTpl<Scalar>::createData(DataCollectorAbstract* const data) {
  return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                    data);
}

template <typename Scalar>
void CostModelResidualTpl<Scalar>::print(std::ostream& os) const {
  os << "CostModelResidual {" << *residual_ << ", " << *activation_ << "}";
}

}  // namespace crocoddyl
