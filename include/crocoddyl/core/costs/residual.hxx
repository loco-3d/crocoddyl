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
CostModelResidualTpl<Scalar>::CostModelResidualTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                                   boost::shared_ptr<ResidualModelAbstract> residual,
                                                   boost::shared_ptr<ActivationModelAbstract> activation)
    : Base(state, activation, residual) {
  if (activation_->get_nr() != residual->get_nr()) {
    throw_pretty("Invalid argument: "
                 << "activation.nr is equals to residual.nr=" + residual->get_nr());
  }
}

template <typename Scalar>
CostModelResidualTpl<Scalar>::CostModelResidualTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                                   boost::shared_ptr<ResidualModelAbstract> residual)
    : Base(state, boost::make_shared<ResidualModelControl>(state)) {}

template <typename Scalar>
CostModelResidualTpl<Scalar>::~CostModelResidualTpl() {}

template <typename Scalar>
void CostModelResidualTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                        const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  // Compute the cost residual
  residual_->calc(data->residual, x, u);

  // Compute the cost
  activation_->calc(data->activation, data->residual->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelResidualTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                            const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  // Compute the derivatives of the activation and contact wrench cone residual models
  residual_->calcDiff(data->residual, x, u);
  activation_->calcDiff(data->activation, data->residual->r);

  // Compute the derivatives of the cost function based on a Gauss-Newton approximation
  data->Lx.noalias() = data->residual->Rx.transpose() * data->activation->Ar;
  data->Lu.noalias() = data->residual->Ru.transpose() * data->activation->Ar;
  d->Arr_Rx.noalias() = data->activation->Arr * data->residual->Rx;
  d->Arr_Ru.noalias() = data->activation->Arr * data->residual->Ru;
  data->Lxx.noalias() = data->residual->Rx.transpose() * d->Arr_Rx;
  data->Lxu.noalias() = data->residual->Rx.transpose() * d->Arr_Ru;
  data->Luu.noalias() = data->residual->Ru.transpose() * d->Arr_Ru;
}

}  // namespace crocoddyl
