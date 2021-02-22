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
  Data* d = static_cast<Data*>(data.get());
  residual_->calcDiff(data->residual, x, u);
  activation_->calcDiff(data->activation, data->residual->r);

  // Compute the derivatives of the cost function based on a Gauss-Newton approximation
  const bool is_rq = residual_->get_q_dependent();
  const bool is_rv = residual_->get_v_dependent();
  const bool is_ru = residual_->get_u_dependent() || nu_ == 0;
  const std::size_t nv = state_->get_nv();
  if (is_ru) {
    d->Arr_Ru.noalias() = data->activation->Arr * data->residual->Ru;
    data->Luu.noalias() = data->residual->Ru.transpose() * d->Arr_Ru;
  }
  if (is_rq && is_rv) {
    data->Lx.noalias() = data->residual->Rx.transpose() * data->activation->Ar;
    d->Arr_Rx.noalias() = data->activation->Arr * data->residual->Rx;
    data->Lxx.noalias() = data->residual->Rx.transpose() * d->Arr_Rx;
    if (is_ru) {
      data->Lxu.noalias() = data->residual->Rx.transpose() * d->Arr_Ru;
    }
  } else if (is_rq) {
    Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, true> Rq = data->residual->Rx.leftCols(nv);
    data->Lx.head(nv).noalias() = Rq.transpose() * data->activation->Ar;
    data->Arr_Rx.head(nv).noalias() = data->activation->Arr * Rq;
    data->Lxx.topLeftCorner(nv, nv).noalias() = Rq.transpose() * data->Arr_Rx.head(nv);
    if (is_ru) {
      data->Lxu.topRows(nv).noalias() = Rq.transpose() * d->Arr_Ru;
    }
  } else if (is_rv) {
    Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, true> Rv = data->residual->Rx.rightCols(nv);
    data->Lx.tail(nv).noalias() = Rv.transpose() * data->activation->Ar;
    data->Arr_Rx.tail(nv).noalias() = data->activation->Arr * Rv;
    data->Lxx.bottomRightCorner(nv, nv).noalias() = Rv.transpose() * data->Arr_Rx.tail(nv);
    if (is_ru) {
      data->Lxu.bottomRows(nv).noalias() = Rv.transpose() * d->Arr_Ru;
    }
  }
}

}  // namespace crocoddyl
