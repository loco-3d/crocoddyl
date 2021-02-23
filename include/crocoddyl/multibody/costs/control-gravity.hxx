///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/rnea.hpp>

namespace crocoddyl {

template <typename Scalar>
CostModelControlGravTpl<Scalar>::CostModelControlGravTpl(boost::shared_ptr<StateMultibody> state,
                                                         boost::shared_ptr<ActivationModelAbstract> activation,
                                                         const std::size_t nu)
    : Base(state, activation, boost::make_shared<ResidualModelControlGrav>(state, nu)) {
  if (activation_->get_nr() != state_->get_nv()) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(state_->get_nv()));
  }
}

template <typename Scalar>
CostModelControlGravTpl<Scalar>::CostModelControlGravTpl(boost::shared_ptr<StateMultibody> state,
                                                         boost::shared_ptr<ActivationModelAbstract> activation)
    : Base(state, activation, boost::make_shared<ResidualModelControlGrav>(state)) {
  if (activation_->get_nr() != state_->get_nv()) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(state_->get_nv()));
  }
}

template <typename Scalar>
CostModelControlGravTpl<Scalar>::CostModelControlGravTpl(boost::shared_ptr<StateMultibody> state, const std::size_t nu)
    : Base(state, boost::make_shared<ResidualModelControlGrav>(state, nu)) {}

template <typename Scalar>
CostModelControlGravTpl<Scalar>::CostModelControlGravTpl(boost::shared_ptr<StateMultibody> state)
    : Base(state, boost::make_shared<ResidualModelControlGrav>(state)) {}

template <typename Scalar>
CostModelControlGravTpl<Scalar>::~CostModelControlGravTpl() {}

template <typename Scalar>
void CostModelControlGravTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract> &data,
                                           const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u) {
  // Compute the cost residual
  residual_->calc(data->residual, x, u);

  // Compute the cost
  activation_->calc(data->activation, data->residual->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelControlGravTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract> &data,
                                               const Eigen::Ref<const VectorXs> &x,
                                               const Eigen::Ref<const VectorXs> &u) {
  // Compute the derivatives of the activation and control gravity residual models
  Data *d = static_cast<Data *>(data.get());
  residual_->calcDiff(data->residual, x, u);
  activation_->calcDiff(data->activation, data->residual->r);

  // Compute the derivatives of the cost function based on a Gauss-Newton approximation
  const std::size_t nv = state_->get_nv();
  Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, true> Rq = data->residual->Rx.leftCols(nv);
  data->Lx.head(nv).noalias() = Rq.transpose() * data->activation->Ar;
  data->Lu.noalias() = data->residual->Ru.transpose() * data->activation->Ar;
  d->Arr_Rq.noalias() = data->activation->Arr * Rq;
  d->Arr_Ru.noalias() = data->activation->Arr * data->residual->Ru;
  data->Lxx.topLeftCorner(nv, nv).noalias() = Rq.transpose() * d->Arr_Rq;
  data->Lxu.topRows(nv).noalias() = Rq.transpose() * d->Arr_Ru;
  data->Luu.noalias() = data->residual->Ru.transpose() * d->Arr_Ru;
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelControlGravTpl<Scalar>::createData(
    DataCollectorAbstract *const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

}  // namespace crocoddyl
