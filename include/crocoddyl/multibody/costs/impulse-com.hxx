///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/costs/impulse-com.hpp"

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <pinocchio/algorithm/center-of-mass-derivatives.hpp>

namespace crocoddyl {

template <typename Scalar>
CostModelImpulseCoMTpl<Scalar>::CostModelImpulseCoMTpl(boost::shared_ptr<StateMultibody> state,
                                                       boost::shared_ptr<ActivationModelAbstract> activation)
    : Base(state, activation, boost::make_shared<ResidualModelImpulseCoM>(state)) {
  if (activation_->get_nr() != 3) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to 3");
  }
}

template <typename Scalar>
CostModelImpulseCoMTpl<Scalar>::CostModelImpulseCoMTpl(boost::shared_ptr<StateMultibody> state)
    : Base(state, boost::make_shared<ResidualModelImpulseCoM>(state)) {}

template <typename Scalar>
CostModelImpulseCoMTpl<Scalar>::~CostModelImpulseCoMTpl() {}

template <typename Scalar>
void CostModelImpulseCoMTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                          const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  // Compute the cost residual given the impulse CoM
  Data* d = static_cast<Data*>(data.get());
  residual_->calc(d->residual, x, u);

  // Compute the cost
  activation_->calc(data->activation, data->residual->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelImpulseCoMTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                              const Eigen::Ref<const VectorXs>& x,
                                              const Eigen::Ref<const VectorXs>& u) {
  // Compute the derivatives of the activation and impulse CoM residual models
  Data* d = static_cast<Data*>(data.get());
  residual_->calcDiff(d->residual, x, u);
  activation_->calcDiff(d->activation, data->residual->r);

  // Compute the derivatives of the cost function based on a Gauss-Newton approximation
  d->Arr_Rx.noalias() = data->activation->Arr * data->residual->Rx;
  data->Lx.noalias() = data->residual->Rx.transpose() * data->activation->Ar;
  data->Lxx.noalias() = data->residual->Rx.transpose() * d->Arr_Rx;
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelImpulseCoMTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

}  // namespace crocoddyl
