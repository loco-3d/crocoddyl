///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
template <typename Scalar>
CostModelControlGravTpl<Scalar>::CostModelControlGravTpl(boost::shared_ptr<StateMultibody> state,
                                                 boost::shared_ptr<ActivationModelAbstract> activation)
    : Base(state, activation), pin_model_(state->get_pinocchio()) {
}

template <typename Scalar>
CostModelControlGravTpl<Scalar>::CostModelControlGravTpl(boost::shared_ptr<StateMultibody> state,
                                                 boost::shared_ptr<ActivationModelAbstract> activation,
                                                 const std::size_t& nu)
    : Base(state, activation, nu), pin_model_(state->get_pinocchio()) {
  if (activation_->get_nr() != nu_) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(nu_));
  }
}

template <typename Scalar>
CostModelControlGravTpl<Scalar>::CostModelControlGravTpl(boost::shared_ptr<StateMultibody> state)
    : Base(state, state->get_nv()), pin_model_(state->get_pinocchio()) {
}

template <typename Scalar>
CostModelControlGravTpl<Scalar>::CostModelControlGravTpl(boost::shared_ptr<StateMultibody> state,
                                                 const std::size_t& nu)
    : Base(state, nu, nu), pin_model_(state->get_pinocchio()) {
}

template <typename Scalar>
CostModelControlGravTpl<Scalar>::~CostModelControlGravTpl() {}

template <typename Scalar>
void CostModelControlGravTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                       const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (nu_ == 0) {
    throw_pretty("Invalid argument: "
                 << "it seems to be an autonomous system, if so, don't add this cost function");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  //const boost::shared_ptr<StateMultibody>& state = boost::static_pointer_cast<StateMultibody>(state_);
  data->r = u - pinocchio::rnea(*pin_model_,*(d->pinocchio),x.head(state_->get_nq()),x.tail(state_->get_nv()),Eigen::VectorXd::Zero(state_->get_nv()));

  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelControlGravTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                           const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  if (nu_ == 0) {
    throw_pretty("Invalid argument: "
                 << "it seems to be an autonomous system, if so, don't add this cost function");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  //const boost::shared_ptr<StateMultibody>& state = boost::static_pointer_cast<StateMultibody>(state_);
  pinocchio::computeRNEADerivatives(*pin_model_,*(d->pinocchio),x.head(state_->get_nq()),x.tail(state_->get_nv()),Eigen::VectorXd::Zero(state_->get_nv()),
                                    d->rnea_partial_dx.topRows(state_->get_nq()),d->rnea_partial_dx.bottomRows(state_->get_nv()),d->rnea_partial_da);

  activation_->calcDiff(data->activation, data->r);
  
  data->Lu = data->activation->Ar;
  data->Lx = - d->rnea_partial_dx * data->activation->Ar;
  data->Lxx = d->rnea_partial_dx * data->activation->Arr * d->rnea_partial_dx.transpose();
  data->Lxu = - d->rnea_partial_dx * data->activation->Arr;
  data->Luu.diagonal() = data->activation->Arr.diagonal();
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar> > CostModelControlGravTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this, data);
}

}  // namespace crocoddyl
