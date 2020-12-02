///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
template <typename Scalar>
CostModelControlGravContactTpl<Scalar>::CostModelControlGravContactTpl(
    boost::shared_ptr<StateMultibody> state,
    boost::shared_ptr<ActivationModelAbstract> activation)
    : Base(state, activation), pin_model_(state->get_pinocchio()) {}

template <typename Scalar>
CostModelControlGravContactTpl<Scalar>::CostModelControlGravContactTpl(
    boost::shared_ptr<StateMultibody> state,
    boost::shared_ptr<ActivationModelAbstract> activation,
    const std::size_t &nu)
    : Base(state, activation, nu), pin_model_(state->get_pinocchio()) {
  if (activation_->get_nr() != nu_) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(nu_));
  }
}

template <typename Scalar>
CostModelControlGravContactTpl<Scalar>::CostModelControlGravContactTpl(
    boost::shared_ptr<StateMultibody> state)
    : Base(state, state->get_nv()), pin_model_(state->get_pinocchio()) {}

template <typename Scalar>
CostModelControlGravContactTpl<Scalar>::CostModelControlGravContactTpl(
    boost::shared_ptr<StateMultibody> state, const std::size_t &nu)
    : Base(state, nu, nu), pin_model_(state->get_pinocchio()) {}

template <typename Scalar>
CostModelControlGravContactTpl<Scalar>::~CostModelControlGravContactTpl() {}

template <typename Scalar>
void CostModelControlGravContactTpl<Scalar>::calc(
    const boost::shared_ptr<CostDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u) {
  if (nu_ == 0) {
    throw_pretty("Invalid argument: "
                 << "it seems to be an autonomous system, if so, don't add "
                    "this cost function");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " +
                        std::to_string(nu_) + ")");
  }
  Data *d = static_cast<Data *>(data.get());

  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());

  data->r = u - pinocchio::computeStaticTorque(*pin_model_.get(), *d->pinocchio, q, d->fext).tail(nu_);
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelControlGravContactTpl<Scalar>::calcDiff(
    const boost::shared_ptr<CostDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u) {

  if (nu_ == 0) {
    throw_pretty("Invalid argument: "
                 << "it seems to be an autonomous system, if so, don't add "
                    "this cost function");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " +
                        std::to_string(nu_) + ")");
  }
  Data *d = static_cast<Data *>(data.get());

  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());

  pinocchio::computeStaticTorqueDerivatives(*pin_model_.get(), *d->pinocchio, q,
                                    d->fext, d->dg_dx.topRows(state_->get_nv()));

  activation_->calcDiff(data->activation, data->r);
  
  data->Lu.noalias() =  data->activation->Ar; 
  data->Lx.noalias() =  -d->dg_dx.rightCols(nu_) * data->activation->Ar; 

  d->Arr_dgdx.noalias() = data->activation->Arr * (d->dg_dx.rightCols(nu_)).transpose(); 
  data->Lxx.noalias() =  d->dg_dx.rightCols(nu_) * d->Arr_dgdx; 

  data->Lxu.noalias() =  -d->Arr_dgdx; 
  data->Luu.diagonal().noalias() = data->activation->Arr.diagonal();
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar>>
CostModelControlGravContactTpl<Scalar>::createData(
    DataCollectorAbstract *const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                      data);
}

} // namespace crocoddyl
