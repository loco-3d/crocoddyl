///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"
#include "pinocchio/algorithm/rnea.hpp"

namespace crocoddyl {
template <typename Scalar>
CostModelControlGravTpl<Scalar>::CostModelControlGravTpl(
    boost::shared_ptr<StateMultibody> state,
    boost::shared_ptr<ActivationModelAbstract> activation,
    boost::shared_ptr<ActuationModelFull> actuation_model)
    : Base(state, activation,actuation_model->get_nu()), pin_model_(*state->get_pinocchio()), actuation_model_(*actuation_model) {  
   if (activation_->get_nr() != state_->get_nv()) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(state_->get_nv()));
  }
  if (nu_ == 0) {
    throw_pretty("Invalid argument: "
                 << "it seems to be an autonomous system, if so, don't add "
                    "this cost function");
  }
}

template <typename Scalar>
CostModelControlGravTpl<Scalar>::CostModelControlGravTpl(
    boost::shared_ptr<StateMultibody> state, boost::shared_ptr<ActuationModelFull> actuation_model)
    : Base(state, state->get_nv(), actuation_model->get_nu()), 
           pin_model_(*state->get_pinocchio()), actuation_model_(*actuation_model) {
   if (nu_ == 0) {
    throw_pretty("Invalid argument: "
                 << "it seems to be an autonomous system, if so, don't add "
                    "this cost function");
  }
}

template <typename Scalar>
CostModelControlGravTpl<Scalar>::~CostModelControlGravTpl() {}

template <typename Scalar>
void CostModelControlGravTpl<Scalar>::calc(
    const boost::shared_ptr<CostDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u) {
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " +
                        std::to_string(nu_) + ")");
  }
  Data *d = static_cast<Data *>(data.get());
  
  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());
  
  actuation_model_.calc(d->actuation,x,u); 
  data->r = d->actuation->tau - pinocchio::computeGeneralizedGravity(pin_model_, *d->pinocchio, q);

  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelControlGravTpl<Scalar>::calcDiff(
    const boost::shared_ptr<CostDataAbstract> &data,
    const Eigen::Ref<const VectorXs> &x, const Eigen::Ref<const VectorXs> &u) {
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " +
                        std::to_string(nu_) + ")");
  }
  Data *d = static_cast<Data *>(data.get());

  const Eigen::VectorBlock<const Eigen::Ref<const VectorXs>, Eigen::Dynamic> q =
      x.head(state_->get_nq());
  
  pinocchio::computeGeneralizedGravityDerivatives(pin_model_, *d->pinocchio, q,
                                    d->dg_dq);
                                    
  const std::size_t& nv = state_->get_nv();
  activation_->calcDiff(data->activation, data->r);
  actuation_model_.calcDiff(d->actuation,x,u); 

  data->Lu.noalias() = d->actuation->dtau_du.transpose() * data->activation->Ar;
  data->Lx.noalias() = d->actuation->dtau_dx.transpose() * data->activation->Ar;
  data->Lx.head(nv).noalias() -= d->dg_dq.transpose() * data->activation->Ar;

  d->Arr_dgdq.noalias() = data->activation->Arr * d->dg_dq;
  d->Arr_dtaudx.noalias() = data->activation->Arr * d->actuation->dtau_dx;
  d->Arr_dtaudu.noalias() = data->activation->Arr * d->actuation->dtau_du;
  data->Lxx.noalias() = d->actuation->dtau_dx.transpose() * d->Arr_dtaudx;
  data->Lxx.topLeftCorner(nv,nv).noalias() += d->dg_dq.transpose() * d->Arr_dgdq;
  data->Lxx.topRows(nv).noalias() -= d->dg_dq.transpose() * d->Arr_dtaudx;
  data->Lxx.leftCols(nv).noalias() -= d->actuation->dtau_dx.transpose() * d->Arr_dgdq;
  
  data->Lxu.noalias() = d->Arr_dtaudx.transpose() * d->actuation->dtau_du;
  data->Lxu.topRows(nv).noalias() -= d->Arr_dgdq.transpose() * d->actuation->dtau_du;
  data->Luu.diagonal().noalias() = (d->actuation->dtau_du.transpose() * d->Arr_dtaudu).diagonal();
}

template <typename Scalar>
boost::shared_ptr<CostDataAbstractTpl<Scalar>>
CostModelControlGravTpl<Scalar>::createData(DataCollectorAbstract *const data) {
  return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this,
                                      data);
}

} // namespace crocoddyl
