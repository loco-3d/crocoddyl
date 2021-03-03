///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
template <typename Scalar>
CostModelControlTpl<Scalar>::CostModelControlTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                                 boost::shared_ptr<ActivationModelAbstract> activation,
                                                 const VectorXs& uref)
    : Base(state, activation, boost::make_shared<ResidualModelControl>(state, uref)), uref_(uref) {
  if (activation_->get_nr() != nu_) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(nu_));
  }
}

template <typename Scalar>
CostModelControlTpl<Scalar>::CostModelControlTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                                 boost::shared_ptr<ActivationModelAbstract> activation)
    : Base(state, activation, boost::make_shared<ResidualModelControl>(state)),
      uref_(VectorXs::Zero(activation->get_nr())) {
  if (activation_->get_nr() != nu_) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(nu_));
  }
}

template <typename Scalar>
CostModelControlTpl<Scalar>::CostModelControlTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                                 boost::shared_ptr<ActivationModelAbstract> activation,
                                                 const std::size_t nu)
    : Base(state, activation, boost::make_shared<ResidualModelControl>(state, nu)), uref_(VectorXs::Zero(nu)) {
  if (activation_->get_nr() != nu_) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(nu_));
  }
}

template <typename Scalar>
CostModelControlTpl<Scalar>::CostModelControlTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                                 const VectorXs& uref)
    : Base(state, boost::make_shared<ResidualModelControl>(state, uref)), uref_(uref) {}

template <typename Scalar>
CostModelControlTpl<Scalar>::CostModelControlTpl(boost::shared_ptr<typename Base::StateAbstract> state)
    : Base(state, boost::make_shared<ResidualModelControl>(state)), uref_(VectorXs::Zero(state->get_nv())) {}

template <typename Scalar>
CostModelControlTpl<Scalar>::CostModelControlTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                                 const std::size_t nu)
    : Base(state, boost::make_shared<ResidualModelControl>(state, nu)), uref_(VectorXs::Zero(nu)) {}

template <typename Scalar>
CostModelControlTpl<Scalar>::~CostModelControlTpl() {}

template <typename Scalar>
void CostModelControlTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                       const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  residual_->calc(data->residual, x, u);
  activation_->calc(data->activation, data->residual->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelControlTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                           const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  activation_->calcDiff(data->activation, data->residual->r);
  data->Lu = data->activation->Ar;
  data->Luu.diagonal() = data->activation->Arr.diagonal();
}

template <typename Scalar>
void CostModelControlTpl<Scalar>::set_referenceImpl(const std::type_info& ti, const void* pv) {
  if (ti == typeid(VectorXs)) {
    if (static_cast<std::size_t>(static_cast<const VectorXs*>(pv)->size()) != nu_) {
      throw_pretty("Invalid argument: "
                   << "reference has wrong dimension (it should be " + std::to_string(nu_) + ")");
    }
    uref_ = *static_cast<const VectorXs*>(pv);
    ResidualModelControl* residual = static_cast<ResidualModelControl*>(residual_.get());
    residual->set_reference(uref_);
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be VectorXs)");
  }
}

template <typename Scalar>
void CostModelControlTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) {
  if (ti == typeid(VectorXs)) {
    VectorXs& tmp = *static_cast<VectorXs*>(pv);
    tmp.resize(nu_);
    Eigen::Map<VectorXs> ref_map(static_cast<VectorXs*>(pv)->data(), nu_);
    ResidualModelControl* residual = static_cast<ResidualModelControl*>(residual_.get());
    uref_ = residual->get_reference();
    for (std::size_t i = 0; i < nu_; ++i) {
      ref_map[i] = uref_[i];
    }
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be VectorXs)");
  }
}

}  // namespace crocoddyl
