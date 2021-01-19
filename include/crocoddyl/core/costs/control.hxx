///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
template <typename Scalar>
CostModelControlTpl<Scalar>::CostModelControlTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                                 boost::shared_ptr<ActivationModelAbstract> activation,
                                                 const VectorXs& uref)
    : Base(state, activation, static_cast<std::size_t>(uref.size())), uref_(uref) {
  if (activation_->get_nr() != nu_) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(nu_));
  }
}

template <typename Scalar>
CostModelControlTpl<Scalar>::CostModelControlTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                                 boost::shared_ptr<ActivationModelAbstract> activation)
    : Base(state, activation), uref_(VectorXs::Zero(activation->get_nr())) {}

template <typename Scalar>
CostModelControlTpl<Scalar>::CostModelControlTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                                 boost::shared_ptr<ActivationModelAbstract> activation,
                                                 const std::size_t& nu)
    : Base(state, activation, nu), uref_(VectorXs::Zero(nu)) {
  if (activation_->get_nr() != nu_) {
    throw_pretty("Invalid argument: "
                 << "nr is equals to " + std::to_string(nu_));
  }
}

template <typename Scalar>
CostModelControlTpl<Scalar>::CostModelControlTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                                 const VectorXs& uref)
    : Base(state, static_cast<std::size_t>(uref.size()), static_cast<std::size_t>(uref.size())), uref_(uref) {}

template <typename Scalar>
CostModelControlTpl<Scalar>::CostModelControlTpl(boost::shared_ptr<typename Base::StateAbstract> state)
    : Base(state, state->get_nv()), uref_(VectorXs::Zero(state->get_nv())) {}

template <typename Scalar>
CostModelControlTpl<Scalar>::CostModelControlTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                                                 const std::size_t& nu)
    : Base(state, nu, nu), uref_(VectorXs::Zero(nu)) {}

template <typename Scalar>
CostModelControlTpl<Scalar>::~CostModelControlTpl() {}

template <typename Scalar>
void CostModelControlTpl<Scalar>::calc(const boost::shared_ptr<CostDataAbstract>& data,
                                       const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>& u) {
  if (nu_ == 0) {
    throw_pretty("Invalid argument: "
                 << "it seems to be an autonomous system, if so, don't add this cost function");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }

  data->r = u - uref_;
  activation_->calc(data->activation, data->r);
  data->cost = data->activation->a_value;
}

template <typename Scalar>
void CostModelControlTpl<Scalar>::calcDiff(const boost::shared_ptr<CostDataAbstract>& data,
                                           const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>& u) {
  if (nu_ == 0) {
    throw_pretty("Invalid argument: "
                 << "it seems to be an autonomous system, if so, don't add this cost function");
  }
  if (static_cast<std::size_t>(u.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }

  activation_->calcDiff(data->activation, data->r);
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
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be VectorXs)");
  }
}

template <typename Scalar>
void CostModelControlTpl<Scalar>::get_referenceImpl(const std::type_info& ti, void* pv) const {
  if (ti == typeid(VectorXs)) {
    VectorXs& tmp = *static_cast<VectorXs*>(pv);
    tmp.resize(nu_);
    Eigen::Map<VectorXs> ref_map(static_cast<VectorXs*>(pv)->data(), nu_);
    for (std::size_t i = 0; i < nu_; ++i) {
      ref_map[i] = uref_[i];
    }
  } else {
    throw_pretty("Invalid argument: incorrect type (it should be VectorXs)");
  }
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& CostModelControlTpl<Scalar>::get_uref() const {
  return uref_;
}

template <typename Scalar>
void CostModelControlTpl<Scalar>::set_uref(const VectorXs& uref_in) {
  if (static_cast<std::size_t>(uref_in.size()) != nu_) {
    throw_pretty("Invalid argument: "
                 << "uref has wrong dimension (it should be " + std::to_string(nu_) + ")");
  }
  uref_ = uref_in;
}

}  // namespace crocoddyl
