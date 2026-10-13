///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/inertial-parameters.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelInertialParametersTpl<Scalar>::ResidualModelInertialParametersTpl(
    std::shared_ptr<typename Base::StateAbstract> state,
    const VectorXs& psi_ref, const std::size_t nu,
    const std::string& param_name)
    : ResidualModelInertialParametersTpl(
          state, psi_ref, nu, static_cast<std::size_t>(psi_ref.size()),
          param_name) {}

template <typename Scalar>
void ResidualModelInertialParametersTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  ParameterDataManager* d = dynamic_cast<ParameterDataManager*>(data->shared);
  if (d == nullptr) {
    DataCollectorParamsTpl<Scalar>* collector =
        dynamic_cast<DataCollectorParamsTpl<Scalar>*>(data->shared);
    d = collector != nullptr ? collector->parameter_data : nullptr;
  }
  if (d == nullptr) {
    throw_pretty(
        "Invalid argument: shared data must provide ParameterDataManagerTpl");
  }
  internal::ParameterDataManagerAccessTpl<Scalar>::getActiveOffset(*d,
                                                                   param_name_);
  auto it = d->dynamics_params.find(param_name_);
  if (it == d->dynamics_params.end()) {
    throw_pretty("Invalid argument: dynamics_params does not contain '" +
                 param_name_ + "'");
  }
  MultibodyInertialParamsData* inertial =
      dynamic_cast<MultibodyInertialParamsData*>(it->second.get());
  if (inertial == nullptr) {
    throw_pretty("Invalid argument: '" + param_name_ +
                 "' is not a MultibodyInertialParamsData");
  }
  const std::size_t nbodies = inertial->psi.size();
  for (std::size_t i = 0; i < nbodies; ++i) {
    data->r.segment(10 * i, 10) =
        inertial->psi[i] - psi_ref_.segment(10 * i, 10);
  }
}

template <typename Scalar>
void ResidualModelInertialParametersTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  calc(data, x, unone_);
}

template <typename Scalar>
void ResidualModelInertialParametersTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  ResidualDataInertialParameters* rd =
      static_cast<ResidualDataInertialParameters*>(data.get());
  ParameterDataManager* d = dynamic_cast<ParameterDataManager*>(data->shared);
  if (d == nullptr) {
    DataCollectorParamsTpl<Scalar>* collector =
        dynamic_cast<DataCollectorParamsTpl<Scalar>*>(data->shared);
    d = collector != nullptr ? collector->parameter_data : nullptr;
  }
  if (d == nullptr) {
    throw_pretty(
        "Invalid argument: shared data must provide ParameterDataManagerTpl");
  }
  rd->np_offset =
      internal::ParameterDataManagerAccessTpl<Scalar>::getActiveOffset(
          *d, param_name_);
  auto it = d->dynamics_params.find(param_name_);
  if (it == d->dynamics_params.end()) {
    throw_pretty("Invalid argument: dynamics_params does not contain '" +
                 param_name_ + "'");
  }
  MultibodyInertialParamsData* inertial =
      dynamic_cast<MultibodyInertialParamsData*>(it->second.get());
  if (inertial == nullptr) {
    throw_pretty("Invalid argument: '" + param_name_ +
                 "' is not a MultibodyInertialParamsData");
  }
  const std::size_t nbodies = inertial->dpsi_dp.size();
  data->Rp.setZero();
  for (std::size_t i = 0; i < nbodies; ++i) {
    data->Rp.block(10 * i, rd->np_offset + 10 * i, 10, 10) =
        inertial->dpsi_dp[i];
  }
}

template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar>>
ResidualModelInertialParametersTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  ParameterDataManager* d = dynamic_cast<ParameterDataManager*>(data);
  if (d == nullptr) {
    DataCollectorParamsTpl<Scalar>* collector =
        dynamic_cast<DataCollectorParamsTpl<Scalar>*>(data);
    d = collector != nullptr ? collector->parameter_data : nullptr;
  }
  if (d == nullptr) {
    throw_pretty(
        "Invalid argument: shared data must provide ParameterDataManagerTpl");
  }
  if (d->params == nullptr) {
    throw_pretty("Invalid argument: aggregate parameter data is null");
  }
  if (static_cast<std::size_t>(d->params->np) != np_) {
    throw_pretty("Invalid argument: total np of the ParameterDataManager (" +
                 std::to_string(d->params->np) + ") does not match np (" +
                 std::to_string(np_) + ")");
  }
  auto it = d->dynamics_params.find(param_name_);
  if (it == d->dynamics_params.end()) {
    throw_pretty("Invalid argument: dynamics_params does not contain '" +
                 param_name_ + "'");
  }
  MultibodyInertialParamsData* inertial =
      dynamic_cast<MultibodyInertialParamsData*>(it->second.get());
  if (inertial == nullptr) {
    throw_pretty("Invalid argument: '" + param_name_ +
                 "' is not a MultibodyInertialParamsData");
  }
  if (inertial->np != nr_) {
    throw_pretty("Invalid argument: np of '" + param_name_ + "' (" +
                 std::to_string(inertial->np) + ") does not match nr (" +
                 std::to_string(nr_) + ")");
  }
  const std::size_t np_offset =
      internal::ParameterDataManagerAccessTpl<Scalar>::getActiveOffset(
          *d, param_name_);
  std::shared_ptr<ResidualDataInertialParameters> rd =
      std::allocate_shared<ResidualDataInertialParameters>(
          Eigen::aligned_allocator<ResidualDataInertialParameters>(), this,
          data);
  rd->np_offset = np_offset;
  return rd;
}

template <typename Scalar>
template <typename NewScalar>
ResidualModelInertialParametersTpl<NewScalar>
ResidualModelInertialParametersTpl<Scalar>::cast() const {
  typedef ResidualModelInertialParametersTpl<NewScalar> ReturnType;
  ReturnType ret(state_->template cast<NewScalar>(),
                 psi_ref_.template cast<NewScalar>(), nu_, np_, param_name_);
  return ret;
}

template <typename Scalar>
void ResidualModelInertialParametersTpl<Scalar>::print(std::ostream& os) const {
  os << "ResidualModelInertialParameters {param=" << param_name_
     << ", nr=" << nr_ << "}";
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
ResidualModelInertialParametersTpl<Scalar>::get_reference() const {
  return psi_ref_;
}

template <typename Scalar>
void ResidualModelInertialParametersTpl<Scalar>::set_reference(
    const VectorXs& reference) {
  if (static_cast<std::size_t>(reference.size()) != nr_) {
    throw_pretty(
        "Invalid argument: the inertial-parameter reference has "
        "wrong dimension ("
        << reference.size() << " provided - it should be "
        << std::to_string(nr_) << ")");
  }
  psi_ref_ = reference;
}

template <typename Scalar>
const std::string& ResidualModelInertialParametersTpl<Scalar>::get_param_name()
    const {
  return param_name_;
}

}  // namespace crocoddyl
