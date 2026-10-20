///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/actuation-parameters.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelActuationParametersTpl<Scalar>::
    ResidualModelActuationParametersTpl(
        std::shared_ptr<typename Base::StateAbstract> state,
        const VectorXs& gamma_ref, const std::size_t nu, const std::size_t np,
        const std::string& param_name)
    : Base(state, static_cast<std::size_t>(gamma_ref.size()), nu,
           /*q_dependent=*/false, /*v_dependent=*/false,
           /*u_dependent=*/false, np),
      gamma_ref_(gamma_ref),
      param_name_(param_name) {}

template <typename Scalar>
void ResidualModelActuationParametersTpl<Scalar>::calc(
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
        "Invalid argument: shared data must be ParameterDataManagerTpl");
  }
  internal::ParameterDataManagerAccessTpl<Scalar>::getActiveOffset(*d,
                                                                   param_name_);
  auto it = d->dynamics_params.find(param_name_);
  if (it == d->dynamics_params.end()) {
    throw_pretty("Invalid argument: dynamics_params does not contain '" +
                 param_name_ + "'");
  }
  ActuationMultibodyParamsData* actuation =
      dynamic_cast<ActuationMultibodyParamsData*>(it->second.get());
  if (actuation == nullptr) {
    throw_pretty("Invalid argument: '" + param_name_ +
                 "' is not an ActuationMultibodyParamsData");
  }
  data->r = actuation->gamma - gamma_ref_;
}

template <typename Scalar>
void ResidualModelActuationParametersTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  calc(data, x, unone_);
}

template <typename Scalar>
void ResidualModelActuationParametersTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  ResidualDataActuationParameters* d =
      static_cast<ResidualDataActuationParameters*>(data.get());
  ParameterDataManager* pm = dynamic_cast<ParameterDataManager*>(data->shared);
  if (pm == nullptr) {
    DataCollectorParamsTpl<Scalar>* collector =
        dynamic_cast<DataCollectorParamsTpl<Scalar>*>(data->shared);
    pm = collector != nullptr ? collector->parameter_data : nullptr;
  }
  if (pm == nullptr) {
    throw_pretty(
        "Invalid argument: shared data must be ParameterDataManagerTpl");
  }
  d->np_offset =
      internal::ParameterDataManagerAccessTpl<Scalar>::getActiveOffset(
          *pm, param_name_);
  auto it = pm->dynamics_params.find(param_name_);
  if (it == pm->dynamics_params.end()) {
    throw_pretty("Invalid argument: dynamics_params does not contain '" +
                 param_name_ + "'");
  }
  ActuationMultibodyParamsData* actuation =
      dynamic_cast<ActuationMultibodyParamsData*>(it->second.get());
  if (actuation == nullptr) {
    throw_pretty("Invalid argument: '" + param_name_ +
                 "' is not an ActuationMultibodyParamsData");
  }
  data->Rp.setZero();
  data->Rp.block(0, d->np_offset, nr_, nr_) = actuation->dgamma_dp;
}

template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar>>
ResidualModelActuationParametersTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  ParameterDataManager* d = dynamic_cast<ParameterDataManager*>(data);
  if (d == nullptr) {
    DataCollectorParamsTpl<Scalar>* collector =
        dynamic_cast<DataCollectorParamsTpl<Scalar>*>(data);
    d = collector != nullptr ? collector->parameter_data : nullptr;
  }
  if (d == nullptr) {
    throw_pretty(
        "Invalid argument: shared data must be ParameterDataManagerTpl");
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
  ActuationMultibodyParamsData* actuation =
      dynamic_cast<ActuationMultibodyParamsData*>(it->second.get());
  if (actuation == nullptr) {
    throw_pretty("Invalid argument: '" + param_name_ +
                 "' is not an ActuationMultibodyParamsData");
  }
  if (static_cast<std::size_t>(actuation->np_dynamics) != nr_) {
    throw_pretty("Invalid argument: np_dynamics of '" + param_name_ + "' (" +
                 std::to_string(actuation->np_dynamics) +
                 ") does not match nr (" + std::to_string(nr_) + ")");
  }
  const std::size_t np_offset =
      internal::ParameterDataManagerAccessTpl<Scalar>::getActiveOffset(
          *d, param_name_);

  std::shared_ptr<ResidualDataActuationParameters> rd =
      std::allocate_shared<ResidualDataActuationParameters>(
          Eigen::aligned_allocator<ResidualDataActuationParameters>(), this,
          data);
  rd->np_offset = np_offset;
  return rd;
}

template <typename Scalar>
template <typename NewScalar>
ResidualModelActuationParametersTpl<NewScalar>
ResidualModelActuationParametersTpl<Scalar>::cast() const {
  typedef ResidualModelActuationParametersTpl<NewScalar> ReturnType;
  ReturnType ret(state_->template cast<NewScalar>(),
                 gamma_ref_.template cast<NewScalar>(), nu_, np_, param_name_);
  return ret;
}

template <typename Scalar>
void ResidualModelActuationParametersTpl<Scalar>::print(
    std::ostream& os) const {
  os << "ResidualModelActuationParameters {param=" << param_name_
     << ", nr=" << nr_ << ", np=" << np_ << "}";
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
ResidualModelActuationParametersTpl<Scalar>::get_reference() const {
  return gamma_ref_;
}

template <typename Scalar>
void ResidualModelActuationParametersTpl<Scalar>::set_reference(
    const VectorXs& reference) {
  if (static_cast<std::size_t>(reference.size()) != nr_) {
    throw_pretty(
        "Invalid argument: the actuation-parameter reference has "
        "wrong dimension ("
        << reference.size() << " provided - it should be "
        << std::to_string(nr_) << ")");
  }
  gamma_ref_ = reference;
}

template <typename Scalar>
const std::string& ResidualModelActuationParametersTpl<Scalar>::get_param_name()
    const {
  return param_name_;
}

}  // namespace crocoddyl
