///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/total-mass.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelTotalMassTpl<Scalar>::ResidualModelTotalMassTpl(
    std::shared_ptr<typename Base::StateAbstract> state, const Scalar mass_ref,
    const std::size_t nu, const std::size_t np, const std::string& param_name)
    : Base(state, 1, nu,
           /*q_dependent=*/false, /*v_dependent=*/false,
           /*u_dependent=*/false, np),
      mass_ref_(mass_ref),
      param_name_(param_name),
      pin_model_(nullptr) {
  std::shared_ptr<StateMultibody> multibody =
      std::dynamic_pointer_cast<StateMultibody>(state_);
  if (multibody == nullptr) {
    throw_pretty("Invalid argument: state must be StateMultibodyTpl");
  }
  pin_model_ = multibody->get_pinocchio();
}

template <typename Scalar>
void ResidualModelTotalMassTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  ParameterDataManager* d = dynamic_cast<ParameterDataManager*>(data->shared);
  if (d == nullptr) {
    DataCollectorParams* collector =
        dynamic_cast<DataCollectorParams*>(data->shared);
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
  internal::ParameterDataManagerAccessTpl<Scalar>::getActiveOffset(*d,
                                                                   param_name_);
  data->r[0] = pinocchio::computeTotalMass(*pin_model_) - mass_ref_;
}

template <typename Scalar>
void ResidualModelTotalMassTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  calc(data, x, unone_);
}

template <typename Scalar>
void ResidualModelTotalMassTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  ResidualDataTotalMass* d = static_cast<ResidualDataTotalMass*>(data.get());
  ParameterDataManager* pm = dynamic_cast<ParameterDataManager*>(data->shared);
  if (pm == nullptr) {
    DataCollectorParams* collector =
        dynamic_cast<DataCollectorParams*>(data->shared);
    pm = collector != nullptr ? collector->parameter_data : nullptr;
  }
  if (pm == nullptr) {
    throw_pretty(
        "Invalid argument: shared data must provide ParameterDataManagerTpl");
  }
  if (pm->params == nullptr) {
    throw_pretty("Invalid argument: aggregate parameter data is null");
  }
  if (static_cast<std::size_t>(pm->params->np) != np_) {
    throw_pretty("Invalid argument: total np of the ParameterDataManager (" +
                 std::to_string(pm->params->np) + ") does not match np (" +
                 std::to_string(np_) + ")");
  }
  d->np_offset =
      internal::ParameterDataManagerAccessTpl<Scalar>::getActiveOffset(
          *pm, param_name_);
  auto it = pm->dynamics_params.find(param_name_);
  if (it == pm->dynamics_params.end()) {
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
    // Row 0 of dpsi_dp[i] is d(mass_i)/d(p_local_i): shape (1, 10)
    data->Rp.block(0, d->np_offset + 10 * i, 1, 10) =
        inertial->dpsi_dp[i].row(0);
  }
}

template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar>>
ResidualModelTotalMassTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  ParameterDataManager* d = dynamic_cast<ParameterDataManager*>(data);
  if (d == nullptr) {
    DataCollectorParams* collector = dynamic_cast<DataCollectorParams*>(data);
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
  const std::size_t np_offset =
      internal::ParameterDataManagerAccessTpl<Scalar>::getActiveOffset(
          *d, param_name_);

  std::shared_ptr<ResidualDataTotalMass> rd =
      std::allocate_shared<ResidualDataTotalMass>(
          Eigen::aligned_allocator<ResidualDataTotalMass>(), this, data);
  rd->np_offset = np_offset;
  return rd;
}

template <typename Scalar>
template <typename NewScalar>
ResidualModelTotalMassTpl<NewScalar> ResidualModelTotalMassTpl<Scalar>::cast()
    const {
  typedef ResidualModelTotalMassTpl<NewScalar> ReturnType;
  ReturnType ret(state_->template cast<NewScalar>(),
                 scalar_cast<NewScalar>(mass_ref_), nu_, np_, param_name_);
  return ret;
}

template <typename Scalar>
void ResidualModelTotalMassTpl<Scalar>::print(std::ostream& os) const {
  os << "ResidualModelTotalMass {param=" << param_name_
     << ", mass_ref=" << mass_ref_ << ", np=" << np_ << "}";
}

template <typename Scalar>
Scalar ResidualModelTotalMassTpl<Scalar>::get_reference() const {
  return mass_ref_;
}

template <typename Scalar>
void ResidualModelTotalMassTpl<Scalar>::set_reference(const Scalar reference) {
  mass_ref_ = reference;
}

template <typename Scalar>
const std::string& ResidualModelTotalMassTpl<Scalar>::get_param_name() const {
  return param_name_;
}

}  // namespace crocoddyl
