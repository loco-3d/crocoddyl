///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/residuals/symmetry-parameters.hpp"

namespace crocoddyl {

template <typename Scalar>
ResidualModelSymmetryParametersTpl<Scalar>::ResidualModelSymmetryParametersTpl(
    std::shared_ptr<typename Base::StateAbstract> state, const MatrixXs& S,
    const std::size_t nu, const std::size_t np)
    : ResidualModelSymmetryParametersTpl(state, S, nu, np, std::string()) {}

template <typename Scalar>
ResidualModelSymmetryParametersTpl<Scalar>::ResidualModelSymmetryParametersTpl(
    std::shared_ptr<typename Base::StateAbstract> state, const MatrixXs& S,
    const std::size_t nu, const std::size_t np, const std::string& param_name)
    : Base(state, static_cast<std::size_t>(S.rows()), nu,
           /*q_dependent=*/false, /*v_dependent=*/false,
           /*u_dependent=*/false, np),
      S_(S),
      param_name_(param_name) {
  if (S.rows() != S.cols()) {
    throw_pretty("Invalid argument: symmetry matrix S must be square ("
                 << S.rows() << "x" << S.cols() << " provided)");
  }
  if (param_name_.empty() && static_cast<std::size_t>(S.rows()) != np_) {
    throw_pretty("Invalid argument: generic symmetry matrix S must have size "
                 << np_ << "x" << np_ << " (" << S.rows() << "x" << S.cols()
                 << " provided)");
  }
}

template <typename Scalar>
void ResidualModelSymmetryParametersTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  if (param_name_.empty()) {
    DataCollectorParams* params =
        dynamic_cast<DataCollectorParams*>(data->shared);
    if (params == nullptr || params->params == nullptr) {
      throw_pretty(
          "Invalid argument: shared data must provide DataCollectorParamsTpl");
    }
    if (static_cast<std::size_t>(params->params->np) != np_ ||
        static_cast<std::size_t>(params->params->p.size()) != np_) {
      throw_pretty(
          "Invalid argument: aggregate parameter dimension does not match np");
    }
    data->r.noalias() = S_ * params->params->p;
    return;
  }
  ResidualDataSymmetryParameters* d =
      static_cast<ResidualDataSymmetryParameters*>(data.get());
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
  if (static_cast<std::size_t>(pm->params->np) != np_ ||
      static_cast<std::size_t>(pm->params->p.size()) != np_) {
    throw_pretty(
        "Invalid argument: aggregate parameter dimension does not match np");
  }
  internal::ParameterDataManagerAccessTpl<Scalar>::getActiveOffset(*pm,
                                                                   param_name_);
  if (d->actuation_data != nullptr) {
    data->r.noalias() = S_ * d->actuation_data->gamma;
  } else if (d->inertial_data != nullptr) {
    const std::size_t nbodies = d->inertial_data->psi.size();
    data->r.setZero();
    for (std::size_t i = 0; i < nbodies; ++i) {
      data->r.noalias() += S_.middleCols(10 * i, 10) * d->inertial_data->psi[i];
    }
  } else {
    throw_pretty("Invalid argument: named symmetry data is not initialized");
  }
}

template <typename Scalar>
void ResidualModelSymmetryParametersTpl<Scalar>::calc(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  calc(data, x, unone_);
}

template <typename Scalar>
void ResidualModelSymmetryParametersTpl<Scalar>::calcDiff(
    const std::shared_ptr<ResidualDataAbstract>& data,
    const Eigen::Ref<const VectorXs>&, const Eigen::Ref<const VectorXs>&) {
  if (param_name_.empty()) {
    return;
  }
  ResidualDataSymmetryParameters* d =
      static_cast<ResidualDataSymmetryParameters*>(data.get());
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
  if (static_cast<std::size_t>(pm->params->np) != np_ ||
      static_cast<std::size_t>(pm->params->p.size()) != np_) {
    throw_pretty(
        "Invalid argument: aggregate parameter dimension does not match np");
  }
  d->np_offset =
      internal::ParameterDataManagerAccessTpl<Scalar>::getActiveOffset(
          *pm, param_name_);
  data->Rp.setZero();
  if (d->actuation_data != nullptr) {
    data->Rp.block(0, d->np_offset, nr_, nr_).noalias() =
        S_ * d->actuation_data->dgamma_dp;
  } else if (d->inertial_data != nullptr) {
    const std::size_t nbodies = d->inertial_data->dpsi_dp.size();
    for (std::size_t i = 0; i < nbodies; ++i) {
      data->Rp.block(0, d->np_offset + 10 * i, nr_, 10).noalias() =
          S_.middleCols(10 * i, 10) * d->inertial_data->dpsi_dp[i];
    }
  } else {
    throw_pretty("Invalid argument: named symmetry data is not initialized");
  }
}

template <typename Scalar>
std::shared_ptr<ResidualDataAbstractTpl<Scalar>>
ResidualModelSymmetryParametersTpl<Scalar>::createData(
    DataCollectorAbstract* const data) {
  DataCollectorParams* params = dynamic_cast<DataCollectorParams*>(data);
  if (params == nullptr || params->params == nullptr) {
    throw_pretty(
        "Invalid argument: shared data must provide DataCollectorParamsTpl");
  }
  if (static_cast<std::size_t>(params->params->np) != np_) {
    throw_pretty("Invalid argument: total np of the parameter payload (" +
                 std::to_string(params->params->np) + ") does not match np (" +
                 std::to_string(np_) + ")");
  }
  if (static_cast<std::size_t>(params->params->p.size()) != np_) {
    throw_pretty("Invalid argument: aggregate p has wrong dimension");
  }

  std::shared_ptr<ResidualDataSymmetryParameters> rd =
      std::allocate_shared<ResidualDataSymmetryParameters>(
          Eigen::aligned_allocator<ResidualDataSymmetryParameters>(), this,
          data);
  if (param_name_.empty()) {
    rd->Rp = S_;
    return rd;
  }
  ParameterDataManager* d = dynamic_cast<ParameterDataManager*>(data);
  if (d == nullptr) {
    d = params->parameter_data;
  }
  if (d == nullptr) {
    throw_pretty(
        "Invalid argument: shared data must provide ParameterDataManagerTpl");
  }
  if (d->params == nullptr) {
    throw_pretty("Invalid argument: aggregate parameter data is null");
  }
  if (static_cast<std::size_t>(d->params->np) != np_ ||
      static_cast<std::size_t>(d->params->p.size()) != np_) {
    throw_pretty(
        "Invalid argument: aggregate parameter dimension does not match np");
  }
  auto it = d->dynamics_params.find(param_name_);
  if (it == d->dynamics_params.end()) {
    throw_pretty("Invalid argument: dynamics_params does not contain '" +
                 param_name_ + "'");
  }

  // Detect param type and validate dimension
  ActuationMultibodyParamsData* actuation =
      dynamic_cast<ActuationMultibodyParamsData*>(it->second.get());
  MultibodyInertialParamsData* inertial =
      dynamic_cast<MultibodyInertialParamsData*>(it->second.get());

  if (actuation == nullptr && inertial == nullptr) {
    throw_pretty("Invalid argument: '" + param_name_ +
                 "' is neither an ActuationMultibodyParamsData nor a "
                 "MultibodyInertialParamsData");
  }

  const std::size_t np_local =
      (actuation != nullptr) ? actuation->np_dynamics : inertial->np;
  if (np_local != nr_) {
    throw_pretty("Invalid argument: local parameter dimension of '" +
                 param_name_ + "' (" + std::to_string(np_local) +
                 ") does not match S side length nr (" + std::to_string(nr_) +
                 ")");
  }

  const std::size_t np_offset =
      internal::ParameterDataManagerAccessTpl<Scalar>::getActiveOffset(
          *d, param_name_);

  rd->np_offset = np_offset;
  rd->actuation_data = actuation;
  rd->inertial_data = inertial;
  return rd;
}

template <typename Scalar>
template <typename NewScalar>
ResidualModelSymmetryParametersTpl<NewScalar>
ResidualModelSymmetryParametersTpl<Scalar>::cast() const {
  typedef ResidualModelSymmetryParametersTpl<NewScalar> ReturnType;
  ReturnType ret(state_->template cast<NewScalar>(),
                 S_.template cast<NewScalar>(), nu_, np_, param_name_);
  return ret;
}

template <typename Scalar>
void ResidualModelSymmetryParametersTpl<Scalar>::print(std::ostream& os) const {
  os << "ResidualModelSymmetryParameters {param=" << param_name_
     << ", nr=" << nr_ << ", np=" << np_ << "}";
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs&
ResidualModelSymmetryParametersTpl<Scalar>::get_S() const {
  return S_;
}

template <typename Scalar>
void ResidualModelSymmetryParametersTpl<Scalar>::set_S(const MatrixXs& S) {
  if (S.rows() != S.cols()) {
    throw_pretty("Invalid argument: symmetry matrix S must be square ("
                 << S.rows() << "x" << S.cols() << " provided)");
  }
  if (static_cast<std::size_t>(S.rows()) != nr_) {
    throw_pretty("Invalid argument: symmetry matrix S has wrong size ("
                 << S.rows() << "x" << S.cols() << " provided - it should be "
                 << nr_ << "x" << nr_ << ")");
  }
  S_ = S;
}

template <typename Scalar>
const std::string& ResidualModelSymmetryParametersTpl<Scalar>::get_param_name()
    const {
  return param_name_;
}

}  // namespace crocoddyl
