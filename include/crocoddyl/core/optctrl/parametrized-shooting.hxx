///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ParametrizedShootingProblemTpl<Scalar>::ParametrizedShootingProblemTpl(
    const VectorXs& x0,
    const std::vector<std::shared_ptr<ActionModelAbstract> >& running_models,
    std::shared_ptr<ActionModelAbstract> terminal_model,
    std::shared_ptr<ParameterManager> params)
    : ParametrizedShootingProblemTpl(
          x0,
          std::vector<std::vector<std::shared_ptr<ActionModelAbstract> > >{
              running_models},
          terminal_model,
          std::vector<std::shared_ptr<ParameterManager> >{params}) {}

template <typename Scalar>
ParametrizedShootingProblemTpl<Scalar>::ParametrizedShootingProblemTpl(
    const VectorXs& x0,
    const std::vector<std::shared_ptr<ActionModelAbstract> >& running_models,
    std::shared_ptr<ActionModelAbstract> terminal_model,
    std::shared_ptr<ParameterManager> params,
    std::shared_ptr<ConstraintModelManager> parameter_constraints)
    : ParametrizedShootingProblemTpl(
          x0,
          std::vector<std::vector<std::shared_ptr<ActionModelAbstract> > >{
              running_models},
          terminal_model,
          std::vector<std::shared_ptr<ParameterManager> >{params},
          std::vector<std::shared_ptr<ConstraintModelManager> >{
              parameter_constraints}) {}

template <typename Scalar>
ParametrizedShootingProblemTpl<Scalar>::ParametrizedShootingProblemTpl(
    const VectorXs& x0,
    const std::vector<std::vector<std::shared_ptr<ActionModelAbstract> > >&
        model_phases,
    std::shared_ptr<ActionModelAbstract> terminal_model,
    const std::vector<std::shared_ptr<ParameterManager> >& params)
    : ParametrizedShootingProblemTpl(
          x0, model_phases, terminal_model, params,
          std::vector<std::shared_ptr<ConstraintModelManager> >()) {}

template <typename Scalar>
ParametrizedShootingProblemTpl<Scalar>::ParametrizedShootingProblemTpl(
    const VectorXs& x0,
    const std::vector<std::vector<std::shared_ptr<ActionModelAbstract> > >&
        model_phases,
    std::shared_ptr<ActionModelAbstract> terminal_model,
    const std::vector<std::shared_ptr<ParameterManager> >& params,
    const std::vector<std::shared_ptr<ConstraintModelManager> >&
        parameter_constraints)
    : Base(x0, flattenModelPhases(model_phases),
           checkedTerminalModel(terminal_model)),
      n_phases_(model_phases.size()),
      params_(params),
      parameter_constraints_(parameter_constraints) {
  if (!parameter_constraints_.empty() &&
      parameter_constraints_.size() != n_phases_) {
    throw_pretty(
        "Invalid argument: parameter_constraints must be empty or have one "
        "entry per phase (got "
        << parameter_constraints_.size() << " constraints for " << n_phases_
        << " phases)");
  }
  init(model_phases, parameter_constraints_);
}

template <typename Scalar>
std::shared_ptr<
    typename ParametrizedShootingProblemTpl<Scalar>::ActionModelAbstract>
ParametrizedShootingProblemTpl<Scalar>::checkedTerminalModel(
    std::shared_ptr<ActionModelAbstract> terminal_model) {
  if (terminal_model == nullptr) {
    throw_pretty("Invalid argument: terminal_model is null");
  }
  return terminal_model;
}

template <typename Scalar>
std::vector<std::shared_ptr<
    typename ParametrizedShootingProblemTpl<Scalar>::ActionModelAbstract> >
ParametrizedShootingProblemTpl<Scalar>::flattenModelPhases(
    const std::vector<std::vector<std::shared_ptr<ActionModelAbstract> > >&
        model_phases) {
  if (model_phases.empty()) {
    throw_pretty("Invalid argument: model_phases is empty");
  }

  std::vector<std::shared_ptr<ActionModelAbstract> > models;
  for (std::size_t i = 0; i < model_phases.size(); ++i) {
    if (model_phases[i].empty()) {
      throw_pretty("Invalid argument: phase " << i << " has no models");
    }
    for (std::size_t j = 0; j < model_phases[i].size(); ++j) {
      if (model_phases[i][j] == nullptr) {
        throw_pretty("Invalid argument: model in phase " << i << ", node " << j
                                                         << " is null");
      }
      models.push_back(model_phases[i][j]);
    }
  }
  return models;
}

template <typename Scalar>
void ParametrizedShootingProblemTpl<Scalar>::init(
    const std::vector<std::vector<std::shared_ptr<ActionModelAbstract> > >&
        model_phases,
    const std::vector<std::shared_ptr<ConstraintModelManager> >&
        parameter_constraints) {
  if (params_.size() != n_phases_) {
    throw_pretty("Invalid argument: params must have one entry per phase (got "
                 << params_.size() << " params for " << n_phases_
                 << " phases)");
  }

  params_data_.reserve(n_phases_);
  parameter_constraints_data_.resize(n_phases_);
  phase_start_.reserve(n_phases_);
  phase_end_.reserve(n_phases_);

  std::size_t t = 0;
  for (std::size_t i = 0; i < n_phases_; ++i) {
    const std::shared_ptr<ParameterManager>& params = params_[i];
    if (params == nullptr) {
      throw_pretty("Invalid argument: params[" << i << "] is null");
    }
    if (params->get_state() == nullptr ||
        params->get_state()->get_nx() != this->nx_ ||
        params->get_state()->get_ndx() != this->ndx_) {
      throw_pretty("Invalid argument: params["
                   << i << "] has an incompatible state");
    }

    params_data_.push_back(params->createData());
    phase_start_.push_back(t);
    for (std::size_t j = 0; j < model_phases[i].size(); ++j) {
      const std::shared_ptr<ActionModelAbstract>& model = model_phases[i][j];
      if (model->get_np() != params->get_np()) {
        throw_pretty("Invalid argument: model in phase "
                     << i << ", node " << j << " has np=" << model->get_np()
                     << " but params[" << i << "] has np=" << params->get_np());
      }
      this->running_datas_[t] = model->createData(params_data_[i]);
      model->set_params(this->running_datas_[t], params);
      ++t;
    }
    if (!parameter_constraints.empty() && parameter_constraints[i] != nullptr) {
      const std::shared_ptr<ConstraintModelManager>& constraints =
          parameter_constraints[i];
      if (constraints->get_state() == nullptr ||
          constraints->get_state()->get_nx() != this->nx_ ||
          constraints->get_state()->get_ndx() != this->ndx_) {
        throw_pretty("Invalid argument: parameter_constraints["
                     << i << "] has an incompatible state");
      }
      if (constraints->get_np() != params->get_np()) {
        throw_pretty("Invalid argument: parameter_constraints["
                     << i << "] has np=" << constraints->get_np()
                     << " but params[" << i << "] has np=" << params->get_np());
      }
      if (constraints->get_nu() != model_phases[i][0]->get_nu()) {
        throw_pretty("Invalid argument: parameter_constraints["
                     << i << "] has nu=" << constraints->get_nu()
                     << " but the phase control dimension is "
                     << model_phases[i][0]->get_nu());
      }
      parameter_constraints_data_[i] =
          constraints->createData(params_data_[i].get());
    }
    phase_end_.push_back(t);
  }

  if (this->terminal_model_->get_np() != params_.back()->get_np()) {
    throw_pretty("Invalid argument: terminal_model has np="
                 << this->terminal_model_->get_np()
                 << " but the final phase params has np="
                 << params_.back()->get_np());
  }
  this->terminal_data_ = this->terminal_model_->createData(params_data_.back());
  this->terminal_model_->set_params(this->terminal_data_, params_.back());
}

template <typename Scalar>
void ParametrizedShootingProblemTpl<Scalar>::update_p(
    const Eigen::Ref<const VectorXs>& p, const std::size_t phase_idx) {
  if (phase_idx >= n_phases_) {
    throw_pretty("Invalid argument: phase_idx " << phase_idx << " >= n_phases "
                                                << n_phases_);
  }
  params_[phase_idx]->update(params_data_[phase_idx], p);
}

template <typename Scalar>
std::size_t ParametrizedShootingProblemTpl<Scalar>::get_n_phases() const {
  return n_phases_;
}

template <typename Scalar>
const std::vector<std::shared_ptr<
    typename ParametrizedShootingProblemTpl<Scalar>::ParameterManager> >&
ParametrizedShootingProblemTpl<Scalar>::get_params() const {
  return params_;
}

template <typename Scalar>
const std::vector<std::shared_ptr<
    typename ParametrizedShootingProblemTpl<Scalar>::ParameterDataManager> >&
ParametrizedShootingProblemTpl<Scalar>::get_params_data() const {
  return params_data_;
}

template <typename Scalar>
const std::vector<std::size_t>&
ParametrizedShootingProblemTpl<Scalar>::get_phase_idxs() const {
  return phase_start_;
}

template <typename Scalar>
const std::vector<std::size_t>&
ParametrizedShootingProblemTpl<Scalar>::get_phase_edxs() const {
  return phase_end_;
}

template <typename Scalar>
std::vector<std::shared_ptr<
    typename ParametrizedShootingProblemTpl<Scalar>::ActionModelAbstract> >
ParametrizedShootingProblemTpl<Scalar>::get_running_phase_models(
    const std::size_t phase_idx) const {
  if (phase_idx >= n_phases_) {
    throw_pretty("Invalid argument: phase_idx " << phase_idx << " >= n_phases "
                                                << n_phases_);
  }
  return std::vector<std::shared_ptr<ActionModelAbstract> >(
      this->running_models_.begin() + phase_start_[phase_idx],
      this->running_models_.begin() + phase_end_[phase_idx]);
}

template <typename Scalar>
std::vector<std::shared_ptr<
    typename ParametrizedShootingProblemTpl<Scalar>::ActionDataAbstract> >
ParametrizedShootingProblemTpl<Scalar>::get_running_phase_datas(
    const std::size_t phase_idx) const {
  if (phase_idx >= n_phases_) {
    throw_pretty("Invalid argument: phase_idx " << phase_idx << " >= n_phases "
                                                << n_phases_);
  }
  return std::vector<std::shared_ptr<ActionDataAbstract> >(
      this->running_datas_.begin() + phase_start_[phase_idx],
      this->running_datas_.begin() + phase_end_[phase_idx]);
}

template <typename Scalar>
const std::vector<std::shared_ptr<
    typename ParametrizedShootingProblemTpl<Scalar>::ConstraintModelManager> >&
ParametrizedShootingProblemTpl<Scalar>::get_parameter_constraints_models()
    const {
  return parameter_constraints_;
}

template <typename Scalar>
const std::vector<std::shared_ptr<
    typename ParametrizedShootingProblemTpl<Scalar>::ConstraintDataManager> >&
ParametrizedShootingProblemTpl<Scalar>::get_parameter_constraints_datas()
    const {
  return parameter_constraints_data_;
}

template <typename Scalar>
bool ParametrizedShootingProblemTpl<Scalar>::has_parameter_constraints() const {
  for (std::size_t i = 0; i < parameter_constraints_.size(); ++i) {
    const std::shared_ptr<ConstraintModelManager>& constraints =
        parameter_constraints_[i];
    if (constraints != nullptr &&
        (constraints->get_nh() != 0 || constraints->get_ng() != 0)) {
      return true;
    }
  }
  return false;
}

}  // namespace crocoddyl
