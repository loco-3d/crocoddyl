///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
ObservationProblemTpl<Scalar>::ObservationProblemTpl(
    const VectorXs& x0, const std::vector<VectorXs>& tau_meas,
    const std::vector<std::vector<std::shared_ptr<ObserverModelAbstract> > >&
        model_phases,
    std::shared_ptr<ObserverModelAbstract> terminal_model,
    const std::vector<std::shared_ptr<ParameterManager> >& params)
    : ObservationProblemTpl(
          x0, tau_meas, model_phases, terminal_model, params,
          std::vector<std::shared_ptr<ConstraintModelManager> >()) {}

template <typename Scalar>
ObservationProblemTpl<Scalar>::ObservationProblemTpl(
    const VectorXs& x0, const std::vector<VectorXs>& tau_meas,
    const std::vector<std::vector<std::shared_ptr<ObserverModelAbstract> > >&
        model_phases,
    std::shared_ptr<ObserverModelAbstract> terminal_model,
    const std::vector<std::shared_ptr<ParameterManager> >& params,
    const std::vector<std::shared_ptr<ConstraintModelManager> >&
        parameter_constraints)
    : cost_(Scalar(0)),
      T_(0),
      x0_(x0),
      nx_(0),
      ndx_(0),
      is_updated_(false),
      n_phases_(model_phases.size()),
      params_(params),
      parameter_constraints_(parameter_constraints) {
  if (params_.size() != n_phases_) {
    throw_pretty("Invalid argument: params must have one entry per phase (got "
                 << params_.size() << " params for " << n_phases_
                 << " phases)");
  }
  if (!parameter_constraints_.empty() &&
      parameter_constraints_.size() != n_phases_) {
    throw_pretty(
        "Invalid argument: parameter_constraints must be empty or have one "
        "entry per phase (got "
        << parameter_constraints_.size() << " constraints for " << n_phases_
        << " phases)");
  }
  init(x0, tau_meas, model_phases, terminal_model, parameter_constraints_);
}

template <typename Scalar>
ObservationProblemTpl<Scalar>::ObservationProblemTpl(
    const VectorXs& x0, const std::vector<VectorXs>& tau_meas,
    const std::vector<std::shared_ptr<ObserverModelAbstract> >& running_models,
    std::shared_ptr<ObserverModelAbstract> terminal_model,
    std::shared_ptr<ParameterManager> params)
    : ObservationProblemTpl(
          x0, tau_meas,
          std::vector<std::vector<std::shared_ptr<ObserverModelAbstract> > >{
              running_models},
          terminal_model,
          std::vector<std::shared_ptr<ParameterManager> >{params}) {}

template <typename Scalar>
ObservationProblemTpl<Scalar>::ObservationProblemTpl(
    const VectorXs& x0, const std::vector<VectorXs>& tau_meas,
    const std::vector<std::shared_ptr<ObserverModelAbstract> >& running_models,
    std::shared_ptr<ObserverModelAbstract> terminal_model,
    std::shared_ptr<ParameterManager> params,
    std::shared_ptr<ConstraintModelManager> parameter_constraints)
    : ObservationProblemTpl(
          x0, tau_meas,
          std::vector<std::vector<std::shared_ptr<ObserverModelAbstract> > >{
              running_models},
          terminal_model,
          std::vector<std::shared_ptr<ParameterManager> >{params},
          std::vector<std::shared_ptr<ConstraintModelManager> >{
              parameter_constraints}) {}

template <typename Scalar>
void ObservationProblemTpl<Scalar>::init(
    const VectorXs& x0, const std::vector<VectorXs>& tau_meas,
    const std::vector<std::vector<std::shared_ptr<ObserverModelAbstract> > >&
        model_phases,
    std::shared_ptr<ObserverModelAbstract> terminal_model,
    const std::vector<std::shared_ptr<ConstraintModelManager> >&
        parameter_constraints) {
  if (model_phases.empty()) {
    throw_pretty("Invalid argument: model_phases is empty");
  }
  if (terminal_model == nullptr) {
    throw_pretty("Invalid argument: terminal_model is null");
  }

  for (std::size_t i = 0; i < n_phases_; ++i) {
    if (model_phases[i].empty()) {
      throw_pretty("Invalid argument: phase " << i << " has no models");
    }
    T_ += model_phases[i].size();
  }
  if (tau_meas.size() != T_) {
    throw_pretty("Invalid argument: tau_meas must have T="
                 << T_ << " entries (got " << tau_meas.size() << ")");
  }

  const std::shared_ptr<ObserverModelAbstract>& first = model_phases[0][0];
  if (first == nullptr || first->get_state() == nullptr) {
    throw_pretty("Invalid argument: model in phase 0, node 0 is null");
  }
  nx_ = first->get_state()->get_nx();
  ndx_ = first->get_state()->get_ndx();
  if (static_cast<std::size_t>(x0.size()) != nx_) {
    throw_pretty("Invalid argument: x0 has wrong dimension (it should be "
                 << nx_ << ")");
  }
  if (terminal_model->get_state() == nullptr ||
      terminal_model->get_state()->get_nx() != nx_ ||
      terminal_model->get_state()->get_ndx() != ndx_) {
    throw_pretty("Invalid argument: terminal_model has an incompatible state");
  }

  running_models_.reserve(T_);
  running_datas_.resize(T_);
  params_data_.reserve(n_phases_);
  parameter_constraints_data_.resize(n_phases_);
  phase_start_.reserve(n_phases_);
  phase_end_.reserve(n_phases_);

  std::size_t t = 0;
  for (std::size_t i = 0; i < n_phases_; ++i) {
    const std::shared_ptr<ParameterManager>& params = params_[i];
    if (params == nullptr || params->get_state() == nullptr) {
      throw_pretty("Invalid argument: params[" << i << "] is null");
    }
    if (params->get_state()->get_nx() != nx_ ||
        params->get_state()->get_ndx() != ndx_) {
      throw_pretty("Invalid argument: params["
                   << i << "] has an incompatible state");
    }

    params_data_.push_back(params->createData());
    phase_start_.push_back(t);
    for (std::size_t j = 0; j < model_phases[i].size(); ++j) {
      const std::shared_ptr<ObserverModelAbstract>& model = model_phases[i][j];
      if (model == nullptr || model->get_state() == nullptr) {
        throw_pretty("Invalid argument: model in phase " << i << ", node " << j
                                                         << " is null");
      }
      if (model->get_state()->get_nx() != nx_ ||
          model->get_state()->get_ndx() != ndx_) {
        throw_pretty("Invalid argument: model in phase "
                     << i << ", node " << j << " has an incompatible state");
      }
      if (model->get_np() != params->get_np()) {
        throw_pretty("Invalid argument: model in phase "
                     << i << ", node " << j << " has np=" << model->get_np()
                     << " but params[" << i << "] has np=" << params->get_np());
      }
      running_models_.push_back(model);
      running_datas_[t] = model->createData(params_data_[i]);
      model->set_params(running_datas_[t], params);
      model->update_tau(tau_meas[t]);
      ++t;
    }

    if (!parameter_constraints.empty() && parameter_constraints[i] != nullptr) {
      const std::shared_ptr<ConstraintModelManager>& constraints =
          parameter_constraints[i];
      if (constraints->get_state() == nullptr ||
          constraints->get_state()->get_nx() != nx_ ||
          constraints->get_state()->get_ndx() != ndx_) {
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

  if (terminal_model->get_np() != params_.back()->get_np()) {
    throw_pretty("Invalid argument: terminal_model has np="
                 << terminal_model->get_np()
                 << " but the final phase params has np="
                 << params_.back()->get_np());
  }
  terminal_model_ = terminal_model;
  terminal_data_ = terminal_model->createData(params_data_.back());
  terminal_model->set_params(terminal_data_, params_.back());
}

template <typename Scalar>
Scalar ObservationProblemTpl<Scalar>::calc(const std::vector<VectorXs>& xs,
                                           const std::vector<VectorXs>& us) {
  if (xs.size() != T_ + 1) {
    throw_pretty("Invalid argument: xs has wrong dimension (it should be "
                 << T_ + 1 << ")");
  }
  if (us.size() != T_) {
    throw_pretty("Invalid argument: us has wrong dimension (it should be "
                 << T_ << ")");
  }
  for (std::size_t t = 0; t < T_; ++t) {
    running_models_[t]->calc(running_datas_[t], xs[t], us[t]);
  }
  terminal_model_->calc(terminal_data_, xs.back());
  for (std::size_t i = 0; i < parameter_constraints_.size(); ++i) {
    const std::shared_ptr<ConstraintModelManager>& constraints =
        parameter_constraints_[i];
    const std::shared_ptr<ConstraintDataManager>& data =
        parameter_constraints_data_[i];
    if (constraints != nullptr && data != nullptr) {
      data->resize(constraints.get(), true);
      constraints->calc(data, xs[phase_start_[i]], us[phase_start_[i]]);
    }
  }

  cost_ = Scalar(0);
  for (std::size_t t = 0; t < T_; ++t) {
    cost_ += running_datas_[t]->cost;
  }
  cost_ += terminal_data_->cost;
  return cost_;
}

template <typename Scalar>
Scalar ObservationProblemTpl<Scalar>::calcDiff(
    const std::vector<VectorXs>& xs, const std::vector<VectorXs>& us) {
  if (xs.size() != T_ + 1) {
    throw_pretty("Invalid argument: xs has wrong dimension (it should be "
                 << T_ + 1 << ")");
  }
  if (us.size() != T_) {
    throw_pretty("Invalid argument: us has wrong dimension (it should be "
                 << T_ << ")");
  }
  for (std::size_t t = 0; t < T_; ++t) {
    running_models_[t]->calcDiff(running_datas_[t], xs[t], us[t]);
  }
  terminal_model_->calcDiff(terminal_data_, xs.back());
  for (std::size_t i = 0; i < parameter_constraints_.size(); ++i) {
    const std::shared_ptr<ConstraintModelManager>& constraints =
        parameter_constraints_[i];
    const std::shared_ptr<ConstraintDataManager>& data =
        parameter_constraints_data_[i];
    if (constraints != nullptr && data != nullptr) {
      data->resize(constraints.get(), true);
      constraints->calc(data, xs[phase_start_[i]], us[phase_start_[i]]);
      constraints->calcDiff(data, xs[phase_start_[i]], us[phase_start_[i]]);
    }
  }

  cost_ = Scalar(0);
  for (std::size_t t = 0; t < T_; ++t) {
    cost_ += running_datas_[t]->cost;
  }
  cost_ += terminal_data_->cost;
  return cost_;
}

template <typename Scalar>
void ObservationProblemTpl<Scalar>::rollout(const std::vector<VectorXs>& us,
                                            std::vector<VectorXs>& xs) {
  if (xs.size() != T_ + 1) {
    throw_pretty("Invalid argument: xs has wrong dimension (it should be "
                 << T_ + 1 << ")");
  }
  if (us.size() != T_) {
    throw_pretty("Invalid argument: us has wrong dimension (it should be "
                 << T_ << ")");
  }
  xs[0] = x0_;
  for (std::size_t t = 0; t < T_; ++t) {
    running_models_[t]->calc(running_datas_[t], xs[t], us[t]);
    xs[t + 1] = running_datas_[t]->xnext;
  }
  terminal_model_->calc(terminal_data_, xs.back());
}

template <typename Scalar>
void ObservationProblemTpl<Scalar>::update_p(
    const Eigen::Ref<const VectorXs>& p, const std::size_t phase_idx) {
  if (phase_idx >= n_phases_) {
    throw_pretty("Invalid argument: phase_idx " << phase_idx << " >= n_phases "
                                                << n_phases_);
  }
  params_[phase_idx]->update(params_data_[phase_idx], p);
}

template <typename Scalar>
void ObservationProblemTpl<Scalar>::update_tau(
    const std::size_t t, const Eigen::Ref<const VectorXs>& tau_meas) {
  if (t >= T_) {
    throw_pretty("Invalid argument: t=" << t << " >= T=" << T_);
  }
  std::static_pointer_cast<ObserverModelAbstract>(running_models_[t])
      ->update_tau(tau_meas);
}

template <typename Scalar>
void ObservationProblemTpl<Scalar>::update_us(
    const std::vector<VectorXs>& tau_meas) {
  if (tau_meas.size() != T_) {
    throw_pretty("Invalid argument: tau_meas must have T="
                 << T_ << " entries (got " << tau_meas.size() << ")");
  }
  for (std::size_t t = 0; t < T_; ++t) {
    update_tau(t, tau_meas[t]);
  }
}

template <typename Scalar>
std::vector<std::shared_ptr<
    typename ObservationProblemTpl<Scalar>::ObserverModelAbstract> >
ObservationProblemTpl<Scalar>::get_running_phase_models(
    const std::size_t phase_idx) const {
  if (phase_idx >= n_phases_) {
    throw_pretty("Invalid argument: phase_idx " << phase_idx << " >= n_phases "
                                                << n_phases_);
  }
  std::vector<std::shared_ptr<ObserverModelAbstract> > models;
  models.reserve(phase_end_[phase_idx] - phase_start_[phase_idx]);
  for (std::size_t t = phase_start_[phase_idx]; t < phase_end_[phase_idx];
       ++t) {
    models.push_back(
        std::static_pointer_cast<ObserverModelAbstract>(running_models_[t]));
  }
  return models;
}

template <typename Scalar>
std::vector<std::shared_ptr<
    typename ObservationProblemTpl<Scalar>::ActionDataAbstract> >
ObservationProblemTpl<Scalar>::get_running_phase_datas(
    const std::size_t phase_idx) const {
  if (phase_idx >= n_phases_) {
    throw_pretty("Invalid argument: phase_idx " << phase_idx << " >= n_phases "
                                                << n_phases_);
  }
  return std::vector<std::shared_ptr<ActionDataAbstract> >(
      running_datas_.begin() + phase_start_[phase_idx],
      running_datas_.begin() + phase_end_[phase_idx]);
}

template <typename Scalar>
std::size_t ObservationProblemTpl<Scalar>::get_T() const {
  return T_;
}

template <typename Scalar>
const typename ObservationProblemTpl<Scalar>::VectorXs&
ObservationProblemTpl<Scalar>::get_x0() const {
  return x0_;
}

template <typename Scalar>
std::size_t ObservationProblemTpl<Scalar>::get_nx() const {
  return nx_;
}

template <typename Scalar>
std::size_t ObservationProblemTpl<Scalar>::get_ndx() const {
  return ndx_;
}

template <typename Scalar>
std::size_t ObservationProblemTpl<Scalar>::get_nthreads() const {
  return 1;
}

template <typename Scalar>
const std::vector<std::shared_ptr<
    typename ObservationProblemTpl<Scalar>::ActionModelAbstract> >&
ObservationProblemTpl<Scalar>::get_runningModels() const {
  return running_models_;
}

template <typename Scalar>
const std::shared_ptr<
    typename ObservationProblemTpl<Scalar>::ActionModelAbstract>&
ObservationProblemTpl<Scalar>::get_terminalModel() const {
  return terminal_model_;
}

template <typename Scalar>
const std::vector<std::shared_ptr<
    typename ObservationProblemTpl<Scalar>::ActionDataAbstract> >&
ObservationProblemTpl<Scalar>::get_runningDatas() const {
  return running_datas_;
}

template <typename Scalar>
const std::shared_ptr<
    typename ObservationProblemTpl<Scalar>::ActionDataAbstract>&
ObservationProblemTpl<Scalar>::get_terminalData() const {
  return terminal_data_;
}

template <typename Scalar>
bool ObservationProblemTpl<Scalar>::is_updated() {
  const bool value = is_updated_;
  is_updated_ = false;
  return value;
}

template <typename Scalar>
void ObservationProblemTpl<Scalar>::set_is_updated(const bool val) {
  is_updated_ = val;
}

template <typename Scalar>
std::size_t ObservationProblemTpl<Scalar>::get_n_phases() const {
  return n_phases_;
}

template <typename Scalar>
const std::vector<
    std::shared_ptr<typename ObservationProblemTpl<Scalar>::ParameterManager> >&
ObservationProblemTpl<Scalar>::get_params() const {
  return params_;
}

template <typename Scalar>
const std::vector<std::shared_ptr<
    typename ObservationProblemTpl<Scalar>::ParameterDataManager> >&
ObservationProblemTpl<Scalar>::get_params_data() const {
  return params_data_;
}

template <typename Scalar>
const std::vector<std::size_t>& ObservationProblemTpl<Scalar>::get_phase_idxs()
    const {
  return phase_start_;
}

template <typename Scalar>
const std::vector<std::size_t>& ObservationProblemTpl<Scalar>::get_phase_edxs()
    const {
  return phase_end_;
}

template <typename Scalar>
const std::vector<std::shared_ptr<
    typename ObservationProblemTpl<Scalar>::ConstraintModelManager> >&
ObservationProblemTpl<Scalar>::get_parameter_constraints_models() const {
  return parameter_constraints_;
}

template <typename Scalar>
const std::vector<std::shared_ptr<
    typename ObservationProblemTpl<Scalar>::ConstraintDataManager> >&
ObservationProblemTpl<Scalar>::get_parameter_constraints_datas() const {
  return parameter_constraints_data_;
}

template <typename Scalar>
bool ObservationProblemTpl<Scalar>::has_parameter_constraints() const {
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
