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
    const std::vector<std::shared_ptr<ObserverModelAbstract> >& running_models,
    std::shared_ptr<ObserverModelAbstract> terminal_model)
    : ObservationProblemTpl(
          x0, tau_meas,
          std::vector<std::vector<std::shared_ptr<ObserverModelAbstract> > >{
              running_models},
          terminal_model,
          std::vector<std::shared_ptr<ParameterPhaseModel> >()) {}

template <typename Scalar>
ObservationProblemTpl<Scalar>::ObservationProblemTpl(
    const VectorXs& x0, const std::vector<VectorXs>& tau_meas,
    const std::vector<std::vector<std::shared_ptr<ObserverModelAbstract> > >&
        model_phases,
    std::shared_ptr<ObserverModelAbstract> terminal_model,
    const std::vector<std::shared_ptr<ParameterPhaseModel> >& params_model)
    : cost_(Scalar(0)),
      T_(0),
      x0_(x0),
      nx_(0),
      ndx_(0),
      is_updated_(false),
      n_phases_(params_model.empty() ? 0 : model_phases.size()),
      params_model_(params_model) {
  if (!params_model_.empty() && params_model_.size() != model_phases.size()) {
    throw_pretty(
        "Invalid argument: paramsModel must have one entry per phase (got "
        << params_model_.size() << " parameter models for "
        << model_phases.size() << " phases)");
  }
  init(x0, tau_meas, model_phases, terminal_model);
}

template <typename Scalar>
ObservationProblemTpl<Scalar>::ObservationProblemTpl(
    const VectorXs& x0, const std::vector<VectorXs>& tau_meas,
    const std::vector<std::shared_ptr<ObserverModelAbstract> >& running_models,
    std::shared_ptr<ObserverModelAbstract> terminal_model,
    std::shared_ptr<ParameterPhaseModel> params_model)
    : ObservationProblemTpl(
          x0, tau_meas,
          std::vector<std::vector<std::shared_ptr<ObserverModelAbstract> > >{
              running_models},
          terminal_model,
          std::vector<std::shared_ptr<ParameterPhaseModel> >{params_model}) {}

template <typename Scalar>
void ObservationProblemTpl<Scalar>::init(
    const VectorXs& x0, const std::vector<VectorXs>& tau_meas,
    const std::vector<std::vector<std::shared_ptr<ObserverModelAbstract> > >&
        model_phases,
    std::shared_ptr<ObserverModelAbstract> terminal_model) {
  if (model_phases.empty()) {
    throw_pretty("Invalid argument: model_phases is empty");
  }
  if (terminal_model == nullptr) {
    throw_pretty("Invalid argument: terminal_model is null");
  }

  for (std::size_t i = 0; i < model_phases.size(); ++i) {
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
  phase_start_.reserve(n_phases_);
  phase_end_.reserve(n_phases_);

  std::size_t t = 0;
  for (std::size_t i = 0; i < model_phases.size(); ++i) {
    std::shared_ptr<ParameterPhaseModel> params_model;
    if (n_phases_ != 0) {
      params_model = params_model_[i];
      if (params_model == nullptr || params_model->get_state() == nullptr) {
        throw_pretty("Invalid argument: paramsModel[" << i << "] is null");
      }
      if (params_model->get_state()->get_nx() != nx_ ||
          params_model->get_state()->get_ndx() != ndx_) {
        throw_pretty("Invalid argument: paramsModel["
                     << i << "] has an incompatible state");
      }
      params_data_.push_back(params_model->createData());
      phase_start_.push_back(t);
    }
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
      running_models_.push_back(model);
      if (n_phases_ != 0) {
        if (model->get_np() != params_model->get_np()) {
          throw_pretty("Invalid argument: model in phase "
                       << i << ", node " << j << " has np=" << model->get_np()
                       << " but paramsModel[" << i
                       << "] has np=" << params_model->get_np());
        }
        running_datas_[t] = model->createData(params_data_[i]->params);
        model->set_params(running_datas_[t], params_model->get_params());
      } else {
        if (model->get_np() != 0) {
          throw_pretty("Invalid argument: non-parameterized model in node "
                       << t << " has np=" << model->get_np());
        }
        running_datas_[t] = model->createData();
      }
      model->update_tau(tau_meas[t]);
      ++t;
    }

    if (n_phases_ != 0 && params_model->get_constraints() != nullptr) {
      const std::shared_ptr<
          typename ParameterPhaseModel::ConstraintModelManager>& constraints =
          params_model->get_constraints();
      if (constraints->get_nu() != model_phases[i][0]->get_nu()) {
        throw_pretty("Invalid argument: paramsModel["
                     << i << "] has nu=" << constraints->get_nu()
                     << " but the phase control dimension is "
                     << model_phases[i][0]->get_nu());
      }
    }
    if (n_phases_ != 0) {
      phase_end_.push_back(t);
    }
  }

  terminal_model_ = terminal_model;
  if (n_phases_ != 0) {
    if (terminal_model->get_np() != params_model_.back()->get_np()) {
      throw_pretty("Invalid argument: terminal_model has np="
                   << terminal_model->get_np()
                   << " but the final phase paramsModel has np="
                   << params_model_.back()->get_np());
    }
    terminal_data_ = terminal_model->createData(params_data_.back()->params);
    terminal_model->set_params(terminal_data_,
                               params_model_.back()->get_params());
  } else {
    if (terminal_model->get_np() != 0) {
      throw_pretty("Invalid argument: non-parameterized terminal_model has np="
                   << terminal_model->get_np());
    }
    terminal_data_ = terminal_model->createData();
  }
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
  for (std::size_t i = 0; i < params_model_.size(); ++i) {
    params_model_[i]->calc(params_data_[i], xs[phase_start_[i]],
                           us[phase_start_[i]]);
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
  for (std::size_t i = 0; i < params_model_.size(); ++i) {
    params_model_[i]->calc(params_data_[i], xs[phase_start_[i]],
                           us[phase_start_[i]]);
    params_model_[i]->calcDiff(params_data_[i], xs[phase_start_[i]],
                               us[phase_start_[i]]);
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
  params_model_[phase_idx]->update(params_data_[phase_idx], p);
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
ObservationProblemTpl<Scalar>::get_runningPhaseModels(
    const std::size_t phase_idx) const {
  if (phase_idx >= n_phases_) {
    throw_pretty("Invalid argument: phase_idx " << phase_idx << " >= n_phases "
                                                << n_phases_);
  }
  std::vector<std::shared_ptr<ObserverModelAbstract> > phase_models;
  phase_models.reserve(phase_end_[phase_idx] - phase_start_[phase_idx]);
  for (std::size_t t = phase_start_[phase_idx]; t < phase_end_[phase_idx];
       ++t) {
    phase_models.push_back(
        std::static_pointer_cast<ObserverModelAbstract>(running_models_[t]));
  }
  return phase_models;
}

template <typename Scalar>
std::vector<std::shared_ptr<
    typename ObservationProblemTpl<Scalar>::ActionDataAbstract> >
ObservationProblemTpl<Scalar>::get_runningPhaseDatas(
    const std::size_t phase_idx) const {
  if (phase_idx >= n_phases_) {
    throw_pretty("Invalid argument: phase_idx " << phase_idx << " >= n_phases "
                                                << n_phases_);
  }
  std::vector<std::shared_ptr<ActionDataAbstract> > phase_datas;
  phase_datas.reserve(phase_end_[phase_idx] - phase_start_[phase_idx]);
  for (std::size_t t = phase_start_[phase_idx]; t < phase_end_[phase_idx];
       ++t) {
    phase_datas.push_back(running_datas_[t]);
  }
  return phase_datas;
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
const std::vector<std::shared_ptr<
    typename ObservationProblemTpl<Scalar>::ParameterPhaseModel> >&
ObservationProblemTpl<Scalar>::get_paramsModel() const {
  return params_model_;
}

template <typename Scalar>
const std::vector<std::shared_ptr<
    typename ObservationProblemTpl<Scalar>::ParameterPhaseData> >&
ObservationProblemTpl<Scalar>::get_paramsData() const {
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
bool ObservationProblemTpl<Scalar>::has_parameter_constraints() const {
  for (std::size_t i = 0; i < params_model_.size(); ++i) {
    if (params_model_[i]->has_constraints()) {
      return true;
    }
  }
  return false;
}

}  // namespace crocoddyl
