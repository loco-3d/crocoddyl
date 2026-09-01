///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, LAAS-CNRS, University of Edinburgh,
//                          University of Oxford, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/stop-watch.hpp"

namespace crocoddyl {

template <typename Scalar>
ShootingProblemTpl<Scalar>::ShootingProblemTpl(
    const VectorXs& x0,
    const std::vector<std::shared_ptr<ActionModelAbstract> >& running_models,
    std::shared_ptr<ActionModelAbstract> terminal_model)
    : cost_(Scalar(0.)),
      T_(running_models.size()),
      x0_(x0),
      terminal_model_(terminal_model),
      running_models_(running_models),
      nx_(running_models[0]->get_state()->get_nx()),
      ndx_(running_models[0]->get_state()->get_ndx()),
      nthreads_(1),
      is_updated_(false),
      n_phases_(0) {
  if (static_cast<std::size_t>(x0.size()) != nx_) {
    throw_pretty(
        "Invalid argument: " << "x0 has wrong dimension (it should be " +
                                    std::to_string(nx_) + ")");
  }
  for (std::size_t i = 1; i < T_; ++i) {
    const std::shared_ptr<ActionModelAbstract>& model = running_models_[i];
    if (model->get_state()->get_nx() != nx_) {
      throw_pretty("Invalid argument: "
                   << "nx in " << i
                   << " node is not consistent with the other nodes")
    }
    if (model->get_state()->get_ndx() != ndx_) {
      throw_pretty("Invalid argument: "
                   << "ndx in " << i
                   << " node is not consistent with the other nodes")
    }
  }
  if (terminal_model_->get_state()->get_nx() != nx_) {
    throw_pretty(
        "Invalid argument: "
        << "nx in terminal node is not consistent with the other nodes")
  }
  if (terminal_model_->get_state()->get_ndx() != ndx_) {
    throw_pretty(
        "Invalid argument: "
        << "ndx in terminal node is not consistent with the other nodes")
  }
  allocateData();

#ifdef CROCODDYL_WITH_MULTITHREADING
  if (enableMultithreading()) {
    nthreads_ = CROCODDYL_WITH_NTHREADS;
  }
#endif
}

template <typename Scalar>
ShootingProblemTpl<Scalar>::ShootingProblemTpl(
    const VectorXs& x0,
    const std::vector<std::shared_ptr<ActionModelAbstract> >& running_models,
    std::shared_ptr<ActionModelAbstract> terminal_model,
    const std::vector<std::shared_ptr<ActionDataAbstract> >& running_datas,
    std::shared_ptr<ActionDataAbstract> terminal_data)
    : cost_(Scalar(0.)),
      T_(running_models.size()),
      x0_(x0),
      terminal_model_(terminal_model),
      terminal_data_(terminal_data),
      running_models_(running_models),
      running_datas_(running_datas),
      nx_(running_models[0]->get_state()->get_nx()),
      ndx_(running_models[0]->get_state()->get_ndx()),
      nthreads_(1),
      is_updated_(false),
      n_phases_(0) {
  if (static_cast<std::size_t>(x0.size()) != nx_) {
    throw_pretty(
        "Invalid argument: " << "x0 has wrong dimension (it should be " +
                                    std::to_string(nx_) + ")");
  }
  const std::size_t Td = running_datas.size();
  if (Td != T_) {
    throw_pretty(
        "Invalid argument: "
        << "the number of running models and datas are not the same (" +
               std::to_string(T_) + " != " + std::to_string(Td) + ")")
  }
  for (std::size_t i = 0; i < T_; ++i) {
    const std::shared_ptr<ActionModelAbstract>& model = running_models_[i];
    const std::shared_ptr<ActionDataAbstract>& data = running_datas_[i];
    if (model->get_state()->get_nx() != nx_) {
      throw_pretty("Invalid argument: "
                   << "nx in " << i
                   << " node is not consistent with the other nodes")
    }
    if (model->get_state()->get_ndx() != ndx_) {
      throw_pretty("Invalid argument: "
                   << "ndx in " << i
                   << " node is not consistent with the other nodes")
    }
    if (!model->checkData(data)) {
      throw_pretty("Invalid argument: "
                   << "action data in " << i
                   << " node is not consistent with the action model")
    }
  }
  if (!terminal_model->checkData(terminal_data)) {
    throw_pretty("Invalid argument: "
                 << "terminal action data is not consistent with the terminal "
                    "action model")
  }

#ifdef CROCODDYL_WITH_MULTITHREADING
  if (enableMultithreading()) {
    nthreads_ = CROCODDYL_WITH_NTHREADS;
  }
#endif
}

template <typename Scalar>
ShootingProblemTpl<Scalar>::ShootingProblemTpl(
    const VectorXs& x0,
    const std::vector<std::shared_ptr<ActionModelAbstract> >& running_models,
    std::shared_ptr<ActionModelAbstract> terminal_model,
    std::shared_ptr<ParameterPhaseModel> params_model)
    : ShootingProblemTpl(
          x0,
          std::vector<std::vector<std::shared_ptr<ActionModelAbstract> > >{
              running_models},
          terminal_model,
          std::vector<std::shared_ptr<ParameterPhaseModel> >{params_model}) {}

template <typename Scalar>
ShootingProblemTpl<Scalar>::ShootingProblemTpl(
    const VectorXs& x0,
    const std::vector<std::vector<std::shared_ptr<ActionModelAbstract> > >&
        model_phases,
    std::shared_ptr<ActionModelAbstract> terminal_model,
    const std::vector<std::shared_ptr<ParameterPhaseModel> >& params_model)
    : ShootingProblemTpl(x0, flattenModelPhases(model_phases),
                         checkedTerminalModel(terminal_model)) {
  n_phases_ = model_phases.size();
  params_model_ = params_model;
  initParameterization(model_phases);
}

template <typename Scalar>
ShootingProblemTpl<Scalar>::ShootingProblemTpl(
    const ShootingProblemTpl<Scalar>& problem)
    : cost_(Scalar(0.)),
      T_(problem.get_T()),
      x0_(problem.get_x0()),
      terminal_model_(problem.get_terminalModel()),
      terminal_data_(problem.get_terminalData()),
      running_models_(problem.get_runningModels()),
      running_datas_(problem.get_runningDatas()),
      nx_(problem.get_nx()),
      ndx_(problem.get_ndx()),
      nthreads_(problem.nthreads_),
      is_updated_(problem.is_updated_),
      n_phases_(problem.n_phases_),
      params_model_(problem.params_model_),
      params_data_(problem.params_data_),
      phase_start_(problem.phase_start_),
      phase_end_(problem.phase_end_) {}

template <typename Scalar>
ShootingProblemTpl<Scalar>::~ShootingProblemTpl() {}

template <typename Scalar>
std::shared_ptr<typename ShootingProblemTpl<Scalar>::ActionModelAbstract>
ShootingProblemTpl<Scalar>::checkedTerminalModel(
    std::shared_ptr<ActionModelAbstract> terminal_model) {
  if (terminal_model == nullptr) {
    throw_pretty("Invalid argument: terminal_model is null");
  }
  return terminal_model;
}

template <typename Scalar>
std::vector<
    std::shared_ptr<typename ShootingProblemTpl<Scalar>::ActionModelAbstract> >
ShootingProblemTpl<Scalar>::flattenModelPhases(
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
void ShootingProblemTpl<Scalar>::initParameterization(
    const std::vector<std::vector<std::shared_ptr<ActionModelAbstract> > >&
        model_phases) {
  if (params_model_.size() != n_phases_) {
    throw_pretty(
        "Invalid argument: paramsModel must have one entry per phase (got "
        << params_model_.size() << " parameter models for " << n_phases_
        << " phases)");
  }

  params_data_.reserve(n_phases_);
  phase_start_.reserve(n_phases_);
  phase_end_.reserve(n_phases_);

  std::size_t t = 0;
  for (std::size_t i = 0; i < n_phases_; ++i) {
    const std::shared_ptr<ParameterPhaseModel>& params_model = params_model_[i];
    if (params_model == nullptr) {
      throw_pretty("Invalid argument: paramsModel[" << i << "] is null");
    }
    if (params_model->get_state() == nullptr ||
        params_model->get_state()->get_nx() != nx_ ||
        params_model->get_state()->get_ndx() != ndx_) {
      throw_pretty("Invalid argument: paramsModel["
                   << i << "] has an incompatible state");
    }

    const std::shared_ptr<typename ParameterPhaseModel::ParameterManager>&
        params = params_model->get_params();
    params_data_.push_back(params_model->createData());
    phase_start_.push_back(t);
    for (std::size_t j = 0; j < model_phases[i].size(); ++j) {
      const std::shared_ptr<ActionModelAbstract>& model = model_phases[i][j];
      if (model->get_np() != params_model->get_np()) {
        throw_pretty("Invalid argument: model in phase "
                     << i << ", node " << j << " has np=" << model->get_np()
                     << " but paramsModel[" << i
                     << "] has np=" << params_model->get_np());
      }
      running_datas_[t] = model->createData(params_data_[i]->params);
      model->set_params(running_datas_[t], params);
      ++t;
    }
    if (params_model->get_constraints() != nullptr) {
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
    phase_end_.push_back(t);
  }

  if (terminal_model_->get_np() != params_model_.back()->get_np()) {
    throw_pretty("Invalid argument: terminal_model has np="
                 << terminal_model_->get_np()
                 << " but the final phase paramsModel has np="
                 << params_model_.back()->get_np());
  }
  terminal_data_ = terminal_model_->createData(params_data_.back()->params);
  terminal_model_->set_params(terminal_data_,
                              params_model_.back()->get_params());
}

template <typename Scalar>
Scalar ShootingProblemTpl<Scalar>::calc(const std::vector<VectorXs>& xs,
                                        const std::vector<VectorXs>& us) {
  if (xs.size() != T_ + 1) {
    throw_pretty(
        "Invalid argument: " << "xs has wrong dimension (it should be " +
                                    std::to_string(T_ + 1) + ")");
  }
  if (us.size() != T_) {
    throw_pretty(
        "Invalid argument: " << "us has wrong dimension (it should be " +
                                    std::to_string(T_) + ")");
  }
  START_PROFILER("ShootingProblem::calc");

#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(nthreads_)
#endif
  for (std::size_t i = 0; i < T_; ++i) {
    running_models_[i]->calc(running_datas_[i], xs[i], us[i]);
  }
  terminal_model_->calc(terminal_data_, xs.back());

  cost_ = Scalar(0.);
#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp simd reduction(+ : cost_)
#endif
  for (std::size_t i = 0; i < T_; ++i) {
    cost_ += running_datas_[i]->cost;
  }
  cost_ += terminal_data_->cost;
  STOP_PROFILER("ShootingProblem::calc");
  return cost_;
}

template <typename Scalar>
Scalar ShootingProblemTpl<Scalar>::calcDiff(const std::vector<VectorXs>& xs,
                                            const std::vector<VectorXs>& us) {
  if (xs.size() != T_ + 1) {
    throw_pretty(
        "Invalid argument: " << "xs has wrong dimension (it should be " +
                                    std::to_string(T_ + 1) + ")");
  }
  if (us.size() != T_) {
    throw_pretty(
        "Invalid argument: " << "us has wrong dimension (it should be " +
                                    std::to_string(T_) + ")");
  }
  START_PROFILER("ShootingProblem::calcDiff");

#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(nthreads_)
#endif
  for (std::size_t i = 0; i < T_; ++i) {
    running_models_[i]->calcDiff(running_datas_[i], xs[i], us[i]);
  }
  terminal_model_->calcDiff(terminal_data_, xs.back());

  cost_ = Scalar(0.);
  // Apply SIMD only for floating-point types
  if (std::is_floating_point<Scalar>::value) {
#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp simd reduction(+ : cost_)
#endif
    for (std::size_t i = 0; i < T_; ++i) {
      cost_ += running_datas_[i]->cost;
    }
    cost_ += terminal_data_->cost;
  } else {  // For non-floating-point types (e.g., CppAD types), use the normal
            // loop without SIMD
    for (std::size_t i = 0; i < T_; ++i) {
      cost_ += running_datas_[i]->cost;
    }
    cost_ += terminal_data_->cost;
  }
  STOP_PROFILER("ShootingProblem::calcDiff");
  return cost_;
}

template <typename Scalar>
void ShootingProblemTpl<Scalar>::rollout(const std::vector<VectorXs>& us,
                                         std::vector<VectorXs>& xs) {
  if (xs.size() != T_ + 1) {
    throw_pretty(
        "Invalid argument: " << "xs has wrong dimension (it should be " +
                                    std::to_string(T_ + 1) + ")");
  }
  if (us.size() != T_) {
    throw_pretty(
        "Invalid argument: " << "us has wrong dimension (it should be " +
                                    std::to_string(T_) + ")");
  }
  START_PROFILER("ShootingProblem::rollout");

  xs[0] = x0_;
  for (std::size_t i = 0; i < T_; ++i) {
    const std::shared_ptr<ActionDataAbstract>& data = running_datas_[i];
    running_models_[i]->calc(data, xs[i], us[i]);
    xs[i + 1] = data->xnext;
  }
  terminal_model_->calc(terminal_data_, xs.back());
  STOP_PROFILER("ShootingProblem::rollout");
}

template <typename Scalar>
std::vector<typename MathBaseTpl<Scalar>::VectorXs>
ShootingProblemTpl<Scalar>::rollout_us(const std::vector<VectorXs>& us) {
  std::vector<VectorXs> xs;
  xs.resize(T_ + 1);
  rollout(us, xs);
  return xs;
}

template <typename Scalar>
void ShootingProblemTpl<Scalar>::quasiStatic(std::vector<VectorXs>& us,
                                             const std::vector<VectorXs>& xs) {
  if (xs.size() != T_) {
    throw_pretty(
        "Invalid argument: " << "xs has wrong dimension (it should be " +
                                    std::to_string(T_) + ")");
  }
  if (us.size() != T_) {
    throw_pretty(
        "Invalid argument: " << "us has wrong dimension (it should be " +
                                    std::to_string(T_) + ")");
  }

#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(nthreads_)
#endif
  for (std::size_t i = 0; i < T_; ++i) {
    running_models_[i]->quasiStatic(running_datas_[i], us[i], xs[i]);
  }
}

template <typename Scalar>
std::vector<typename MathBaseTpl<Scalar>::VectorXs>
ShootingProblemTpl<Scalar>::quasiStatic_xs(const std::vector<VectorXs>& xs) {
  std::vector<VectorXs> us;
  us.resize(T_);
  for (std::size_t i = 0; i < T_; ++i) {
    us[i] = VectorXs::Zero(running_models_[i]->get_nu());
  }
  quasiStatic(us, xs);
  return us;
}

template <typename Scalar>
void ShootingProblemTpl<Scalar>::circularAppend(
    std::shared_ptr<ActionModelAbstract> model,
    std::shared_ptr<ActionDataAbstract> data) {
  if (this->get_n_phases() != 0) {
    throw_pretty("Invalid call: problem must be reconstructed");
  }
  if (!model->checkData(data)) {
    throw_pretty("Invalid argument: "
                 << "action data is not consistent with the action model")
  }
  if (model->get_state()->get_nx() != nx_) {
    throw_pretty(
        "Invalid argument: " << "nx is not consistent with the other nodes")
  }
  if (model->get_state()->get_ndx() != ndx_) {
    throw_pretty("Invalid argument: "
                 << "ndx node is not consistent with the other nodes")
  }
  is_updated_ = true;
  for (std::size_t i = 0; i < T_ - 1; ++i) {
    running_models_[i] = running_models_[i + 1];
    running_datas_[i] = running_datas_[i + 1];
  }
  running_models_.back() = model;
  running_datas_.back() = data;
}

template <typename Scalar>
void ShootingProblemTpl<Scalar>::circularAppend(
    std::shared_ptr<ActionModelAbstract> model) {
  if (this->get_n_phases() != 0) {
    throw_pretty("Invalid call: problem must be reconstructed");
  }
  if (model->get_state()->get_nx() != nx_) {
    throw_pretty(
        "Invalid argument: " << "nx is not consistent with the other nodes")
  }
  if (model->get_state()->get_ndx() != ndx_) {
    throw_pretty("Invalid argument: "
                 << "ndx node is not consistent with the other nodes")
  }
  is_updated_ = true;
  for (std::size_t i = 0; i < T_ - 1; ++i) {
    running_models_[i] = running_models_[i + 1];
    running_datas_[i] = running_datas_[i + 1];
  }
  running_models_.back() = model;
  running_datas_.back() = model->createData();
}

template <typename Scalar>
void ShootingProblemTpl<Scalar>::updateNode(
    const std::size_t i, std::shared_ptr<ActionModelAbstract> model,
    std::shared_ptr<ActionDataAbstract> data) {
  if (this->get_n_phases() != 0) {
    throw_pretty("Invalid call: problem must be reconstructed");
  }
  if (i >= T_ + 1) {
    throw_pretty("Invalid argument: "
                 << "i is bigger than the allocated horizon (it should be less "
                    "than or equal to " +
                        std::to_string(T_ + 1) + ")");
  }
  if (!model->checkData(data)) {
    throw_pretty("Invalid argument: "
                 << "action data is not consistent with the action model")
  }
  if (model->get_state()->get_nx() != nx_) {
    throw_pretty(
        "Invalid argument: " << "nx is not consistent with the other nodes")
  }
  if (model->get_state()->get_ndx() != ndx_) {
    throw_pretty("Invalid argument: "
                 << "ndx node is not consistent with the other nodes")
  }
  is_updated_ = true;
  if (i == T_) {
    terminal_model_ = model;
    terminal_data_ = data;
  } else {
    running_models_[i] = model;
    running_datas_[i] = data;
  }
}

template <typename Scalar>
void ShootingProblemTpl<Scalar>::updateModel(
    const std::size_t i, std::shared_ptr<ActionModelAbstract> model) {
  if (this->get_n_phases() != 0) {
    throw_pretty("Invalid call: problem must be reconstructed");
  }
  if (i >= T_ + 1) {
    throw_pretty(
        "Invalid argument: "
        << "i is bigger than the allocated horizon (it should be lower than " +
               std::to_string(T_ + 1) + ")");
  }
  if (model->get_state()->get_nx() != nx_) {
    throw_pretty(
        "Invalid argument: " << "nx is not consistent with the other nodes")
  }
  if (model->get_state()->get_ndx() != ndx_) {
    throw_pretty(
        "Invalid argument: " << "ndx is not consistent with the other nodes")
  }
  is_updated_ = true;
  if (i == T_) {
    terminal_model_ = model;
    terminal_data_ = terminal_model_->createData();
  } else {
    running_models_[i] = model;
    running_datas_[i] = model->createData();
  }
}

template <typename Scalar>
template <typename NewScalar>
ShootingProblemTpl<NewScalar> ShootingProblemTpl<Scalar>::cast() const {
  typedef ShootingProblemTpl<NewScalar> ReturnType;
  typedef ActionModelAbstractTpl<NewScalar> NewActionModel;
  typedef ParameterPhaseModelTpl<NewScalar> NewParameterPhaseModel;
  const std::shared_ptr<NewActionModel> terminal_model =
      terminal_model_->template cast<NewScalar>();
  if (n_phases_ == 0) {
    ReturnType ret(x0_.template cast<NewScalar>(),
                   vector_cast<NewScalar>(running_models_), terminal_model);
    ret.set_nthreads(static_cast<int>(nthreads_));
    return ret;
  }

  std::vector<std::vector<std::shared_ptr<NewActionModel> > > model_phases(
      n_phases_);
  for (std::size_t i = 0; i < n_phases_; ++i) {
    model_phases[i].reserve(phase_end_[i] - phase_start_[i]);
    for (std::size_t t = phase_start_[i]; t < phase_end_[i]; ++t) {
      model_phases[i].push_back(running_models_[t]->template cast<NewScalar>());
    }
  }
  std::vector<std::shared_ptr<NewParameterPhaseModel> > params_model;
  params_model.reserve(n_phases_);
  for (std::size_t i = 0; i < n_phases_; ++i) {
    params_model.push_back(std::make_shared<NewParameterPhaseModel>(
        params_model_[i]->template cast<NewScalar>()));
  }
  ReturnType ret(x0_.template cast<NewScalar>(), model_phases, terminal_model,
                 params_model);
  for (std::size_t i = 0; i < n_phases_; ++i) {
    ret.update_p(params_data_[i]->params->params->p.template cast<NewScalar>(),
                 i);
  }
  ret.set_nthreads(static_cast<int>(nthreads_));
  return ret;
}

template <typename Scalar>
std::size_t ShootingProblemTpl<Scalar>::get_T() const {
  return T_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
ShootingProblemTpl<Scalar>::get_x0() const {
  return x0_;
}

template <typename Scalar>
void ShootingProblemTpl<Scalar>::allocateData() {
  running_datas_.resize(T_);
  for (std::size_t i = 0; i < T_; ++i) {
    const std::shared_ptr<ActionModelAbstract>& model = running_models_[i];
    running_datas_[i] = model->createData();
  }
  terminal_data_ = terminal_model_->createData();
}

template <typename Scalar>
const std::vector<std::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar> > >&
ShootingProblemTpl<Scalar>::get_runningModels() const {
  return running_models_;
}

template <typename Scalar>
const std::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar> >&
ShootingProblemTpl<Scalar>::get_terminalModel() const {
  return terminal_model_;
}

template <typename Scalar>
const std::vector<std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar> > >&
ShootingProblemTpl<Scalar>::get_runningDatas() const {
  return running_datas_;
}

template <typename Scalar>
const std::shared_ptr<crocoddyl::ActionDataAbstractTpl<Scalar> >&
ShootingProblemTpl<Scalar>::get_terminalData() const {
  return terminal_data_;
}

template <typename Scalar>
void ShootingProblemTpl<Scalar>::set_x0(const VectorXs& x0_in) {
  if (x0_in.size() != x0_.size()) {
    throw_pretty("Invalid argument: "
                 << "invalid size of x0 provided: Expected " << x0_.size()
                 << ", received " << x0_in.size());
  }
  x0_ = x0_in;
}

template <typename Scalar>
void ShootingProblemTpl<Scalar>::set_runningModels(
    const std::vector<std::shared_ptr<ActionModelAbstract> >& models) {
  if (this->get_n_phases() != 0) {
    throw_pretty("Invalid call: problem must be reconstructed");
  }
  for (std::size_t i = 0; i < T_; ++i) {
    const std::shared_ptr<ActionModelAbstract>& model = models[i];
    if (model->get_state()->get_nx() != nx_) {
      throw_pretty("Invalid argument: "
                   << "nx in " << i
                   << " node is not consistent with the other nodes")
    }
    if (model->get_state()->get_ndx() != ndx_) {
      throw_pretty("Invalid argument: "
                   << "ndx in " << i
                   << " node is not consistent with the other nodes")
    }
  }
  is_updated_ = true;
  T_ = models.size();
  running_models_.clear();
  running_datas_.clear();
  for (std::size_t i = 0; i < T_; ++i) {
    const std::shared_ptr<ActionModelAbstract>& model = models[i];
    running_models_.push_back(model);
    running_datas_.push_back(model->createData());
  }
}

template <typename Scalar>
void ShootingProblemTpl<Scalar>::set_terminalModel(
    std::shared_ptr<ActionModelAbstract> model) {
  if (this->get_n_phases() != 0) {
    throw_pretty("Invalid call: problem must be reconstructed");
  }
  if (model->get_state()->get_nx() != nx_) {
    throw_pretty(
        "Invalid argument: " << "nx is not consistent with the other nodes")
  }
  if (model->get_state()->get_ndx() != ndx_) {
    throw_pretty(
        "Invalid argument: " << "ndx is not consistent with the other nodes")
  }
  is_updated_ = true;
  terminal_model_ = model;
  terminal_data_ = terminal_model_->createData();
}

template <typename Scalar>
void ShootingProblemTpl<Scalar>::set_nthreads(const int nthreads) {
#ifndef CROCODDYL_WITH_MULTITHREADING
  (void)nthreads;
  std::cerr << "Warning: the number of threads won't affect the computational "
               "performance as multithreading "
               "support is not enabled."
            << std::endl;
#else
  if (nthreads < 1) {
    nthreads_ = CROCODDYL_WITH_NTHREADS;
  } else {
    nthreads_ = static_cast<std::size_t>(nthreads);
  }
  if (!enableMultithreading()) {
    std::cerr << "Warning: the number of threads won't affect the "
                 "computational performance as multithreading "
                 "support is not enabled."
              << std::endl;
    nthreads_ = 1;
  }
#endif
}

template <typename Scalar>
void ShootingProblemTpl<Scalar>::set_is_updated(const bool is_updated) {
  is_updated_ = is_updated;
}

template <typename Scalar>
std::size_t ShootingProblemTpl<Scalar>::get_nx() const {
  return nx_;
}

template <typename Scalar>
std::size_t ShootingProblemTpl<Scalar>::get_ndx() const {
  return ndx_;
}

template <typename Scalar>
std::size_t ShootingProblemTpl<Scalar>::get_nthreads() const {
#ifndef CROCODDYL_WITH_MULTITHREADING
  std::cerr << "Warning: the number of threads won't affect the computational "
               "performance as multithreading "
               "support is not enabled."
            << std::endl;
#endif
  return nthreads_;
}

template <typename Scalar>
bool ShootingProblemTpl<Scalar>::is_updated() {
  const bool status = is_updated_;
  is_updated_ = false;
  return status;
}

template <typename Scalar>
void ShootingProblemTpl<Scalar>::update_p(const Eigen::Ref<const VectorXs>& p,
                                          const std::size_t phase_idx) {
  if (n_phases_ == 0) {
    throw_pretty("Invalid call: shooting problem has no parameter phases");
  }
  if (phase_idx >= n_phases_) {
    throw_pretty("Invalid argument: phase_idx " << phase_idx << " >= n_phases "
                                                << n_phases_);
  }
  params_model_[phase_idx]->update(params_data_[phase_idx], p);
}

template <typename Scalar>
std::size_t ShootingProblemTpl<Scalar>::get_n_phases() const {
  return n_phases_;
}

template <typename Scalar>
std::vector<
    std::shared_ptr<typename ShootingProblemTpl<Scalar>::ActionModelAbstract> >
ShootingProblemTpl<Scalar>::get_runningPhaseModels(
    const std::size_t phase_idx) const {
  if (phase_idx >= n_phases_) {
    throw_pretty("Invalid argument: phase_idx " << phase_idx << " >= n_phases "
                                                << n_phases_);
  }
  std::vector<std::shared_ptr<ActionModelAbstract> > phase_models;
  phase_models.reserve(phase_end_[phase_idx] - phase_start_[phase_idx]);
  for (std::size_t t = phase_start_[phase_idx]; t < phase_end_[phase_idx];
       ++t) {
    phase_models.push_back(running_models_[t]);
  }
  return phase_models;
}

template <typename Scalar>
std::vector<
    std::shared_ptr<typename ShootingProblemTpl<Scalar>::ActionDataAbstract> >
ShootingProblemTpl<Scalar>::get_runningPhaseDatas(
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
const std::vector<
    std::shared_ptr<typename ShootingProblemTpl<Scalar>::ParameterPhaseModel> >&
ShootingProblemTpl<Scalar>::get_paramsModel() const {
  return params_model_;
}

template <typename Scalar>
const std::vector<
    std::shared_ptr<typename ShootingProblemTpl<Scalar>::ParameterPhaseData> >&
ShootingProblemTpl<Scalar>::get_paramsData() const {
  return params_data_;
}

template <typename Scalar>
const std::vector<std::size_t>& ShootingProblemTpl<Scalar>::get_phase_idxs()
    const {
  return phase_start_;
}

template <typename Scalar>
const std::vector<std::size_t>& ShootingProblemTpl<Scalar>::get_phase_edxs()
    const {
  return phase_end_;
}

template <typename Scalar>
bool ShootingProblemTpl<Scalar>::has_parameter_constraints() const {
  for (std::size_t i = 0; i < params_model_.size(); ++i) {
    if (params_model_[i]->has_constraints()) {
      return true;
    }
  }
  return false;
}

template <typename Scalar>
std::ostream& operator<<(std::ostream& os,
                         const ShootingProblemTpl<Scalar>& problem) {
  os << "ShootingProblem (T=" << problem.get_T() << ", nx=" << problem.get_nx()
     << ", ndx=" << problem.get_ndx() << ") " << std::endl
     << "  Models:" << std::endl;
  const std::vector<
      std::shared_ptr<crocoddyl::ActionModelAbstractTpl<Scalar> > >&
      runningModels = problem.get_runningModels();
  for (std::size_t t = 0; t < problem.get_T(); ++t) {
    os << "    " << t << ": " << *runningModels[t] << std::endl;
  }
  os << "    " << problem.get_T() << ": " << *problem.get_terminalModel();
  return os;
}

}  // namespace crocoddyl
