///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
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
      is_updated_(false) {
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
      nthreads_(1) {
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
    const ShootingProblemTpl<Scalar>& problem)
    : cost_(Scalar(0.)),
      T_(problem.get_T()),
      x0_(problem.get_x0()),
      terminal_model_(problem.get_terminalModel()),
      terminal_data_(problem.get_terminalData()),
      running_models_(problem.get_runningModels()),
      running_datas_(problem.get_runningDatas()),
      nx_(problem.get_nx()),
      ndx_(problem.get_ndx()) {}

template <typename Scalar>
ShootingProblemTpl<Scalar>::~ShootingProblemTpl() {}

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
  ReturnType ret(x0_.template cast<NewScalar>(),
                 vector_cast<NewScalar>(running_models_),
                 terminal_model_->template cast<NewScalar>());
  ret.set_nthreads((int)nthreads_);
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
    const std::shared_ptr<ActionModelAbstract>& model = running_models_[i];
    running_models_.push_back(model);
    running_datas_.push_back(model->createData());
  }
}

template <typename Scalar>
void ShootingProblemTpl<Scalar>::set_terminalModel(
    std::shared_ptr<ActionModelAbstract> model) {
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
