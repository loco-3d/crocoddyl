///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
SolverAbstractTpl<Scalar>::SolverAbstractTpl(
    std::shared_ptr<ShootingProblem> problem)
    : problem_(problem),
      th_acceptstep_(Scalar(0.1)),
      th_stop_(ScaleNumerics<Scalar>(1e-9)),
      th_gaptol_(Scalar(1e-16)),
      feasnorm_(LInf),
      iter_(0),
      tmp_feas_(Scalar(0.)) {
  allocateData();
}

template <typename Scalar>
void SolverAbstractTpl<Scalar>::resizeData() {
  START_PROFILER("SolverAbstractTpl<Scalar>::resizeData");
  resizeRunningData();
  resizeTerminalData();
  STOP_PROFILER("SolverAbstractTpl<Scalar>::resizeData");
}

template <typename Scalar>
void SolverAbstractTpl<Scalar>::resizeRunningData() {
  START_PROFILER("SolverAbstractTpl<Scalar>::resizeRunningData");
  const std::size_t T = problem_->get_T();
  const std::size_t ng_T = problem_->get_terminalModel()->get_ng_T();
  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    const std::size_t ng = model->get_ng();
    us_[t].conservativeResize(nu);
    us_try_[t].conservativeResize(nu);
    g_adj_[t].conservativeResize(ng);
  }
  g_adj_.back().conservativeResize(ng_T);
  STOP_PROFILER("SolverAbstractTpl<Scalar>::resizeRunningData");
}

template <typename Scalar>
void SolverAbstractTpl<Scalar>::resizeTerminalData() {}

template <typename Scalar>
Scalar SolverAbstractTpl<Scalar>::computeFeasibility(
    const std::vector<VectorXs>& fs) {
  tmp_feas_ = Scalar(0.);
  switch (feasnorm_) {
    case LInf:
      for (std::size_t t = 0; t < fs.size(); ++t) {
        tmp_feas_ =
            std::max(tmp_feas_, fs[t].template lpNorm<Eigen::Infinity>());
      }
      break;
    case L1:
      for (std::size_t t = 0; t < fs.size(); ++t) {
        tmp_feas_ += fs_[t].template lpNorm<1>();
      }
      break;
  }
  return tmp_feas_;
}

template <typename Scalar>
Scalar SolverAbstractTpl<Scalar>::computeDynamicFeasibility() {
  START_PROFILER("SolverAbstractTpl<Scalar>::computeDynamicFeasibility");
  tmp_feas_ = Scalar(0.);
  const std::size_t T = problem_->get_T();
  const VectorXs& x0 = problem_->get_x0();
  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract>>& datas =
      problem_->get_runningDatas();

  models[0]->get_state()->diff(xs_[0], x0, fs_[0]);
#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(problem_->get_nthreads())
#endif
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& m = models[t];
    const std::shared_ptr<ActionDataAbstract>& d = datas[t];
    m->get_state()->diff(xs_[t + 1], d->xnext, fs_[t + 1]);
  }
  switch (feasnorm_) {
    case LInf:
      tmp_feas_ =
          std::max(tmp_feas_, fs_[0].template lpNorm<Eigen::Infinity>());
      for (std::size_t t = 0; t < T; ++t) {
        tmp_feas_ =
            std::max(tmp_feas_, fs_[t + 1].template lpNorm<Eigen::Infinity>());
      }
      break;
    case L1:
      tmp_feas_ = fs_[0].template lpNorm<1>();
      for (std::size_t t = 0; t < T; ++t) {
        tmp_feas_ += fs_[t + 1].template lpNorm<1>();
      }
      break;
  }
  STOP_PROFILER("SolverAbstractTpl<Scalar>::computeDynamicFeasibility");
  return tmp_feas_;
}

template <typename Scalar>
Scalar SolverAbstractTpl<Scalar>::computeInequalityFeasibility() {
  START_PROFILER("SolverAbstractTpl<Scalar>::computeInequalityFeasibility");
  tmp_feas_ = Scalar(0.);
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract>>& datas =
      problem_->get_runningDatas();
  switch (feasnorm_) {
    case LInf:
      for (std::size_t t = 0; t < T; ++t) {
        if (models[t]->get_ng() > 0) {
          g_adj_[t] = datas[t]
                          ->g.cwiseMax(models[t]->get_g_lb())
                          .cwiseMin(models[t]->get_g_ub());
          tmp_feas_ = std::max(
              tmp_feas_,
              (datas[t]->g - g_adj_[t]).template lpNorm<Eigen::Infinity>());
        }
      }
      if (problem_->get_terminalModel()->get_ng_T() > 0) {
        g_adj_.back() =
            problem_->get_terminalData()
                ->g.cwiseMax(problem_->get_terminalModel()->get_g_lb())
                .cwiseMin(problem_->get_terminalModel()->get_g_ub());
        tmp_feas_ += (problem_->get_terminalData()->g - g_adj_.back())
                         .template lpNorm<Eigen::Infinity>();
      }
      break;
    case L1:
      for (std::size_t t = 0; t < T; ++t) {
        if (models[t]->get_ng() > 0) {
          g_adj_[t] = datas[t]
                          ->g.cwiseMax(models[t]->get_g_lb())
                          .cwiseMin(models[t]->get_g_ub());
          tmp_feas_ = std::max(tmp_feas_,
                               (datas[t]->g - g_adj_[t]).template lpNorm<1>());
        }
      }
      if (problem_->get_terminalModel()->get_ng_T() > 0) {
        g_adj_.back() =
            problem_->get_terminalData()
                ->g.cwiseMax(problem_->get_terminalModel()->get_g_lb())
                .cwiseMin(problem_->get_terminalModel()->get_g_ub());
        tmp_feas_ += (problem_->get_terminalData()->g - g_adj_.back())
                         .template lpNorm<1>();
      }
      break;
  }
  STOP_PROFILER("SolverAbstractTpl<Scalar>::computeInequalityFeasibility");
  return tmp_feas_;
}

template <typename Scalar>
Scalar SolverAbstractTpl<Scalar>::computeEqualityFeasibility() {
  START_PROFILER("SolverAbstractTpl<Scalar>::computeEqualityFeasibility");
  tmp_feas_ = Scalar(0.);
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract>>& datas =
      problem_->get_runningDatas();
  switch (feasnorm_) {
    case LInf:
      for (std::size_t t = 0; t < T; ++t) {
        if (models[t]->get_nh() > 0) {
          tmp_feas_ = std::max(tmp_feas_,
                               datas[t]->h.template lpNorm<Eigen::Infinity>());
        }
      }
      if (problem_->get_terminalModel()->get_nh_T() > 0) {
        tmp_feas_ = std::max(
            tmp_feas_,
            problem_->get_terminalData()->h.template lpNorm<Eigen::Infinity>());
      }
      break;
    case L1:
      for (std::size_t t = 0; t < T; ++t) {
        if (models[t]->get_nh() > 0) {
          tmp_feas_ += datas[t]->h.template lpNorm<1>();
        }
      }
      if (problem_->get_terminalModel()->get_nh_T() > 0) {
        tmp_feas_ += problem_->get_terminalData()->h.template lpNorm<1>();
      }
      break;
  }
  STOP_PROFILER("SolverAbstractTpl<Scalar>::computeEqualityFeasibility");
  return tmp_feas_;
}

template <typename Scalar>
void SolverAbstractTpl<Scalar>::setCandidate(
    const std::vector<VectorXs>& xs_warm, const std::vector<VectorXs>& us_warm,
    bool is_feasible) {
  START_PROFILER("SolverAbstractTpl<Scalar>::setCandidate");
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
      problem_->get_runningModels();
  if (xs_warm.size() == 0) {
    for (std::size_t t = 0; t < T; ++t) {
      const std::shared_ptr<ActionModelAbstract>& model = models[t];
      xs_[t] = model->get_state()->zero();
    }
    xs_.back() = problem_->get_terminalModel()->get_state()->zero();
  } else {
    if (xs_warm.size() != T + 1) {
      throw_pretty("Warm start state vector has wrong dimension, got "
                   << xs_warm.size() << " expecting " << (T + 1));
    }
    for (std::size_t t = 0; t < T; ++t) {
      const std::size_t nx = models[t]->get_state()->get_nx();
      if (static_cast<std::size_t>(xs_warm[t].size()) != nx) {
        throw_pretty("Invalid argument: "
                     << "xs_init[" + std::to_string(t) +
                            "] has wrong dimension ("
                     << xs_warm[t].size()
                     << " provided - it should be equal to " +
                            std::to_string(nx) + "). ActionModel: "
                     << *models[t]);
      }
    }
    const std::size_t nx = problem_->get_terminalModel()->get_state()->get_nx();
    if (static_cast<std::size_t>(xs_warm[T].size()) != nx) {
      throw_pretty("Invalid argument: "
                   << "xs_init[" + std::to_string(T) +
                          "] (terminal state) has wrong dimension ("
                   << xs_warm[T].size()
                   << " provided - it should be equal to " +
                          std::to_string(nx) + "). ActionModel: "
                   << *problem_->get_terminalModel());
    }
    std::copy(xs_warm.begin(), xs_warm.end(), xs_.begin());
  }
  if (us_warm.size() == 0) {
    for (std::size_t t = 0; t < T; ++t) {
      const std::shared_ptr<ActionModelAbstract>& model = models[t];
      const std::size_t nu = model->get_nu();
      us_[t] = VectorXs::Zero(nu);
    }
  } else {
    if (us_warm.size() != T) {
      throw_pretty("Warm start control has wrong dimension, got "
                   << us_warm.size() << " expecting " << T);
    }
    for (std::size_t t = 0; t < T; ++t) {
      const std::shared_ptr<ActionModelAbstract>& model = models[t];
      const std::size_t nu = model->get_nu();
      if (static_cast<std::size_t>(us_warm[t].size()) != nu) {
        throw_pretty("Invalid argument: "
                     << "us_init[" + std::to_string(t) +
                            "] has wrong dimension ("
                     << us_warm[t].size()
                     << " provided - it should be equal to " +
                            std::to_string(nu) + "). ActionModel: "
                     << *model);
      }
    }
    std::copy(us_warm.begin(), us_warm.end(), us_.begin());
  }
  is_feasible_ = is_feasible;
  STOP_PROFILER("SolverAbstractTpl<Scalar>::setCandidate");
}

template <typename Scalar>
void SolverAbstractTpl<Scalar>::allocateData() {
  // Guess trajectory
  const std::size_t ndx = problem_->get_ndx();
  const std::size_t T = problem_->get_T();
  const std::size_t ng_T = problem_->get_terminalModel()->get_ng_T();
  xs_.resize(T + 1);
  xs_try_.resize(T + 1);
  us_.resize(T);
  us_try_.resize(T);
  fs_.resize(T + 1);
  fs_try_.resize(T + 1);
  g_adj_.resize(T + 1);
  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    const std::size_t ng = model->get_ng();
    xs_[t] = model->get_state()->zero();
    xs_try_[t] = model->get_state()->zero();
    us_[t] = VectorXs::Zero(nu);
    us_try_[t] = VectorXs::Zero(nu);
    fs_[t] = VectorXs::Zero(ndx);
    fs_try_[t] = VectorXs::Zero(ndx);
    g_adj_[t] = VectorXs::Zero(ng);
  }
  xs_.back() = problem_->get_terminalModel()->get_state()->zero();
  xs_try_.back() = problem_->get_terminalModel()->get_state()->zero();
  fs_.back() = VectorXs::Zero(ndx);
  fs_try_.back() = VectorXs::Zero(ndx);
  g_adj_.back() = VectorXs::Zero(ng_T);
  // Cost, merit and convergence
  is_feasible_ = false;
  was_feasible_ = false;
  cost_ = Scalar(0.);
  cost_try_ = Scalar(0.);
  merit_ = Scalar(0.);
  stop_ = Scalar(0.);
  // Expected reduction and improvement
  DV_.setZero();
  CROCODDYL_DISABLE_WARNING_DEPRECATED
  // TODO: remove d_
  d_.setZero();
  CROCODDYL_ENABLE_WARNING_DEPRECATED
  dV_ = Scalar(0.);
  dPhi_ = Scalar(0.);
  dVexp_full_ = Scalar(0.);
  dVexp_ = Scalar(0.);
  dPhiexp_ = Scalar(0.);
  dfeas_ = Scalar(0.);
  // Current and next feasibility
  feas_ = Scalar(0.);
  ffeas_ = Scalar(0.);
  gfeas_ = Scalar(0.);
  hfeas_ = Scalar(0.);
  ffeas_try_ = Scalar(0.);
  gfeas_try_ = Scalar(0.);
  hfeas_try_ = Scalar(0.);
  tmp_feas_ = Scalar(0.);
  // Regularization and step length
  preg_ = Scalar(0.);
  dreg_ = Scalar(0.);
  steplength_ = Scalar(1.);
}

template <typename Scalar>
void SolverAbstractTpl<Scalar>::setCallbacks(
    const std::vector<std::shared_ptr<CallbackAbstract>>& callbacks) {
  callbacks_ = callbacks;
}

template <typename Scalar>
const std::vector<std::shared_ptr<CallbackAbstractTpl<Scalar>>>&
SolverAbstractTpl<Scalar>::getCallbacks() const {
  return callbacks_;
}

template <typename Scalar>
const std::shared_ptr<ShootingProblemTpl<Scalar>>&
SolverAbstractTpl<Scalar>::get_problem() const {
  return problem_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::VectorXs>&
SolverAbstractTpl<Scalar>::get_xs() const {
  return xs_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::VectorXs>&
SolverAbstractTpl<Scalar>::get_us() const {
  return us_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::VectorXs>&
SolverAbstractTpl<Scalar>::get_fs() const {
  return fs_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::VectorXs>&
SolverAbstractTpl<Scalar>::get_xs_try() const {
  return xs_try_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::VectorXs>&
SolverAbstractTpl<Scalar>::get_us_try() const {
  return us_try_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::VectorXs>&
SolverAbstractTpl<Scalar>::get_fs_try() const {
  return fs_try_;
}

template <typename Scalar>
bool SolverAbstractTpl<Scalar>::get_is_feasible() const {
  return is_feasible_;
}

template <typename Scalar>
Scalar SolverAbstractTpl<Scalar>::get_cost() const {
  return cost_;
}

template <typename Scalar>
Scalar SolverAbstractTpl<Scalar>::get_merit() const {
  return merit_;
}

template <typename Scalar>
Scalar SolverAbstractTpl<Scalar>::get_stop() const {
  return stop_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::Vector3s&
SolverAbstractTpl<Scalar>::get_DV() const {
  return DV_;
}

template <typename Scalar>
Scalar SolverAbstractTpl<Scalar>::get_dV() const {
  return dV_;
}

template <typename Scalar>
Scalar SolverAbstractTpl<Scalar>::get_dPhi() const {
  return dPhi_;
}

template <typename Scalar>
Scalar SolverAbstractTpl<Scalar>::get_dVexp() const {
  return dVexp_;
}

template <typename Scalar>
Scalar SolverAbstractTpl<Scalar>::get_dPhiexp() const {
  return dPhiexp_;
}

template <typename Scalar>
Scalar SolverAbstractTpl<Scalar>::get_dfeas() const {
  return dfeas_;
}

template <typename Scalar>
Scalar SolverAbstractTpl<Scalar>::get_feas() const {
  return feas_;
}

template <typename Scalar>
Scalar SolverAbstractTpl<Scalar>::get_ffeas() const {
  return ffeas_;
}

template <typename Scalar>
Scalar SolverAbstractTpl<Scalar>::get_gfeas() const {
  return gfeas_;
}

template <typename Scalar>
Scalar SolverAbstractTpl<Scalar>::get_hfeas() const {
  return hfeas_;
}

template <typename Scalar>
Scalar SolverAbstractTpl<Scalar>::get_ffeas_try() const {
  return ffeas_try_;
}

template <typename Scalar>
Scalar SolverAbstractTpl<Scalar>::get_gfeas_try() const {
  return gfeas_try_;
}

template <typename Scalar>
Scalar SolverAbstractTpl<Scalar>::get_hfeas_try() const {
  return hfeas_try_;
}

template <typename Scalar>
Scalar SolverAbstractTpl<Scalar>::get_preg() const {
  return preg_;
}

template <typename Scalar>
Scalar SolverAbstractTpl<Scalar>::get_dreg() const {
  return dreg_;
}

template <typename Scalar>
Scalar SolverAbstractTpl<Scalar>::get_steplength() const {
  return steplength_;
}

template <typename Scalar>
Scalar SolverAbstractTpl<Scalar>::get_th_acceptstep() const {
  return th_acceptstep_;
}

template <typename Scalar>
Scalar SolverAbstractTpl<Scalar>::get_th_stop() const {
  return th_stop_;
}

template <typename Scalar>
FeasibilityNorm SolverAbstractTpl<Scalar>::get_feasnorm() const {
  return feasnorm_;
}

template <typename Scalar>
std::size_t SolverAbstractTpl<Scalar>::get_iter() const {
  return iter_;
}

template <typename Scalar>
void SolverAbstractTpl<Scalar>::set_xs(const std::vector<VectorXs>& xs) {
  const std::size_t T = problem_->get_T();
  if (xs.size() != T + 1) {
    throw_pretty("Invalid argument: " << "xs list has to be of length " +
                                             std::to_string(T + 1));
  }
  const std::size_t nx = problem_->get_nx();
  for (std::size_t t = 0; t < T; ++t) {
    if (static_cast<std::size_t>(xs[t].size()) != nx) {
      throw_pretty("Invalid argument: "
                   << "xs[" + std::to_string(t) + "] has wrong dimension ("
                   << xs[t].size()
                   << " provided - it should be " + std::to_string(nx) + ")")
    }
  }
  if (static_cast<std::size_t>(xs[T].size()) != nx) {
    throw_pretty("Invalid argument: "
                 << "xs[" + std::to_string(T) +
                        "] (terminal state) has wrong dimension ("
                 << xs[T].size()
                 << " provided - it should be " + std::to_string(nx) + ")")
  }
  xs_ = xs;
}

template <typename Scalar>
void SolverAbstractTpl<Scalar>::set_us(const std::vector<VectorXs>& us) {
  const std::size_t T = problem_->get_T();
  if (us.size() != T) {
    throw_pretty("Invalid argument: " << "us list has to be of length " +
                                             std::to_string(T));
  }
  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    if (static_cast<std::size_t>(us[t].size()) != nu) {
      throw_pretty("Invalid argument: "
                   << "us[" + std::to_string(t) + "] has wrong dimension ("
                   << us[t].size()
                   << " provided - it should be " + std::to_string(nu) + ")")
    }
  }
  us_ = us;
}

template <typename Scalar>
void SolverAbstractTpl<Scalar>::set_preg(const Scalar preg) {
  if (preg < Scalar(0.)) {
    throw_pretty("Invalid argument: " << "preg value has to be positive.");
  }
  preg_ = preg;
}

template <typename Scalar>
void SolverAbstractTpl<Scalar>::set_dreg(const Scalar dreg) {
  if (dreg < Scalar(0.)) {
    throw_pretty("Invalid argument: " << "dreg value has to be positive.");
  }
  dreg_ = dreg;
}

template <typename Scalar>
void SolverAbstractTpl<Scalar>::set_th_acceptstep(const Scalar th_acceptstep) {
  if (Scalar(0.) >= th_acceptstep || th_acceptstep > Scalar(1.)) {
    throw_pretty(
        "Invalid argument: " << "th_acceptstep value should between 0 and 1.");
  }
  th_acceptstep_ = th_acceptstep;
}

template <typename Scalar>
void SolverAbstractTpl<Scalar>::set_th_stop(const Scalar th_stop) {
  if (th_stop <= Scalar(0.)) {
    throw_pretty("Invalid argument: " << "th_stop value has to higher than 0.");
  }
  th_stop_ = th_stop;
}

template <typename Scalar>
void SolverAbstractTpl<Scalar>::set_th_gaptol(const Scalar th_gaptol) {
  if (Scalar(0.) > th_gaptol) {
    throw_pretty("Invalid argument: " << "th_gaptol value has to be positive.");
  }
  th_gaptol_ = th_gaptol;
}

template <typename Scalar>
void SolverAbstractTpl<Scalar>::set_feasnorm(const FeasibilityNorm feasnorm) {
  feasnorm_ = feasnorm;
}

template <typename Scalar>
bool raiseIfNaN(const Scalar value) {
  if (std::isnan(value) || std::isinf(value) || value >= Scalar(1e30)) {
    return true;
  } else {
    return false;
  }
}

}  // namespace crocoddyl
