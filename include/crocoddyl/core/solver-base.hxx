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
      th_stop_(sqrt(std::numeric_limits<Scalar>::epsilon())),
      th_gaptol_(Scalar(1e-16)),
      feasnorm_(LInf),
      iter_(0),
      tmp_feas_(Scalar(0.)),
      reg_min_(ScaleNumerics<Scalar>(1e-9)),
      reg_max_(ScaleNumerics<Scalar>(1e9, 1e-4)),
      nh_T_(problem->get_terminalModel()->get_nh_T()),
      ng_T_(problem->get_terminalModel()->get_ng_T()),
      acceptstep_(false),
      recalcdir_(true) {
  allocateData();
  // Defining the list of step lengths used in the linear search routine
  const std::size_t n_alphas = 10;
  alphas_.resize(n_alphas);
  for (std::size_t n = 0; n < n_alphas; ++n) {
    alphas_[n] = Scalar(1.) / pow(Scalar(2.), static_cast<Scalar>(n));
  }
}

template <typename Scalar>
bool SolverAbstractTpl<Scalar>::solve(const std::vector<VectorXs>& init_xs,
                                      const std::vector<VectorXs>& init_us,
                                      const std::size_t maxiter, const bool,
                                      const Scalar init_reg) {
  START_PROFILER("SolverAbstract::solve");
  const std::size_t nh_T = problem_->get_terminalModel()->get_nh_T();
  const std::size_t ng_T = problem_->get_terminalModel()->get_ng_T();
  if (problem_->is_updated()) {
    resizeData();
  } else if ((nh_T_ != nh_T && nh_T != 0) ||
             (ng_T_ != ng_T && ng_T != 0)) {  // we need to update terminal data
    resizeTerminalData();
    nh_T_ = nh_T;
    ng_T_ = ng_T;
  }
  // TODO: Deprecate isfeasible_. Update setCandidate API.
  setCandidate(init_xs, init_us, false);
  // Initialize the value used for primal and dual regularization
  if (isnan(init_reg)) {
    preg_ = reg_min_;
    dreg_ = reg_min_;
  } else {
    preg_ = init_reg;
    dreg_ = init_reg;
  }
  acceptstep_ = false;
  for (iter_ = 0; iter_ < maxiter; ++iter_) {
    recalcdir_ = true;
    // Compute search direction
    while (true) {
      try {
        computeDirection(recalcdir_);
      } catch (std::exception& e) {
        recalcdir_ = false;
        increaseRegularization();
        if (preg_ == reg_max_) {
          return false;
        } else {
          continue;
        }
      }
      break;
    }
    // Estimate the expected improvement
    expectedImprovement();
    dVexp_full_ = DV_[0] + DV_[1] + Scalar(0.5) * DV_[2];
    updateMeritFunction();
    // Try and evaluate the search direction
    for (typename std::vector<Scalar>::const_iterator it = alphas_.begin();
         it != alphas_.end(); ++it) {
      // TODO: break the forward pass if the allocated time has been reached
      // (c++)
      steplength_ = *it;
      try {
        tryStep(steplength_);
      } catch (std::exception& e) {
        std::string msg = e.what();
        if (msg.find("must be implemented in subclass.") != std::string::npos) {
          std::cerr << msg << std::endl;
          STOP_PROFILER("SolverAbstract::solve");
          return false;
        }
        continue;
      }
      dImpr_ = std::max(dV_, dPhi_);
      acceptstep_ = checkAcceptance();
      // Set candidate guess, cost and feasibilities if we accept the step
      if (acceptstep_) {
        setCandidate(xs_try_, us_try_, false);
        updateCandidate();
        break;
      }
    }
    stoppingCriteria();
    const std::size_t n_callbacks = callbacks_.size();
    for (std::size_t c = 0; c < n_callbacks; ++c) {
      CallbackAbstract& callback = *callbacks_[c];
      callback(*this);
    }
    if (decreaseRegularizationCriteria()) {
      decreaseRegularization();
    }
    if (increaseRegularizationCriteria()) {
      if (preg_ == reg_max_) {
        STOP_PROFILER("SolverAbstract::solve");
        return false;
      }
      increaseRegularization();
    }
    if (stop_ < th_stop_) {
      STOP_PROFILER("SolverAbstract::solve");
      return true;
    }
  }
  STOP_PROFILER("SolverAbstract::solve");
  return false;
}

template <typename Scalar>
void SolverAbstractTpl<Scalar>::computeDirection(const bool) {
  throw_pretty(
      "SolverAbstract.computeDirection(): must be implemented in subclass.");
}

template <typename Scalar>
Scalar SolverAbstractTpl<Scalar>::tryStep(const Scalar steplength) {
  START_PROFILER("SolverAbstract::tryStep");
  // Update primal, dual and slack variables
  computeCandidate(steplength);
  // Compute the expected and current value function improvements
  dVexp_ = DV_[0] + steplength * (DV_[1] + Scalar(0.5) * steplength * DV_[2]);
  dV_ = cost_ - cost_try_;
  // Compute the new infeasibilities
  ffeas_try_ = computeFeasibility(fs_try_);
  gfeas_try_ = computeInequalityFeasibility();
  hfeas_try_ = computeEqualityFeasibility();
  // Compute the infeasibility improvements
  dfeas_ = ffeas_ - ffeas_try_;
  dfeas_ += gfeas_ - gfeas_try_;
  dfeas_ += hfeas_ - hfeas_try_;
  // Compute the expected and current merit function improvements
  computeMeritFunctionImprovement();
  computeExpectedMeritFunctionImprovement();
  STOP_PROFILER("SolverAbstract::tryStep");
  return dV_;
}

template <typename Scalar>
void SolverAbstractTpl<Scalar>::computeCandidate(const Scalar) {
  throw_pretty(
      "SolverAbstract.computeCandidate(): must be implemented in subclass.");
}

template <typename Scalar>
void SolverAbstractTpl<Scalar>::setCandidate(
    const std::vector<VectorXs>& xs_warm, const std::vector<VectorXs>& us_warm,
    bool is_feasible) {
  START_PROFILER("SolverAbstract::setCandidate");
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
  STOP_PROFILER("SolverAbstract::setCandidate");
}

template <typename Scalar>
void SolverAbstractTpl<Scalar>::computeMeritFunctionImprovement() {
  throw_pretty(
      "SolverAbstract.computeMeritFunctionImprovement(): must be implemented "
      "in subclass.");
}

template <typename Scalar>
void SolverAbstractTpl<Scalar>::computeExpectedMeritFunctionImprovement() {
  throw_pretty(
      "SolverAbstract.computeExpectedMeritFunctionImprovement(): must be "
      "implemented in subclass.");
}

template <typename Scalar>
void SolverAbstractTpl<Scalar>::updateMeritFunction() {
  throw_pretty(
      "SolverAbstract.updateMeritFunction(): must be implemented in subclass.");
}

template <typename Scalar>
bool SolverAbstractTpl<Scalar>::checkAcceptance() {
  throw_pretty(
      "SolverAbstract.checkAcceptance(): must be implemented in subclass.");
  return true;
}

template <typename Scalar>
void SolverAbstractTpl<Scalar>::calcDir() {
  START_PROFILER("SolverAbstract::calcDir");
  if (!acceptstep_) {
    problem_->calc(xs_, us_);
  }
  cost_ = problem_->calcDiff(xs_, us_);
  ffeas_ = computeDynamicFeasibility();
  gfeas_ = computeInequalityFeasibility();
  hfeas_ = computeEqualityFeasibility();
  feas_ = ffeas_ + gfeas_ + hfeas_;
  STOP_PROFILER("SolverAbstract::calcDir");
}

template <typename Scalar>
void SolverAbstractTpl<Scalar>::updateCandidate() {
  throw_pretty(
      "SolverAbstract.updateCandidate(): must be implemented in subclass.");
}

template <typename Scalar>
bool SolverAbstractTpl<Scalar>::increaseRegularizationCriteria() {
  throw_pretty(
      "SolverAbstract.increaseRegularizationCriteria(): must be implemented in "
      "subclass.");
  return false;
}

template <typename Scalar>
bool SolverAbstractTpl<Scalar>::decreaseRegularizationCriteria() {
  throw_pretty(
      "SolverAbstract.decreaseRegularizationCriteria(): must be implemented in "
      "subclass.");
  return false;
}

template <typename Scalar>
void SolverAbstractTpl<Scalar>::increaseRegularization() {
  throw_pretty(
      "SolverAbstract.increaseRegularization(): must be implemented in "
      "subclass.");
}

template <typename Scalar>
void SolverAbstractTpl<Scalar>::decreaseRegularization() {
  throw_pretty(
      "SolverAbstract.decreaseRegularization(): must be implemented in "
      "subclass.");
}

template <typename Scalar>
void SolverAbstractTpl<Scalar>::resizeData() {
  START_PROFILER("SolverAbstract::resizeData");
  resizeRunningData();
  resizeTerminalData();
  STOP_PROFILER("SolverAbstract::resizeData");
}

template <typename Scalar>
void SolverAbstractTpl<Scalar>::resizeRunningData() {
  START_PROFILER("SolverAbstract::resizeRunningData");
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
    dus_[t].conservativeResize(nu);
    u_adj_[t].conservativeResize(nu);
    g_adj_[t].conservativeResize(ng);
  }
  g_adj_.back().conservativeResize(ng_T);
  STOP_PROFILER("SolverAbstract::resizeRunningData");
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
        tmp_feas_ += fs[t].template lpNorm<1>();
      }
      break;
    case L2:
      for (std::size_t t = 0; t < fs.size(); ++t) {
        // Sum the squared norms
        tmp_feas_ += fs[t].squaredNorm();
      }
      // Take the square root of the total sum
      tmp_feas_ = std::sqrt(tmp_feas_);
      break;
  }
  return tmp_feas_;
}

template <typename Scalar>
Scalar SolverAbstractTpl<Scalar>::computeDynamicFeasibility() {
  START_PROFILER("SolverAbstract::computeDynamicFeasibility");
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
    case L2:
      // Start with the squared norm of the initial state error
      tmp_feas_ = fs_[0].squaredNorm();
      for (std::size_t t = 0; t < T; ++t) {
        // Accumulate the squared norms of the running state errors
        tmp_feas_ += fs_[t + 1].squaredNorm();
      }
      // Take the square root of the total sum at the very end
      tmp_feas_ = std::sqrt(tmp_feas_);
      break;
  }
  STOP_PROFILER("SolverAbstract::computeDynamicFeasibility");
  return tmp_feas_;
}

template <typename Scalar>
Scalar SolverAbstractTpl<Scalar>::computeStateFeasibility(
    const std::vector<VectorXs>& xs) {
  START_PROFILER("SolverAbstract::computeStateFeasibility");
  tmp_feas_ = Scalar(0.);
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
      problem_->get_runningModels();

  for (std::size_t t = 0; t < T; ++t) {
    if (models[t]->get_state()->get_has_limits()) {
      x_adj_[t] = xs[t]
                      .cwiseMax(models[t]->get_state()->get_lb())
                      .cwiseMin(models[t]->get_state()->get_ub());
    }
  }
  if (problem_->get_terminalModel()->get_state()->get_has_limits()) {
    x_adj_.back() =
        xs[T]
            .cwiseMax(problem_->get_terminalModel()->get_state()->get_lb())
            .cwiseMin(problem_->get_terminalModel()->get_state()->get_ub());
  }

  switch (feasnorm_) {
    case LInf:
      for (std::size_t t = 0; t < T; ++t) {
        if (models[t]->get_state()->get_has_limits()) {
          tmp_feas_ =
              std::max(tmp_feas_,
                       (xs[t] - x_adj_[t]).template lpNorm<Eigen::Infinity>());
        }
      }
      if (problem_->get_terminalModel()->get_state()->get_has_limits()) {
        tmp_feas_ = std::max(
            tmp_feas_,
            (xs[T] - x_adj_.back()).template lpNorm<Eigen::Infinity>());
      }
      break;
    case L1:
      for (std::size_t t = 0; t < T; ++t) {
        if (models[t]->get_state()->get_has_limits()) {
          tmp_feas_ += (xs[t] - x_adj_[t]).template lpNorm<1>();
        }
      }
      if (problem_->get_terminalModel()->get_state()->get_has_limits()) {
        tmp_feas_ += (xs[T] - x_adj_.back()).template lpNorm<1>();
      }
      break;
    case L2:
      for (std::size_t t = 0; t < T; ++t) {
        if (models[t]->get_state()->get_has_limits()) {
          tmp_feas_ += (xs[t] - x_adj_[t]).squaredNorm();
        }
      }
      if (problem_->get_terminalModel()->get_state()->get_has_limits()) {
        tmp_feas_ += (xs[T] - x_adj_.back()).squaredNorm();
      }
      tmp_feas_ = std::sqrt(tmp_feas_);
      break;
  }
  STOP_PROFILER("SolverAbstract::computeStateFeasibility");
  return tmp_feas_;
}

template <typename Scalar>
Scalar SolverAbstractTpl<Scalar>::computeControlFeasibility(
    const std::vector<VectorXs>& us) {
  START_PROFILER("SolverAbstract::computeControlFeasibility");
  tmp_feas_ = Scalar(0.);
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    if (models[t]->get_has_control_limits()) {
      u_adj_[t] =
          us[t].cwiseMax(models[t]->get_u_lb()).cwiseMin(models[t]->get_u_ub());
    }
  }
  switch (feasnorm_) {
    case LInf:
      for (std::size_t t = 0; t < T; ++t) {
        if (models[t]->get_has_control_limits()) {
          tmp_feas_ =
              std::max(tmp_feas_,
                       (us[t] - u_adj_[t]).template lpNorm<Eigen::Infinity>());
        }
      }
      break;
    case L1:
      for (std::size_t t = 0; t < T; ++t) {
        if (models[t]->get_has_control_limits()) {
          tmp_feas_ += (us[t] - u_adj_[t]).template lpNorm<1>();
        }
      }
      break;
    case L2:
      for (std::size_t t = 0; t < T; ++t) {
        if (models[t]->get_has_control_limits()) {
          tmp_feas_ += (us[t] - u_adj_[t]).squaredNorm();
        }
      }
      tmp_feas_ = std::sqrt(tmp_feas_);
      break;
  }
  STOP_PROFILER("SolverAbstract::computeControlFeasibility");
  return tmp_feas_;
}

template <typename Scalar>
Scalar SolverAbstractTpl<Scalar>::computeInequalityFeasibility() {
  START_PROFILER("SolverAbstract::computeInequalityFeasibility");
  tmp_feas_ = Scalar(0.);
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract>>& datas =
      problem_->get_runningDatas();
  for (std::size_t t = 0; t < T; ++t) {
    if (models[t]->get_ng() > 0) {
      g_adj_[t] = datas[t]
                      ->g.cwiseMax(models[t]->get_g_lb())
                      .cwiseMin(models[t]->get_g_ub());
    }
  }
  if (problem_->get_terminalModel()->get_ng_T() > 0) {
    g_adj_.back() = problem_->get_terminalData()
                        ->g.cwiseMax(problem_->get_terminalModel()->get_g_lb())
                        .cwiseMin(problem_->get_terminalModel()->get_g_ub());
  }
  switch (feasnorm_) {
    case LInf:
      for (std::size_t t = 0; t < T; ++t) {
        if (models[t]->get_ng() > 0) {
          tmp_feas_ = std::max(
              tmp_feas_,
              (datas[t]->g - g_adj_[t]).template lpNorm<Eigen::Infinity>());
        }
      }
      if (problem_->get_terminalModel()->get_ng_T() > 0) {
        tmp_feas_ = std::max(tmp_feas_,
                             (problem_->get_terminalData()->g - g_adj_.back())
                                 .template lpNorm<Eigen::Infinity>());
      }
      break;
    case L1:
      for (std::size_t t = 0; t < T; ++t) {
        if (models[t]->get_ng() > 0) {
          tmp_feas_ += (datas[t]->g - g_adj_[t]).template lpNorm<1>();
        }
      }
      if (problem_->get_terminalModel()->get_ng_T() > 0) {
        tmp_feas_ += (problem_->get_terminalData()->g - g_adj_.back())
                         .template lpNorm<1>();
      }
      break;
    case L2:
      for (std::size_t t = 0; t < T; ++t) {
        if (models[t]->get_ng() > 0) {
          tmp_feas_ += (datas[t]->g - g_adj_[t]).squaredNorm();
        }
      }
      if (problem_->get_terminalModel()->get_ng_T() > 0) {
        tmp_feas_ +=
            (problem_->get_terminalData()->g - g_adj_.back()).squaredNorm();
      }
      tmp_feas_ = std::sqrt(tmp_feas_);
      break;
  }
  STOP_PROFILER("SolverAbstract::computeInequalityFeasibility");
  return tmp_feas_;
}

template <typename Scalar>
Scalar SolverAbstractTpl<Scalar>::computeEqualityFeasibility() {
  START_PROFILER("SolverAbstract::computeEqualityFeasibility");
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
    case L2:
      for (std::size_t t = 0; t < T; ++t) {
        if (models[t]->get_nh() > 0) {
          tmp_feas_ += datas[t]->h.squaredNorm();
        }
      }
      if (problem_->get_terminalModel()->get_nh_T() > 0) {
        tmp_feas_ += problem_->get_terminalData()->h.squaredNorm();
      }
      tmp_feas_ = std::sqrt(tmp_feas_);
      break;
  }
  STOP_PROFILER("SolverAbstract::computeEqualityFeasibility");
  return tmp_feas_;
}

template <typename Scalar>
void SolverAbstractTpl<Scalar>::allocateData() {
  // Guess trajectory
  const std::size_t ndx = problem_->get_ndx();
  const std::size_t T = problem_->get_T();
  const std::size_t ng_T = problem_->get_terminalModel()->get_ng_T();
  xs_.resize(T + 1);
  xs_try_.resize(T + 1);
  x_adj_.resize(T + 1);
  us_.resize(T);
  us_try_.resize(T);
  u_adj_.resize(T);
  fs_.resize(T + 1);
  fs_try_.resize(T + 1);
  dxs_.resize(T + 1);
  dus_.resize(T);
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
    dxs_[t] = VectorXs::Zero(ndx);
    dus_[t] = VectorXs::Zero(nu);
    g_adj_[t] = VectorXs::Zero(ng);
    x_adj_[t] = VectorXs::Zero(ndx);
    u_adj_[t] = VectorXs::Zero(nu);
  }
  xs_.back() = problem_->get_terminalModel()->get_state()->zero();
  xs_try_.back() = problem_->get_terminalModel()->get_state()->zero();
  x_adj_.back() = VectorXs::Zero(ndx);
  fs_.back() = VectorXs::Zero(ndx);
  fs_try_.back() = VectorXs::Zero(ndx);
  dxs_.back() = VectorXs::Zero(ndx);
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
  dV_ = Scalar(0.);
  dPhi_ = Scalar(0.);
  dImpr_ = Scalar(0.);
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
const std::vector<Scalar>& SolverAbstractTpl<Scalar>::get_alphas() const {
  return alphas_;
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
SolverAbstractTpl<Scalar>::get_dxs() const {
  return dxs_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::VectorXs>&
SolverAbstractTpl<Scalar>::get_dus() const {
  return dus_;
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
Scalar SolverAbstractTpl<Scalar>::get_reg_min() const {
  return reg_min_;
}

template <typename Scalar>
Scalar SolverAbstractTpl<Scalar>::get_reg_max() const {
  return reg_max_;
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
void SolverAbstractTpl<Scalar>::set_alphas(const std::vector<Scalar>& alphas) {
  Scalar prev_alpha = alphas[0];
  if (prev_alpha != Scalar(1.)) {
    std::cerr << "Warning: alpha[0] should be 1" << std::endl;
  }
  for (std::size_t i = 1; i < alphas.size(); ++i) {
    Scalar alpha = alphas[i];
    if (Scalar(0.) >= alpha) {
      throw_pretty("Invalid argument: " << "alpha values has to be positive.");
    }
    if (alpha >= prev_alpha) {
      throw_pretty(
          "Invalid argument: " << "alpha values are monotonously decreasing.");
    }
    prev_alpha = alpha;
  }
  alphas_ = alphas;
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
                   << " provided - it should be " + std::to_string(nx) + ")");
    }
  }
  if (static_cast<std::size_t>(xs[T].size()) != nx) {
    throw_pretty("Invalid argument: "
                 << "xs[" + std::to_string(T) +
                        "] (terminal state) has wrong dimension ("
                 << xs[T].size()
                 << " provided - it should be " + std::to_string(nx) + ")");
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
                   << " provided - it should be " + std::to_string(nu) + ")");
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
void SolverAbstractTpl<Scalar>::set_reg_min(const Scalar regmin) {
  if (Scalar(0.) > regmin) {
    throw_pretty("Invalid argument: " << "regmin value has to be positive.");
  }
  reg_min_ = regmin;
}

template <typename Scalar>
void SolverAbstractTpl<Scalar>::set_reg_max(const Scalar regmax) {
  if (Scalar(0.) > regmax) {
    throw_pretty("Invalid argument: " << "regmax value has to be positive.");
  }
  reg_max_ = regmax;
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
