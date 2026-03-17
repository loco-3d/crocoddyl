///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2025-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
SolverOdynSQPTpl<Scalar>::SolverOdynSQPTpl(
    std::shared_ptr<ShootingProblem> problem)
    : SolverAbstract(problem),
      reg_incfactor_(Scalar(10.)),
      reg_decfactor_(Scalar(5.)),
      th_grad_(ScaleNumerics<Scalar>(1e-12)),
      th_noimprovement_(
          std::pow(std::numeric_limits<Scalar>::epsilon(), Scalar(0.8))),
      th_stepdec_(Scalar(0.25)),
      th_stepinc_(Scalar(0.25)),
      th_minimprove_(Scalar(1e-2)),
      th_acceptnegstep_(Scalar(8)),
      th_acceptminstep_(Scalar(0.01)),
      rho_(Scalar(0.3)),
      th_minfeas_(std::sqrt(std::numeric_limits<Scalar>::epsilon() /
                            (Scalar(1.) - rho_))),
      upsilon_(Scalar(0.)),
      upsilon_decfactor_(Scalar(0.5)),
      zero_upsilon_(false),
      verbose_level_(odyn::VerboseLevel::Silent) {
  // Allocating the solver's data
  allocateData();
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::computeDirection(const bool recalc) {
  START_PROFILER("SolverOdynSQP::computeDirection");
  // Update the batch's derivatives
  if (recalc) {
    SolverAbstract::calcDir();
  }
  computeQuadraticModel();
  // Solve the QP problem using Odyn
  // qp_params_.stop_abs = 1e-4
  qp_solver_.solve(qp_model_, qp_data_, qp_params_, verbose_level_);
  // Unpack primal into dx/du
  x_ = qp_model_.get_x(qp_data_);
  extractQpDirection(x_);
  STOP_PROFILER("SolverOdynSQP::computeDirection");
}

template <typename Scalar>
Scalar SolverOdynSQPTpl<Scalar>::stoppingCriteria() {
  feas_ = ffeas_ + gfeas_ + hfeas_;
  stop_ =
      std::max(feas_, std::abs(dVexp_full_) / (Scalar(1.) + std::abs(cost_)));
  return stop_;
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::Vector3s
SolverOdynSQPTpl<Scalar>::expectedImprovement() {
  // We define dVexp = Vexp - Vexptry as done for dV
  const std::size_t T = problem_->get_T();
  DV_.setZero();
  const std::vector<std::shared_ptr<ActionDataAbstract>>& datas =
      problem_->get_runningDatas();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionDataAbstract>& d = datas[t];
    Lxx_dx_[t].noalias() = d->Lxx * dxs_[t];
    Luu_du_[t].noalias() = d->Luu * dus_[t];
    Lxu_du_[t].noalias() = d->Lxu * dus_[t];
    DV_[1] -= dxs_[t].dot(d->Lx);
    DV_[1] -= dus_[t].dot(d->Lu);
    DV_[2] -= dxs_[t].dot(Lxx_dx_[t]);
    DV_[2] -= dus_[t].dot(Luu_du_[t]);
    DV_[2] -= Scalar(2.) * dxs_[t].dot(Lxu_du_[t]);
  }
  const std::shared_ptr<ActionDataAbstract>& d = problem_->get_terminalData();
  Lxx_dx_.back().noalias() = d->Lxx * dxs_.back();
  DV_[1] -= dxs_.back().dot(d->Lx);
  DV_[2] -= dxs_.back().dot(Lxx_dx_.back());
  return DV_;
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::computeMeritFunctionImprovement() {
  dPhi_ = dV_ + upsilon_ * dfeas_;
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::computeExpectedMeritFunctionImprovement() {
  dPhiexp_ = dVexp_ + steplength_ * upsilon_ * dfeas_;
}

template <typename Scalar>
bool SolverOdynSQPTpl<Scalar>::checkAcceptance() {
  // Check if we should accept or not the step. The criterio is as follows.
  // When expected to decrease the merit function value (dPhiexp > 0), we
  // analyse if we are actually decreasing or not (dPhi > 0 or dPhi < 0) and
  // define different criterio. For the first case (dPhi > 0), we use the
  // Armijo condition with the merit function. Instead, for the second case,
  // we use the Armijo condition with the cost function as this encourage
  // progress and the possibility of increasing the cost when expectations
  // are unrealistic. Moreover, when it is expected to increase the merit if
  // the feasibility passes our stopping criteria or in the cost function
  // otherwise. This approach enables our solver to increase both
  // infeasibility and cost in order to ensure convergence; it increases the
  // algorithm's globalization. Finally, we accept any improvement for step
  // lengths smaller than th_acceptMinStep. This ensures any possible
  // progress in the iteration.
  acceptstep_ = false;
  if ((std::abs(dPhi_) <= th_noimprovement_) &&
      (std::abs(dPhiexp_) <= th_noimprovement_)) {
    acceptstep_ = true;  // we can't make further improvement
  } else if (dPhiexp_ >= Scalar(0.)) {
    if (dPhi_ > Scalar(0.)) {
      if (dPhi_ > th_acceptstep_ * dPhiexp_ || std::abs(DV_[1]) < th_grad_) {
        acceptstep_ = true;
      }
    } else if (dV_ > th_acceptstep_ * dVexp_ || std::abs(DV_[1]) < th_grad_) {
      acceptstep_ = true;
    }
  } else {
    if (feas_ <= th_stop_) {
      if (dPhi_ > th_acceptnegstep_ * dPhiexp_) {
        acceptstep_ = true;
      }
    } else if (dV_ > th_acceptnegstep_ * dVexp_) {
      acceptstep_ = true;
    }
  }
  // TODO: accept dImpr > 0 when allocated time has been reached (c++)
  if (steplength_ <= th_acceptminstep_ && dImpr_ > Scalar(0.)) {
    acceptstep_ = true;
  }
  return acceptstep_;
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::updateMeritFunction() {
  // Update the penalty parameter for computing the merit function and its
  // directional derivative For more details see Section 3 of "An Interior
  // Point Algorithm for Large Scale Nonlinear Programming"
  if (iter_ == 0 && zero_upsilon_) {
    upsilon_ = 0.;
  }
  if (feas_ >= th_minfeas_) {
    // We incorporate a barrier-reduction strategy that still maintains a the
    // directional derivative be sufficiently negative (as explained in
    // Nocedal's texbook page 542) while allowing for a reduction when it is
    // possible.
    upsilon_ = std::max(upsilon_ * upsilon_decfactor_,
                        dVexp_full_ / ((Scalar(1.) - rho_) * feas_));
  }
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::computeCandidate(const Scalar steplength) {
  START_PROFILER("SolverOdynSQP::computeCandidate");
  if (steplength > Scalar(1.) || steplength < Scalar(0.)) {
    throw_pretty("Invalid argument: "
                 << "invalid step length, value is between 0. to 1.");
  }
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract>>& datas =
      problem_->get_runningDatas();
  // Update the dynamics gap for each node
  models[0]->get_state()->integrate(xs_[0], steplength * dxs_[0], xs_try_[0]);
  fs_try_[0] = fs_[0] * (1 - steplength);
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& m = models[t];
    m->get_state()->integrate(xs_[t + 1], steplength * dxs_[t + 1],
                              xs_try_[t + 1]);
  }
#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(problem_->get_nthreads())
#endif
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& m = models[t];
    const std::shared_ptr<ActionDataAbstract>& d = datas[t];
    if (m->get_nu() != 0) {
      us_try_[t] = us_[t] + steplength * dus_[t];
      m->calc(d, xs_try_[t], us_try_[t]);
    } else {
      m->calc(d, xs_try_[t]);
    }
    m->get_state()->diff(xs_try_[t + 1], d->xnext, fs_try_[t + 1]);
  }
  cost_try_ = Scalar(0.);
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionDataAbstract>& d = datas[t];
    cost_try_ += d->cost;
    if (raiseIfNaN(cost_try_)) {
      STOP_PROFILER("SolverOdynSQP::computeCandidate");
      throw_pretty("computeCandidate");
    }
  }
  const std::shared_ptr<ActionModelAbstract>& m = problem_->get_terminalModel();
  const std::shared_ptr<ActionDataAbstract>& d = problem_->get_terminalData();
  m->calc(d, xs_try_.back());
  cost_try_ += d->cost;
  if (raiseIfNaN(cost_try_)) {
    STOP_PROFILER("SolverOdynSQP::computeCandidate");
    throw_pretty("computeCandidate");
  }
  STOP_PROFILER("SolverOdynSQP::computeCandidate");
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::computeQuadraticModel() {
  START_PROFILER("SolverOdynSQP::computeQuadraticModel");
  auto addBlock = [=](std::vector<Eigen::Triplet<Scalar>>& T, std::size_t i0,
                      std::size_t j0, const MatrixXs& M,
                      Scalar eps = Scalar(0)) {
    const std::size_t r = static_cast<std::size_t>(M.rows());
    const std::size_t c = static_cast<std::size_t>(M.cols());
    for (std::size_t j = 0; j < c; ++j) {
      for (std::size_t i = 0; i < r; ++i) {
        const Scalar v = M(i, j);
        if (v != eps) {
          T.emplace_back(i0 + i, j0 + j, v);
        }
      }
    }
  };
  auto addIdentity = [=](std::vector<Eigen::Triplet<Scalar>>& T, std::size_t i0,
                         std::size_t j0, std::size_t n,
                         Scalar scale = Scalar(1.0)) {
    T.reserve(T.size() + n);
    for (std::size_t k = 0; k < n; ++k) {
      T.emplace_back(i0 + k, j0 + k, scale);
    }
  };
  const std::size_t T = problem_->get_T();
  // Update the state and control indeces in the first iteration
  if (iter_) {
    updateStateAndControlIndex();
  }
  // Resize QP model
  qp_model_.conservativeResize(n_, m_, p_);
  // Building the local QP model
  std::vector<Eigen::Triplet<Scalar>> TQ, TA, TG;
  const std::size_t ndx = problem_->get_ndx();
  std::size_t eq_idx = 0;
  std::size_t ineq_idx = 0;
  addIdentity(TA, eq_idx, xs_idx_[0], ndx, Scalar(1.0));
  qp_model_.b.segment(eq_idx, ndx) = fs_[0];
  eq_idx += ndx;
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model =
        problem_->get_runningModels()[t];
    const std::shared_ptr<ActionDataAbstract>& data =
        problem_->get_runningDatas()[t];
    const std::size_t nu = model->get_nu();
    const std::size_t nh = model->get_nh();
    const std::size_t ng = model->get_ng();
    const std::size_t xp_idx = xs_idx_[t + 1];
    const std::size_t x_idx = xs_idx_[t];
    const std::size_t u_idx = us_idx_[t];
    // Quadratic cost
    addBlock(TQ, x_idx, x_idx, data->Lxx);
    qp_model_.c.segment(x_idx, ndx) = data->Lx;
    if (nu > 0) {
      addBlock(TQ, u_idx, u_idx, data->Luu);
      addBlock(TQ, x_idx, u_idx, data->Lxu);
      addBlock(TQ, u_idx, x_idx, data->Lxu.transpose());
      qp_model_.c.segment(u_idx, nu) = data->Lu;
    }
    // Dynamics equalities: Fx dx + Fu du - dxp = -f
    addBlock(TA, eq_idx, x_idx, data->Fx);
    if (nu > 0) {
      addBlock(TA, eq_idx, u_idx, data->Fu);
    }
    addIdentity(TA, eq_idx, xp_idx, ndx, Scalar(-1.0));
    qp_model_.b.segment(eq_idx, ndx) = -fs_[t + 1];
    eq_idx += ndx;
    // State-control equalities: Hx dx + Hu du = -h
    if (nh > 0) {
      addBlock(TA, eq_idx, x_idx, data->Hx);
      if (nu > 0) {
        addBlock(TA, eq_idx, u_idx, data->Hu);
      }
      qp_model_.b.segment(eq_idx, nh) = -data->h;
      eq_idx += nh;
    }
    // State-control inequalities: g_lb - g <= Gx dx + Gu du <= g_ub - g. This
    // include the bounded residuals only.
    if (ng > 0) {
      // Upper side: Gx dx + Gu du <= g_ub - g
      addBlock(TG, ineq_idx, x_idx, data->Gx);
      if (nu > 0) {
        addBlock(TG, ineq_idx, u_idx, data->Gu);
      }
      qp_model_.h.segment(ineq_idx, ng) = model->get_g_ub() - data->g;
      ineq_idx += ng;
      // Lower side: -Gx dx - Gu du <= g - g_lb
      addBlock(TG, ineq_idx, x_idx, -data->Gx);
      if (nu > 0) {
        addBlock(TG, ineq_idx, u_idx, -data->Gu);
      }
      qp_model_.h.segment(ineq_idx, ng) = data->g - model->get_g_lb();
      ineq_idx += ng;
    }
    // State bounds: x_lb - x <= dx <= x_ub - x.
    if (t > 0 && t < T - 1) {
      // Upper bound: dx <= x_ub - x
      addIdentity(TG, ineq_idx, x_idx, ndx, Scalar(1));
      model->get_state()->safe_diff(xs_[t], model->get_state()->get_ub(),
                                    qp_model_.h.segment(ineq_idx, ndx));
      ineq_idx += ndx;
      // Lower bound: -dx <= x - x_lb
      addIdentity(TG, ineq_idx, x_idx, ndx, Scalar(-1));
      model->get_state()->safe_diff(model->get_state()->get_lb(), xs_[t],
                                    qp_model_.h.segment(ineq_idx, ndx));
      ineq_idx += ndx;
    }
    // Control bounds: u_lb - u <= du <= u_ub - u. This include the bounded
    // residuals only.
    if (nu > 0) {
      // Upper bound: du <= u_ub - u
      addIdentity(TG, ineq_idx, u_idx, nu, Scalar(1));
      qp_model_.h.segment(ineq_idx, nu) = model->get_u_ub() - us_[t];
      ineq_idx += nu;
      // Lower bound: -du <= u - u_lb
      addIdentity(TG, ineq_idx, u_idx, nu, Scalar(-1));
      qp_model_.h.segment(ineq_idx, nu) = us_[t] - model->get_u_lb();
      ineq_idx += nu;
    }
  }
  const std::shared_ptr<ActionModelAbstract>& model_T =
      problem_->get_terminalModel();
  const std::shared_ptr<ActionDataAbstract>& data_T =
      problem_->get_terminalData();
  const std::size_t nh_T = model_T->get_nh_T();
  const std::size_t ng_T = model_T->get_ng_T();
  const std::size_t x_idx = xs_idx_[T];
  // Terminal cost and regularization
  addBlock(TQ, x_idx, x_idx, data_T->Lxx);
  qp_model_.c.segment(x_idx, ndx) = data_T->Lx;
  if (preg_ != Scalar(0)) {
    for (std::size_t k = 0; k < n_; ++k) {
      TQ.emplace_back(k, k, preg_);
    }
  }
  // Terminal equalities: Hx_T dx_T = -h_T
  if (nh_T > 0) {
    addBlock(TA, eq_idx, x_idx, data_T->Hx);
    qp_model_.b.segment(eq_idx, nh_T) = -data_T->h.head(nh_T);
    eq_idx += nh_T;
  }
  // Terminal inequalities: g_lb - g <= Gx dx <= g_ub - g
  if (ng_T > 0) {
    // Upper side: Gx dx <= g_ub - g
    addBlock(TG, ineq_idx, x_idx, data_T->Gx);
    qp_model_.h.segment(ineq_idx, ng_T) = model_T->get_g_ub() - data_T->g;
    ineq_idx += ng_T;
    // Lower side: -Gx dx <= g - g_lb
    addBlock(TG, ineq_idx, x_idx, -data_T->Gx);
    qp_model_.h.segment(ineq_idx, ng_T) = data_T->g - model_T->get_g_lb();
    ineq_idx += ng_T;
  }
  // Finalize sparse matrices
  qp_model_.Q.setFromTriplets(TQ.begin(), TQ.end());
  qp_model_.A.setFromTriplets(TA.begin(), TA.end());
  qp_model_.G.setFromTriplets(TG.begin(), TG.end());
  qp_model_.Q.makeCompressed();
  qp_model_.A.makeCompressed();
  qp_model_.G.makeCompressed();
  // Update the QP model and resize its data
  qp_model_.cropInequalities();
  qp_data_.conservativeResize(qp_model_);
  STOP_PROFILER("SolverOdynSQP::computeQuadraticModel");
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::updateCandidate() {
  cost_ = cost_try_;
  ffeas_ = ffeas_try_;
  gfeas_ = gfeas_try_;
  hfeas_ = hfeas_try_;
  merit_ = cost_ + upsilon_ * (ffeas_ + gfeas_ + hfeas_);
}

template <typename Scalar>
bool SolverOdynSQPTpl<Scalar>::decreaseRegularizationCriteria() {
  return (steplength_ >= th_stepdec_ && std::abs(dImpr_) > th_minimprove_);
}

template <typename Scalar>
bool SolverOdynSQPTpl<Scalar>::increaseRegularizationCriteria() {
  return ((steplength_ >= th_stepinc_ && std::abs(dImpr_) <= th_minimprove_) ||
          !acceptstep_);
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::decreaseRegularization() {
  preg_ /= reg_decfactor_;
  if (preg_ < reg_min_) {
    preg_ = reg_min_;
  }
  dreg_ = preg_;
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::increaseRegularization() {
  preg_ *= reg_incfactor_;
  if (preg_ > reg_max_) {
    preg_ = reg_max_;
  }
  dreg_ = preg_;
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::extractQpDirection(const VectorXs& x) {
  const std::size_t T = problem_->get_T();
  const std::size_t ndx = problem_->get_ndx();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model =
        problem_->get_runningModels()[t];
    const std::size_t nu = model->get_nu();
    dxs_[t] = x.segment(xs_idx_[t], ndx);
    if (nu > 0) {
      dus_[t] = x.segment(us_idx_[t], nu);
    }
  }
  dxs_.back() = x.segment(xs_idx_[T], ndx);
}

template <typename Scalar>
template <typename NewScalar>
SolverOdynSQPTpl<NewScalar> SolverOdynSQPTpl<Scalar>::cast() const {
  typedef SolverOdynSQPTpl<NewScalar> ReturnType;
  typedef ShootingProblemTpl<NewScalar> ProblemType;
  ReturnType ret(
      std::make_shared<ProblemType>(problem_->template cast<NewScalar>()));
  // Setting the abstract parameters
  ret.setCallbacks(vector_cast<NewScalar>(callbacks_));
  ret.set_th_acceptstep(scalar_cast<NewScalar>(th_acceptstep_));
  ret.set_th_gaptol(scalar_cast<NewScalar>(th_gaptol_));
  ret.set_feasnorm(feasnorm_);
  ret.set_th_stop(
      std::sqrt(std::numeric_limits<NewScalar>::epsilon()) < NewScalar(th_stop_)
          ? scalar_cast<NewScalar>(th_stop_)
          : std::sqrt(
                std::numeric_limits<NewScalar>::
                    epsilon()));  // Stopping threshold shouldn't be lower than
                                  // square root of the machine precision
  // Setting the OdynSQP parameters
  ret.set_alphas(vector_cast<NewScalar>(alphas_));
  ret.set_reg_incfactor(scalar_cast<NewScalar>(reg_incfactor_));
  ret.set_reg_decfactor(scalar_cast<NewScalar>(reg_decfactor_));
  ret.set_reg_min(
      ScaleNumerics<Scalar>(1e-9) < NewScalar(reg_min_)
          ? scalar_cast<NewScalar>(reg_min_)
          : ScaleNumerics<NewScalar>(
                1e-9));  // Minimum regularization value shouldn't be lower than
                         // 1e-9 or 1e-5 for doubles or floats
  ret.set_reg_max(
      ScaleNumerics<Scalar>(1e9, 1e-4) > NewScalar(reg_max_)
          ? scalar_cast<NewScalar>(reg_max_)
          : ScaleNumerics<NewScalar>(
                1e9, 1e-4));  // Maximum regularization value shouldn't be
                              // higher than 1e9 or 1e5 for doubles or floats
  ret.set_th_grad(scalar_cast<NewScalar>(ScaleNumerics<NewScalar>(th_grad_)));
  ret.set_th_noimprovement(scalar_cast<NewScalar>(th_noimprovement_));
  ret.set_th_stepdec(scalar_cast<NewScalar>(th_stepdec_));
  ret.set_th_stepinc(scalar_cast<NewScalar>(th_stepinc_));
  ret.set_th_minimprove(scalar_cast<NewScalar>(th_minimprove_));
  ret.set_th_acceptnegstep(scalar_cast<NewScalar>(th_acceptnegstep_));
  ret.set_th_acceptminstep(scalar_cast<NewScalar>(th_acceptminstep_));
  ret.set_rho(scalar_cast<NewScalar>(rho_));
  ret.set_th_minfeas(scalar_cast<NewScalar>(th_minfeas_));
  ret.set_upsilon_decfactor(scalar_cast<NewScalar>(upsilon_decfactor_));
  ret.set_zero_upsilon(zero_upsilon_);
  ret.set_qp_params(qp_params_.template cast<NewScalar>());
  ret.setCandidate(vector_cast<NewScalar>(xs_), vector_cast<NewScalar>(us_),
                   is_feasible_);
  return ret;
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::allocateData() {
  const std::size_t ndx = problem_->get_ndx();
  const std::size_t T = problem_->get_T();
  Lxx_dx_.resize(T + 1);
  Luu_du_.resize(T);
  Lxu_du_.resize(T);
  xs_idx_.resize(T + 1);
  us_idx_.resize(T);
  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
      problem_->get_runningModels();
  n_ = 0;
  m_ = ndx;
  p_ = 0;
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    const std::size_t nh = model->get_nh();
    const std::size_t ng = model->get_ng();
    Lxx_dx_[t] = VectorXs::Zero(ndx);
    Luu_du_[t] = VectorXs::Zero(nu);
    Lxu_du_[t] = VectorXs::Zero(nu);
    n_ += ndx + nu;
    m_ += ndx + nh;
    p_ += ng + nu + ndx;
  }
  Lxx_dx_.back() = VectorXs::Zero(ndx);
  const std::shared_ptr<ActionModelAbstract>& model_T =
      problem_->get_terminalModel();
  const std::size_t nh_T = model_T->get_nh_T();
  const std::size_t ng_T = model_T->get_ng_T();
  n_ += ndx;
  m_ += nh_T;
  p_ += ng_T + ndx;
  p_ *= 2;
  // Store xs and us indeces for decision variables
  updateStateAndControlIndex();
  // Create the Odyn's sparse QP model and data
  qp_model_ = Model(n_, m_, p_);
  qp_data_.conservativeResize(qp_model_);
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::resizeRunningData() {
  START_PROFILER("SolverOdynSQP::resizeRunningData");
  SolverAbstract::resizeRunningData();
  const std::size_t T = problem_->get_T();
  const std::size_t ndx = problem_->get_ndx();
  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
      problem_->get_runningModels();
  n_ = 0;
  m_ = ndx;
  p_ = 0;
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    n_ += ndx + nu;
    m_ += ndx + model->get_nh();
    p_ += model->get_ng() + nu + ndx;
    Luu_du_[t].conservativeResize(nu);
  }
  const std::shared_ptr<ActionModelAbstract>& model_T =
      problem_->get_terminalModel();
  n_ += ndx;
  m_ += model_T->get_nh_T();
  p_ += model_T->get_ng_T() + ndx;
  p_ *= 2;
  // Store xs and us indeces for decision variables
  updateStateAndControlIndex();
  // Store the QP sparse matrices and vectors
  qp_model_.conservativeResize(n_, m_, p_);
  STOP_PROFILER("SolverOdynSQP::resizeRunningData");
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::resizeTerminalData() {
  START_PROFILER("SolverOdynSQP::resizeTerminalData");
  const std::shared_ptr<ActionModelAbstract>& model_T =
      problem_->get_terminalModel();
  m_ += model_T->get_nh_T() - nh_T_;
  p_ += model_T->get_ng_T() - ng_T_;
  p_ *= 2;
  // Store the QP sparse matrices and vectors
  qp_model_.conservativeResize(n_, m_, p_);
  STOP_PROFILER("SolverOdynSQP::resizeTerminalData");
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::updateStateAndControlIndex() {
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
      problem_->get_runningModels();
  std::size_t nvar = 0;
  const std::size_t ndx = problem_->get_ndx();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    nvar += ndx;
    if (nu > 0) {
      us_idx_[t] = nvar;
      nvar += nu;
    }
    xs_idx_[t + 1] = nvar;
  }
}

template <typename Scalar>
odyn::SparseModelTpl<Scalar>& SolverOdynSQPTpl<Scalar>::qp_model() noexcept {
  return qp_model_;
}

template <typename Scalar>
odyn::SparseDataTpl<Scalar>& SolverOdynSQPTpl<Scalar>::qp_data() noexcept {
  return qp_data_;
}

template <typename Scalar>
odyn::ParamsTpl<Scalar>& SolverOdynSQPTpl<Scalar>::qp_params() noexcept {
  return qp_params_;
}

template <typename Scalar>
odyn::VerboseLevel SolverOdynSQPTpl<Scalar>::get_verbose_level()
    const noexcept {
  return verbose_level_;
}

template <typename Scalar>
std::size_t SolverOdynSQPTpl<Scalar>::get_n() const noexcept {
  return n_;
}

template <typename Scalar>
std::size_t SolverOdynSQPTpl<Scalar>::get_m() const noexcept {
  return m_;
}

template <typename Scalar>
std::size_t SolverOdynSQPTpl<Scalar>::get_p() const noexcept {
  return p_;
}

template <typename Scalar>
Scalar SolverOdynSQPTpl<Scalar>::get_reg_incfactor() const {
  return reg_incfactor_;
}

template <typename Scalar>
Scalar SolverOdynSQPTpl<Scalar>::get_reg_decfactor() const {
  return reg_decfactor_;
}

template <typename Scalar>
Scalar SolverOdynSQPTpl<Scalar>::get_th_grad() const {
  return th_grad_;
}

template <typename Scalar>
Scalar SolverOdynSQPTpl<Scalar>::get_th_stepdec() const {
  return th_stepdec_;
}

template <typename Scalar>
Scalar SolverOdynSQPTpl<Scalar>::get_th_stepinc() const {
  return th_stepinc_;
}

template <typename Scalar>
Scalar SolverOdynSQPTpl<Scalar>::get_th_minimprove() const {
  return th_minimprove_;
}

template <typename Scalar>
Scalar SolverOdynSQPTpl<Scalar>::get_th_acceptnegstep() const {
  return th_acceptnegstep_;
}

template <typename Scalar>
Scalar SolverOdynSQPTpl<Scalar>::get_th_acceptminstep() const {
  return th_acceptminstep_;
}

template <typename Scalar>
Scalar SolverOdynSQPTpl<Scalar>::get_rho() const {
  return rho_;
}

template <typename Scalar>
Scalar SolverOdynSQPTpl<Scalar>::get_th_minfeas() const {
  return th_minfeas_;
}

template <typename Scalar>
Scalar SolverOdynSQPTpl<Scalar>::get_upsilon() const {
  return upsilon_;
}

template <typename Scalar>
Scalar SolverOdynSQPTpl<Scalar>::get_upsilon_decfactor() const {
  return upsilon_decfactor_;
}

template <typename Scalar>
bool SolverOdynSQPTpl<Scalar>::get_zero_upsilon() const {
  return zero_upsilon_;
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::set_qp_params(
    const odyn::ParamsTpl<Scalar>& qp_params) {
  qp_params_ = qp_params;
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::set_verbose_level(
    const odyn::VerboseLevel verbose_level) {
  verbose_level_ = verbose_level;
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::set_reg_incfactor(const Scalar regfactor) {
  if (regfactor <= Scalar(1.)) {
    throw_pretty(
        "Invalid argument: " << "reg_incfactor value is higher than 1.");
  }
  reg_incfactor_ = regfactor;
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::set_reg_decfactor(const Scalar regfactor) {
  if (regfactor <= Scalar(1.)) {
    throw_pretty(
        "Invalid argument: " << "reg_decfactor value is higher than 1.");
  }
  reg_decfactor_ = regfactor;
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::set_th_grad(const Scalar th_grad) {
  if (Scalar(0.) > th_grad) {
    throw_pretty("Invalid argument: " << "th_grad value has to be positive.");
  }
  th_grad_ = th_grad;
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::set_th_noimprovement(
    const Scalar th_noimprovement) {
  if (Scalar(0.) > th_noimprovement) {
    throw_pretty(
        "Invalid argument: " << "th_noimprovement value has to be positive.");
  }
  th_noimprovement_ = th_noimprovement;
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::set_th_stepdec(const Scalar th_stepdec) {
  if (Scalar(0.) >= th_stepdec || th_stepdec > Scalar(1.)) {
    throw_pretty(
        "Invalid argument: " << "th_stepdec value should between 0 and 1.");
  }
  th_stepdec_ = th_stepdec;
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::set_th_stepinc(const Scalar th_stepinc) {
  if (Scalar(0.) >= th_stepinc || th_stepinc > Scalar(1.)) {
    throw_pretty(
        "Invalid argument: " << "th_stepinc value should between 0 and 1.");
  }
  th_stepinc_ = th_stepinc;
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::set_th_minimprove(const Scalar th_minimprove) {
  if (Scalar(0.) >= th_minimprove || th_minimprove > Scalar(100.)) {
    throw_pretty("Invalid argument: "
                 << "th_minimprove value should between 0 and 100.");
  }
  th_minimprove_ = th_minimprove;
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::set_th_acceptnegstep(
    const Scalar th_acceptnegstep) {
  if (Scalar(0.) > th_acceptnegstep) {
    throw_pretty(
        "Invalid argument: " << "th_acceptnegstep value has to be positive.");
  }
  th_acceptnegstep_ = th_acceptnegstep;
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::set_th_acceptminstep(
    const Scalar th_acceptminstep) {
  if (Scalar(0.) > th_acceptminstep || th_acceptminstep > Scalar(1.)) {
    throw_pretty("Invalid argument: "
                 << "th_acceptminstep value should be between 0 and 1.");
  }
  th_acceptminstep_ = th_acceptminstep;
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::set_rho(const Scalar rho) {
  if (Scalar(0.) >= rho || rho > Scalar(1.)) {
    throw_pretty("Invalid argument: " << "rho value should between 0 and 1.");
  }
  rho_ = rho;
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::set_th_minfeas(const Scalar th_minfeas) {
  th_minfeas_ = th_minfeas;
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::set_upsilon_decfactor(
    const Scalar upsilon_decfactor) {
  if (Scalar(0.) >= upsilon_decfactor || upsilon_decfactor > Scalar(1.)) {
    throw_pretty("Invalid argument: "
                 << "upsilon_decfactor value should between 0 and 1.");
  }
  upsilon_decfactor_ = upsilon_decfactor;
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::set_zero_upsilon(const bool zero_upsilon) {
  zero_upsilon_ = zero_upsilon;
}

}  // namespace crocoddyl
