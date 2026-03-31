///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, LAAS-CNRS, University of Edinburgh
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
SolverFDDPTpl<Scalar>::SolverFDDPTpl(std::shared_ptr<ShootingProblem> problem,
                                     const DynamicsSolverType dyn_solver,
                                     const EqualitySolverType term_solver)
    : SolverAbstract(problem),
      term_solver_(term_solver),
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
      zero_upsilon_(false) {
  // Allocating the solver's data
  allocateData();
  // Setting the dynamics solver
  switch (dyn_solver) {
    case HybridShoot: {
      const std::size_t Tshoot =
          std::max(std::size_t(1),
                   problem_->get_T() /
                       std::max((std::size_t)3, problem_->get_nthreads()));
      set_dynamics_solver(dyn_solver, Tshoot);
      break;
    }
    default:
      set_dynamics_solver(dyn_solver, 0);
      break;
  }
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::computeDirection(const bool recalc) {
  START_PROFILER("SolverFDDP::computeDirection");
  // Update the batch's derivatives
  if (recalc) {
    SolverAbstract::calcDir();
  }
  // Update the search direction associated with the batch's internal
  // constraints
  backwardPass();
  // Update search direction associated with the batch's constraint-to-go
  // conditions
  const std::size_t nh_T = problem_->get_terminalModel()->get_nh_T();
  if (nh_T != 0) {
    linearRollout();
    batchPass();
    updateDir();
  } else if (dyn_solver_ != DynamicsSolverType::SingleShoot) {
    linearRollout();
  }
  STOP_PROFILER("SolverFDDP::computeDirection");
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::computeCandidate(const Scalar steplength) {
  START_PROFILER("SolverFDDP::computeCandidate");
  // Update primal, dual and slack variables
  forwardPass(steplength);
  updateDualsAndSlacks(steplength);
  STOP_PROFILER("SolverFDDP::computeCandidate");
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::forwardPass(const Scalar steplength) {
  START_PROFILER("SolverFDDP::forwardPass");
  switch (dyn_solver_) {
    case FeasShoot:
      feasShootForwardPass(steplength);
      break;
    case MultiShoot:
      multiShootForwardPass(steplength);
      break;
    case HybridShoot:
      hybridShootForwardPass(steplength);
      break;
    case SingleShoot:
      singleShootForwardPass(steplength);
      break;
    default:
      feasShootForwardPass(steplength);
      break;
  }
  STOP_PROFILER("SolverFDDP::forwardPass");
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::updateDualsAndSlacks(
    const Scalar /**steplength**/) {}

template <typename Scalar>
Scalar SolverFDDPTpl<Scalar>::stoppingCriteria() {
  feas_ = ffeas_ + gfeas_ + hfeas_;
  stop_ =
      std::max(feas_, std::abs(dVexp_full_) / (Scalar(1.) + std::abs(cost_)));
  return stop_;
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::Vector3s
SolverFDDPTpl<Scalar>::expectedImprovement() {
  // We define dVexp = Vexp - Vexptry as done for dV
  const std::size_t T = problem_->get_T();
  DV_.setZero();
  switch (dyn_solver_) {
    case SingleShoot:
      DV_[0] -= fs_.back().dot(Vx_.back());
      DV_[0] -= Scalar(0.5) * fs_.back().dot(Vxx_f_.back());
      for (std::size_t t = 0; t < T; ++t) {
        const std::size_t nu = problem_->get_runningModels()[t]->get_nu();
        if (nu != 0) {
          DV_[1] += k_[t].dot(Qu_[t]);
          DV_[2] -= k_[t].dot(Quuk_[t]);
        }
        DV_[0] -= fs_[t].dot(Vx_[t]);
        DV_[0] -= Scalar(0.5) * fs_[t].dot(Vxx_f_[t]);
      }
      break;
    case FeasShoot:
    case MultiShoot:
    case HybridShoot:
      const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
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
      const std::shared_ptr<ActionDataAbstract>& d =
          problem_->get_terminalData();
      Lxx_dx_.back().noalias() = d->Lxx * dxs_.back();
      DV_[1] -= dxs_.back().dot(d->Lx);
      DV_[2] -= dxs_.back().dot(Lxx_dx_.back());
      break;
  }
  return DV_;
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::computeMeritFunctionImprovement() {
  // In single shooting, we do not consider the dynamics feasibility in the
  // merit function. This is because the dynamics are always satisfied.
  switch (dyn_solver_) {
    case SingleShoot:
      ffeas_ = Scalar(0.);
      ffeas_try_ = Scalar(0.);
      dfeas_ -= ffeas_ - ffeas_try_;
      break;
    default:
      break;
  }
  dPhi_ = dV_ + upsilon_ * dfeas_;
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::computeExpectedMeritFunctionImprovement() {
  dPhiexp_ = dVexp_ + steplength_ * upsilon_ * dfeas_;
}

template <typename Scalar>
bool SolverFDDPTpl<Scalar>::checkAcceptance() {
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
void SolverFDDPTpl<Scalar>::updateMeritFunction() {
  // Update the penalty parameter for computing the merit function and its
  // directional derivative For more details see Section 3 of "An Interior
  // Point Algorithm for Large Scale Nonlinear Programming"
  if (iter_ == 0 && zero_upsilon_) {
    upsilon_ = 0.;
  }
  if (feas_ >= th_minfeas_ && dyn_solver_ != SingleShoot) {
    // We incorporate a barrier-reduction strategy that still maintains a the
    // directional derivative be sufficiently negative (as explained in
    // Nocedal's texbook page 542) while allowing for a reduction when it is
    // possible.
    upsilon_ = std::max(upsilon_ * upsilon_decfactor_,
                        dVexp_full_ / ((Scalar(1.) - rho_) * feas_));
  }
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::resizeRunningData() {
  START_PROFILER("SolverFDDP::resizeRunningData");
  SolverAbstract::resizeRunningData();
  const std::size_t T = problem_->get_T();
  const std::size_t ndx = problem_->get_ndx();
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    Luu_du_[t].conservativeResize(nu);
    Qxu_[t].conservativeResize(ndx, nu);
    Quu_[t].conservativeResize(nu, nu);
    Qu_[t].conservativeResize(nu);
    K_[t].conservativeResize(nu, ndx);
    k_[t].conservativeResize(nu);
    FuTVxx_p_[t].conservativeResize(nu, ndx);
    Quuk_[t].conservativeResize(nu);
    if (nu != 0) {
      FuTVxx_p_[t].setZero();
    }
  }
  STOP_PROFILER("SolverFDDP::resizeRunningData");
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::resizeTerminalData() {
  START_PROFILER("SolverFDDP::resizeTerminalData");
  const std::size_t T = problem_->get_T();
  const std::size_t ndx = problem_->get_ndx();
  const std::size_t nh_T = problem_->get_terminalModel()->get_nh_T();
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    Vxc_[t].conservativeResize(ndx, nh_T);
    Qxc_[t].conservativeResize(ndx, nh_T);
    Quc_[t].conservativeResize(nu, nh_T);
    dXc_[t].conservativeResize(ndx, nh_T);
    dUc_[t].conservativeResize(nu, nh_T);
    Kc_[t].conservativeResize(nu, nh_T);
  }
  Vxc_.back().conservativeResize(ndx, nh_T);
  dXc_.back().conservativeResize(ndx, nh_T);
  dHc_.conservativeResize(nh_T, nh_T);
  hc_.conservativeResize(nh_T);
  YZc_.conservativeResize(nh_T, nh_T);
  Yhc_.conservativeResize(nh_T);
  dHcY_.conservativeResize(nh_T, nh_T);
  YdHcY_.conservativeResize(nh_T, nh_T);
  beta_plus_.conservativeResize(nh_T);
  STOP_PROFILER("SolverFDDP::resizeTerminalData");
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::backwardPass() {
  START_PROFILER("SolverFDDP::backwardPass");
  const std::shared_ptr<ActionDataAbstract>& d_T = problem_->get_terminalData();
  Vxx_.back() = d_T->Lxx;
  Vx_.back() = d_T->Lx;
  if (!std::isnan(preg_)) {
    Vxx_.back().diagonal().array() += preg_;
  }
  // Compute and store the Vxx_f gradient
  Vxx_f_.back().noalias() = Vxx_.back() * fs_.back();
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
      problem_->get_runningDatas();
  for (int t = static_cast<int>(problem_->get_T()) - 1; t >= 0; --t) {
    const std::shared_ptr<ActionModelAbstract>& m = models[t];
    const std::shared_ptr<ActionDataAbstract>& d = datas[t];
    // Update action-value function
    computeActionValueFunction(t, m, d);
    // Update policy
    computePolicy(t);
    // Update value function
    computeValueFunction(t, m);
    if (raiseIfNaN(Vx_[t].template lpNorm<Eigen::Infinity>())) {
      throw_pretty("backward_error");
    }
    if (raiseIfNaN(Vxx_[t].template lpNorm<Eigen::Infinity>())) {
      throw_pretty("backward_error");
    }
  }
  STOP_PROFILER("SolverFDDP::backwardPass");
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::batchPass() {
  START_PROFILER("SolverFDDP::batchPass");
  const std::shared_ptr<ActionDataAbstract>& d_T = problem_->get_terminalData();
  const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
      problem_->get_runningDatas();
  Vxc_.back() = -d_T->Hx.transpose();
  for (int t = static_cast<int>(problem_->get_T()) - 1; t >= 0; --t) {
    const std::shared_ptr<ActionDataAbstract>& d = datas[t];
    // Update action-value function associated with the batch's constraint-to-go
    // conditions
    computeBatchActionValueFunction(t, d);
    // Update feed-forward policy associated with the batch's constraint-to-go
    // conditions
    computeBatchPolicy(t);
    // Update value function associated with the batch's constraint-to-go
    // conditions
    computeBatchValueFunction(t);
  }
  dXc_[0].setZero();
  for (std::size_t t = 0; t < problem_->get_T(); ++t) {  // sequence
    const std::shared_ptr<ActionDataAbstract>& d = datas[t];
    dUc_[t] = -Kc_[t];
    dUc_[t].noalias() -= K_[t] * dXc_[t];
    dXc_[t + 1].noalias() = d->Fx * dXc_[t];
    dXc_[t + 1].noalias() += d->Fu * dUc_[t];
  }
  STOP_PROFILER("SolverFDDP::batchPass");
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::updateDir() {
  START_PROFILER("SolverFDDP::updateDir");
  const std::shared_ptr<ActionDataAbstract>& d_T = problem_->get_terminalData();
  dHc_.noalias() = d_T->Hx * dXc_.back();
  hc_ = d_T->h;
  hc_.noalias() += d_T->Hx * dxs_.back();
  switch (term_solver_) {
    // For the LuNull and QrNull solvers, we compute terminal multiplier using
    // nullspace parametrization. Instead of parametrizing Hx, we opt to
    // equivalent parametrize dHc. This approach is much efficient.
    case LuNull: {
      dHc_lu_.compute(dHc_);
      dHc_rank_ = dHc_lu_.rank();
      YZc_.leftCols(dHc_rank_) << dHc_lu_.matrixLU().transpose();
      const std::size_t nh_T = problem_->get_terminalModel()->get_nh_T();
      if (dHc_rank_ < nh_T) {
        YZc_.rightCols(nh_T - dHc_rank_) << dHc_lu_.kernel();
      }
      computeNullTerminalMultiplier();
      break;
    }
    case QrNull: {
      dHc_qr_.compute(dHc_);
      YZc_ = dHc_qr_.householderQ();
      dHc_rank_ = dHc_qr_.rank();
      computeNullTerminalMultiplier();
      break;
    }
    case Schur: {
      YdHcY_llt_.compute(dHc_);
      const Eigen::ComputationInfo& info = YdHcY_llt_.info();
      if (info != Eigen::Success) {
        throw_pretty("backward_error");
      }
      beta_plus_ = hc_;
      YdHcY_llt_.solveInPlace(beta_plus_);
      break;
    }
  }
  // Finally, we update the feed-forward term and search direction.
  for (std::size_t t = 0; t < problem_->get_T(); ++t) {  // parallel
    dus_[t].noalias() -= dUc_[t] * beta_plus_;
    dxs_[t + 1].noalias() -= dXc_[t + 1] * beta_plus_;
    k_[t].noalias() -= Kc_[t] * beta_plus_;
    Quuk_[t].noalias() = Quu_[t] * k_[t];
  }
  STOP_PROFILER("SolverFDDP::updateDir");
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::computeActionValueFunction(
    const std::size_t t, const std::shared_ptr<ActionModelAbstract>& model,
    const std::shared_ptr<ActionDataAbstract>& data) {
  START_PROFILER("SolverFDDP::computeActionValueFunction");
  assert_pretty(t < problem_->get_T(),
                "Invalid argument: t should be between 0 and " +
                    std::to_string(problem_->get_T()););
  const std::size_t nu = model->get_nu();
  const MatrixXs& Vxx_p = Vxx_[t + 1];
  VectorXs& Vx_p = Vx_[t + 1];
  // Update Vx with Vxx f term
  Vx_p += Vxx_f_[t + 1];
  START_PROFILER("SolverFDDP::Qx");
  Qx_[t] = data->Lx;
  Qx_[t].noalias() += data->Fx.transpose() * Vx_p;
  STOP_PROFILER("SolverFDDP::Qx");
  START_PROFILER("SolverFDDP::Qxx");
  FxTVxx_p_[t].noalias() = data->Fx.transpose() * Vxx_p;
  Qxx_[t] = data->Lxx;
  Qxx_[t].noalias() += FxTVxx_p_[t] * data->Fx;
  if (!std::isnan(preg_)) {
    Qxx_[t].diagonal().array() += preg_;
  }
  STOP_PROFILER("SolverFDDP::Qxx");
  if (nu != 0) {
    START_PROFILER("SolverFDDP::Qu");
    Qu_[t] = data->Lu;
    Qu_[t].noalias() += data->Fu.transpose() * Vx_p;
    STOP_PROFILER("SolverFDDP::Qu");
    START_PROFILER("SolverFDDP::Quu");
    FuTVxx_p_[t].noalias() = data->Fu.transpose() * Vxx_p;
    Quu_[t] = data->Luu;
    Quu_[t].noalias() += FuTVxx_p_[t] * data->Fu;
    if (!std::isnan(preg_)) {
      Quu_[t].diagonal().array() += preg_;
    }
    STOP_PROFILER("SolverFDDP::Quu");
    START_PROFILER("SolverFDDP::Qxu");
    Qxu_[t] = data->Lxu;
    Qxu_[t].noalias() += FxTVxx_p_[t] * data->Fu;
    STOP_PROFILER("SolverFDDP::Qxu");
  }
  // Return value
  Vx_p -= Vxx_f_[t + 1];
  STOP_PROFILER("SolverFDDP::computeActionValueFunction");
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::computeBatchActionValueFunction(
    const std::size_t t, const std::shared_ptr<ActionDataAbstract>& data) {
  START_PROFILER("SolverFDDP::computeBatchActionValueFunction");
  Quc_[t].noalias() = data->Fu.transpose() * Vxc_[t + 1];
  Qxc_[t].noalias() = data->Fx.transpose() * Vxc_[t + 1];
  STOP_PROFILER("SolverFDDP::computeBatchActionValueFunction");
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::computePolicy(const std::size_t t) {
  START_PROFILER("SolverFDDP::computePolicy");
  assert_pretty(t < problem_->get_T(),
                "Invalid argument: t should be between 0 and " +
                    std::to_string(problem_->get_T()));
  const std::size_t nu = problem_->get_runningModels()[t]->get_nu();
  if (nu > 0) {
    START_PROFILER("SolverFDDP::Quu_cholesky");
    Quu_llt_[t].compute(Quu_[t]);
    STOP_PROFILER("SolverFDDP::Quu_cholesky");
    const Eigen::ComputationInfo& info = Quu_llt_[t].info();
    if (info != Eigen::Success) {
      STOP_PROFILER("SolverFDDP::computePolicy");
      throw_pretty("backward_error");
    }
    START_PROFILER("SolverFDDP::feedback");
    K_[t] = Qxu_[t].transpose();
    Quu_llt_[t].solveInPlace(K_[t]);
    STOP_PROFILER("SolverFDDP::feedback");
    START_PROFILER("SolverFDDP::feedforward");
    k_[t] = Qu_[t];
    Quu_llt_[t].solveInPlace(k_[t]);
    STOP_PROFILER("SolverFDDP::feedforward");
  }
  STOP_PROFILER("SolverFDDP::computePolicy");
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::computeBatchPolicy(const std::size_t t) {
  START_PROFILER("SolverFDDP::computeBatchPolicy");
  Kc_[t] = Quc_[t];
  const std::size_t nu = problem_->get_runningModels()[t]->get_nu();
  if (nu > 0) {
    Quu_llt_[t].solveInPlace(Kc_[t]);
  }
  STOP_PROFILER("SolverFDDP::computeBatchPolicy");
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::computeValueFunction(
    const std::size_t t, const std::shared_ptr<ActionModelAbstract>& model) {
  START_PROFILER("SolverFDDP::computeValueFunction");
  assert_pretty(t < problem_->get_T(),
                "Invalid argument: t should be between 0 and " +
                    std::to_string(problem_->get_T()););
  const std::size_t nu = model->get_nu();
  Vx_[t] = Qx_[t];
  Vxx_[t] = Qxx_[t];
  if (nu != 0) {
    START_PROFILER("SolverFDDP::Vx");
    Quuk_[t].noalias() = Quu_[t] * k_[t];
    Vx_[t].noalias() -= K_[t].transpose() * Qu_[t];
    STOP_PROFILER("SolverFDDP::Vx");
    START_PROFILER("SolverFDDP::Vxx");
    Vxx_[t].noalias() -= Qxu_[t] * K_[t];
    STOP_PROFILER("SolverFDDP::Vxx");
  }
  Vxx_tmp_ = Scalar(0.5) * (Vxx_[t] + Vxx_[t].transpose());
  Vxx_[t] = Vxx_tmp_;
  Vxx_f_[t].noalias() = Vxx_[t] * fs_[t];
  STOP_PROFILER("SolverFDDP::computeValueFunction");
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::computeBatchValueFunction(const std::size_t t) {
  START_PROFILER("SolverFDDP::computeBatchValueFunction");
  Vxc_[t] = Qxc_[t];
  Vxc_[t].noalias() -= Qxu_[t] * Kc_[t];
  STOP_PROFILER("SolverFDDP::computeBatchValueFunction");
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::linearRollout() {
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
      problem_->get_runningDatas();
  dxs_[0] = fs_[0];
  for (std::size_t t = 0; t < T; ++t) {  // in sequence
    const std::shared_ptr<ActionModelAbstract>& m = models[t];
    const std::shared_ptr<ActionDataAbstract>& d = datas[t];
    dxs_[t + 1].noalias() = d->Fx * dxs_[t];
    dxs_[t + 1] += fs_[t + 1];
    if (m->get_nu() != 0) {
      dus_[t] = -k_[t];
      dus_[t].noalias() -= K_[t] * dxs_[t];
      dxs_[t + 1].noalias() += d->Fu * dus_[t];
    }
  }
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::feasShootForwardPass(const Scalar steplength) {
  START_PROFILER("SolverFDDP::feasShootForwardPass");
  if (steplength > 1. || steplength < 0.) {
    throw_pretty("Invalid argument: "
                 << "invalid step length, value is between 0. to 1.");
  }
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
      problem_->get_runningDatas();
  cost_try_ = Scalar(0.);
  models[0]->get_state()->integrate(xs_[0], steplength * dxs_[0], xs_try_[0]);
  fs_try_[0] = fs_[0] * (1 - steplength);
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& m = models[t];
    const std::shared_ptr<ActionDataAbstract>& d = datas[t];
    const std::size_t nu = m->get_nu();
    if (nu != 0) {
      m->get_state()->diff(xs_[t], xs_try_[t], dx_[t]);
      us_try_[t] = us_[t] - steplength * k_[t];
      us_try_[t].noalias() -= K_[t] * dx_[t];
      m->calc(d, xs_try_[t], us_try_[t]);
    } else {
      m->calc(d, xs_try_[t]);
    }
    fs_try_[t + 1] = fs_[t + 1] * (Scalar(1.) - steplength);
    m->get_state()->integrate(d->xnext, -fs_try_[t + 1], xs_try_[t + 1]);
    cost_try_ += d->cost;
    if (raiseIfNaN(cost_try_)) {
      STOP_PROFILER("SolverFDDP::feasShootForwardPass");
      throw_pretty("forward_error");
    }
  }
  const std::shared_ptr<ActionModelAbstract>& m = problem_->get_terminalModel();
  const std::shared_ptr<ActionDataAbstract>& d = problem_->get_terminalData();
  m->calc(d, xs_try_.back());
  cost_try_ += d->cost;
  if (raiseIfNaN(cost_try_)) {
    STOP_PROFILER("SolverFDDP::feasShootForwardPass");
    throw_pretty("forward_error");
  }
  STOP_PROFILER("SolverFDDP::feasShootForwardPass");
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::multiShootForwardPass(const Scalar steplength) {
  START_PROFILER("SolverFDDP::multiShootForwardPass");
  if (steplength > Scalar(1.) || steplength < Scalar(0.)) {
    throw_pretty("Invalid argument: "
                 << "invalid step length, value is between 0. to 1.");
  }
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
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
      STOP_PROFILER("SolverFDDP::multiShootForwardPass");
      throw_pretty("forward_error");
    }
  }
  const std::shared_ptr<ActionModelAbstract>& m = problem_->get_terminalModel();
  const std::shared_ptr<ActionDataAbstract>& d = problem_->get_terminalData();
  m->calc(d, xs_try_.back());
  cost_try_ += d->cost;
  if (raiseIfNaN(cost_try_)) {
    STOP_PROFILER("SolverFDDP::multiShootForwardPass");
    throw_pretty("forward_error");
  }
  STOP_PROFILER("SolverFDDP::multiShootForwardPass");
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::hybridShootForwardPass(const Scalar steplength) {
  START_PROFILER("SolverFDDP::hybridShootForwardPass");
  if (steplength > Scalar(1.) || steplength < Scalar(0.)) {
    throw_pretty("Invalid argument: "
                 << "invalid step length, value is between 0. to 1.");
  }
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
      problem_->get_runningDatas();
  // Update the initial state of each shooting node
  models[0]->get_state()->integrate(xs_[0], steplength * dxs_[0], xs_try_[0]);
  for (std::size_t i = 1; i < Ts_.size();
       ++i) {  // this can be executed in parallel
    const std::size_t Ti = Ts_[i];
    const std::shared_ptr<ActionModelAbstract>& m = models[Ti - 1];
    m->get_state()->integrate(xs_[Ti], steplength * dxs_[Ti], xs_try_[Ti]);
  }
  // Perform the feasibility-driven nonlinear rollout for each shooting node
#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(problem_->get_nthreads())
#endif
  for (std::size_t i = 1; i < Ts_.size(); ++i) {
    for (std::size_t t = Ts_[i - 1]; t < Ts_[i]; ++t) {
      const std::shared_ptr<ActionModelAbstract>& m = models[t];
      const std::shared_ptr<ActionDataAbstract>& d = datas[t];
      if (m->get_nu() != 0) {
        m->get_state()->diff(xs_[t], xs_try_[t], dx_[t]);
        us_try_[t] = us_[t] - steplength * k_[t];
        us_try_[t].noalias() -= K_[t] * dx_[t];
        m->calc(d, xs_try_[t], us_try_[t]);
      } else {
        m->calc(d, xs_try_[t]);
      }
      if (t + 1 != Ts_[i]) {
        fs_try_[t + 1] = fs_[t + 1] * (Scalar(1.) - steplength);
        m->get_state()->integrate(d->xnext, -fs_try_[t + 1], xs_try_[t + 1]);
      }
    }
  }
  cost_try_ = Scalar(0.);
  for (std::size_t i = 1; i < Ts_.size(); ++i) {
    for (std::size_t t = Ts_[i - 1]; t < Ts_[i]; ++t) {
      const std::shared_ptr<ActionDataAbstract>& d = datas[t];
      cost_try_ += d->cost;
      if (raiseIfNaN(cost_try_)) {
        STOP_PROFILER("SolverFDDP::hybridShootForwardPass");
        throw_pretty("forward_error");
      }
    }
  }
  const std::shared_ptr<ActionModelAbstract>& m = problem_->get_terminalModel();
  const std::shared_ptr<ActionDataAbstract>& d = problem_->get_terminalData();
  m->calc(d, xs_try_.back());
  cost_try_ += d->cost;
  if (raiseIfNaN(cost_try_)) {
    STOP_PROFILER("SolverFDDP::hybridShootForwardPass");
    throw_pretty("forward_error");
  }
  // Update the initial gap of each shooting node
  fs_try_[0] = fs_[0] * (Scalar(1.) - steplength);
  for (std::size_t i = 1; i < Ts_.size();
       ++i) {  // this can be executed in parallel
    const std::size_t Ti = Ts_[i];
    const std::shared_ptr<ActionModelAbstract>& m = models[Ti - 1];
    const std::shared_ptr<ActionDataAbstract>& d = datas[Ti - 1];
    m->get_state()->diff(xs_try_[Ti], d->xnext, fs_try_[Ti]);
  }
  STOP_PROFILER("SolverFDDP::hybridShootForwardPass");
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::singleShootForwardPass(const Scalar steplength) {
  if (steplength > Scalar(1.) || steplength < Scalar(0.)) {
    throw_pretty("Invalid argument: "
                 << "invalid step length, value is between 0. to 1.");
  }
  START_PROFILER("SolverFDDP::singleShootForwardPass");
  cost_try_ = Scalar(0.);
  xs_try_[0] = problem_->get_x0();
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
      problem_->get_runningDatas();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& m = models[t];
    const std::shared_ptr<ActionDataAbstract>& d = datas[t];
    if (m->get_nu() != 0) {
      m->get_state()->diff(xs_[t], xs_try_[t], dx_[t]);
      us_try_[t] = us_[t] - steplength * k_[t];
      us_try_[t].noalias() -= K_[t] * dx_[t];
      m->calc(d, xs_try_[t], us_try_[t]);
    } else {
      m->calc(d, xs_try_[t]);
    }
    xs_try_[t + 1] = d->xnext;
    cost_try_ += d->cost;
    if (raiseIfNaN(cost_try_)) {
      STOP_PROFILER("SolverFDDP::singleShootForwardPass");
      throw_pretty("forward_error");
    }
  }
  const std::shared_ptr<ActionModelAbstract>& m = problem_->get_terminalModel();
  const std::shared_ptr<ActionDataAbstract>& d = problem_->get_terminalData();
  m->calc(d, xs_try_.back());
  cost_try_ += d->cost;
  if (raiseIfNaN(cost_try_)) {
    STOP_PROFILER("SolverFDDP::singleShootForwardPass");
    throw_pretty("forward_error");
  }
  STOP_PROFILER("SolverFDDP::singleShootForwardPass");
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::computeNullTerminalMultiplier() {
  // Compute multiplier using nullspace parametrization. Instead of
  // parametrizing Hx, we opt to equivalent parametrize dHc. This approach
  // is much efficient.
  const Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      Yc = YZc_.leftCols(dHc_rank_);
  Yhc_.noalias() = Yc.transpose() * hc_;
  dHcY_.noalias() = dHc_ * Yc;
  YdHcY_.noalias() = Yc.transpose() * dHcY_;
  YdHcY_llt_.compute(YdHcY_);
  const Eigen::ComputationInfo& info = YdHcY_llt_.info();
  if (info != Eigen::Success) {
    throw_pretty("backward_error");
  }
  YdHcY_llt_.solveInPlace(Yhc_);
  beta_plus_.noalias() = Yc * Yhc_;
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::updateCandidate() {
  cost_ = cost_try_;
  switch (dyn_solver_) {
    case SingleShoot:
      ffeas_ = 0.;
      break;
    default:
      ffeas_ = ffeas_try_;
      break;
  }
  gfeas_ = gfeas_try_;
  hfeas_ = hfeas_try_;
  merit_ = cost_ + upsilon_ * (ffeas_ + gfeas_ + hfeas_);
}

template <typename Scalar>
bool SolverFDDPTpl<Scalar>::decreaseRegularizationCriteria() {
  return (steplength_ >= th_stepdec_ && std::abs(dImpr_) > th_minimprove_);
}

template <typename Scalar>
bool SolverFDDPTpl<Scalar>::increaseRegularizationCriteria() {
  return ((steplength_ >= th_stepinc_ && std::abs(dImpr_) <= th_minimprove_) ||
          !acceptstep_);
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::decreaseRegularization() {
  preg_ /= reg_decfactor_;
  if (preg_ < reg_min_) {
    preg_ = reg_min_;
  }
  dreg_ = preg_;
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::increaseRegularization() {
  preg_ *= reg_incfactor_;
  if (preg_ > reg_max_) {
    preg_ = reg_max_;
  }
  dreg_ = preg_;
}

template <typename Scalar>
template <typename NewScalar>
SolverFDDPTpl<NewScalar> SolverFDDPTpl<Scalar>::cast() const {
  typedef SolverFDDPTpl<NewScalar> ReturnType;
  typedef ShootingProblemTpl<NewScalar> ProblemType;
  ReturnType ret(
      std::make_shared<ProblemType>(problem_->template cast<NewScalar>()),
      dyn_solver_, term_solver_);
  if (dyn_solver_ == HybridShoot && Ts_.size() > 1) {
    ret.set_dynamics_solver(dyn_solver_, Ts_[1] - Ts_[0]);
  }
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
  // Setting the FDDP parameters
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
  ret.setCandidate(vector_cast<NewScalar>(xs_), vector_cast<NewScalar>(us_),
                   is_feasible_);
  return ret;
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::allocateData() {
  const std::size_t ndx = problem_->get_ndx();
  const std::size_t T = problem_->get_T();
  Vxx_tmp_ = MatrixXs::Zero(ndx, ndx);
  Vxx_.resize(T + 1);
  Vxx_f_.resize(T + 1);
  Vx_.resize(T + 1);
  Lxx_dx_.resize(T + 1);
  Luu_du_.resize(T);
  Lxu_du_.resize(T);
  Qxx_.resize(T);
  Qxu_.resize(T);
  Quu_.resize(T);
  Qx_.resize(T);
  Qu_.resize(T);
  K_.resize(T);
  k_.resize(T);
  dx_.resize(T);
  FxTVxx_p_.resize(T);
  FuTVxx_p_.resize(T);
  Quu_llt_.resize(T);
  Quuk_.resize(T);
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    Vxx_[t] = MatrixXs::Zero(ndx, ndx);
    Vxx_f_[t] = VectorXs::Zero(ndx);
    Vx_[t] = VectorXs::Zero(ndx);
    Lxx_dx_[t] = VectorXs::Zero(ndx);
    Luu_du_[t] = VectorXs::Zero(nu);
    Lxu_du_[t] = VectorXs::Zero(ndx);
    Qxx_[t] = MatrixXs::Zero(ndx, ndx);
    Qxu_[t] = MatrixXs::Zero(ndx, nu);
    Quu_[t] = MatrixXs::Zero(nu, nu);
    Qx_[t] = VectorXs::Zero(ndx);
    Qu_[t] = VectorXs::Zero(nu);
    K_[t] = MatrixXsRowMajor::Zero(nu, ndx);
    k_[t] = VectorXs::Zero(nu);
    dx_[t] = VectorXs::Zero(ndx);
    FxTVxx_p_[t] = MatrixXsRowMajor::Zero(ndx, ndx);
    FuTVxx_p_[t] = MatrixXsRowMajor::Zero(nu, ndx);
    Quu_llt_[t] = Eigen::LLT<MatrixXs>(nu);
    Quuk_[t] = VectorXs(nu);
  }
  Vxx_.back() = MatrixXs::Zero(ndx, ndx);
  Vx_.back() = VectorXs::Zero(ndx);
  Lxx_dx_.back() = VectorXs::Zero(ndx);
  fTVxx_p_ = VectorXs::Zero(ndx);
  // Terminal constraint data
  const std::size_t nh_T = problem_->get_terminalModel()->get_nh_T();
  Vxc_.resize(T + 1);
  Qxc_.resize(T);
  Quc_.resize(T);
  dXc_.resize(T + 1);
  dUc_.resize(T);
  Kc_.resize(T);
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    Vxc_[t] = MatrixXs::Zero(ndx, nh_T);
    Qxc_[t] = MatrixXs::Zero(ndx, nh_T);
    Quc_[t] = MatrixXs::Zero(nu, nh_T);
    dXc_[t] = MatrixXs::Zero(ndx, nh_T);
    dUc_[t] = MatrixXs::Zero(nu, nh_T);
    Kc_[t] = MatrixXs::Zero(nu, nh_T);
  }
  Vxc_.back() = MatrixXs::Zero(ndx, nh_T);
  dXc_.back() = MatrixXs::Zero(ndx, nh_T);
  dHc_ = MatrixXs::Zero(nh_T, nh_T);
  hc_ = VectorXs::Zero(nh_T);
  YZc_ = MatrixXs::Zero(nh_T, nh_T);
  Yhc_ = VectorXs::Zero(nh_T);
  dHcY_ = MatrixXs::Zero(nh_T, nh_T);
  YdHcY_ = MatrixXs::Zero(nh_T, nh_T);
  beta_plus_ = VectorXs::Zero(nh_T);
  YdHcY_llt_ = Eigen::LLT<MatrixXs>(nh_T);
  dHc_lu_ = Eigen::FullPivLU<MatrixXs>(nh_T, nh_T);
  dHc_qr_ = Eigen::ColPivHouseholderQR<MatrixXs>(nh_T, nh_T);
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::set_dynamics_solver(const DynamicsSolverType type,
                                                const std::size_t Tshoot) {
  dyn_solver_ = type;
  switch (type) {
    case HybridShoot:
      if (Tshoot == 0) {
        std::cerr << "Warning: the number of nodes per shooting cannot be "
                     "zero. Ignoring this request."
                  << std::endl;
        return;
      }
      Ts_.clear();
      Ts_.push_back(0);
      for (std::size_t i = 0; i < problem_->get_T(); i += Tshoot) {
        if (i + Tshoot < problem_->get_T()) {
          Ts_.push_back(i + Tshoot);
        } else {
          Ts_.push_back(problem_->get_T());
        }
      }
      break;
    default:
      if (Tshoot != 0) {
        std::cerr << "Warning: the number of nodes per shooting is valid for "
                     "hybrid shooting only. Ignoring this request."
                  << std::endl;
      }
      break;
  }
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::set_terminal_solver(const EqualitySolverType type) {
  term_solver_ = type;
}

template <typename Scalar>
DynamicsSolverType SolverFDDPTpl<Scalar>::get_dynamics_solver() const {
  return dyn_solver_;
}

template <typename Scalar>
EqualitySolverType SolverFDDPTpl<Scalar>::get_terminal_solver() const {
  return term_solver_;
}

template <typename Scalar>
Scalar SolverFDDPTpl<Scalar>::get_reg_incfactor() const {
  return reg_incfactor_;
}

template <typename Scalar>
Scalar SolverFDDPTpl<Scalar>::get_reg_decfactor() const {
  return reg_decfactor_;
}

template <typename Scalar>
Scalar SolverFDDPTpl<Scalar>::get_th_grad() const {
  return th_grad_;
}

template <typename Scalar>
Scalar SolverFDDPTpl<Scalar>::get_th_stepdec() const {
  return th_stepdec_;
}

template <typename Scalar>
Scalar SolverFDDPTpl<Scalar>::get_th_stepinc() const {
  return th_stepinc_;
}

template <typename Scalar>
Scalar SolverFDDPTpl<Scalar>::get_th_minimprove() const {
  return th_minimprove_;
}

template <typename Scalar>
Scalar SolverFDDPTpl<Scalar>::get_th_acceptnegstep() const {
  return th_acceptnegstep_;
}

template <typename Scalar>
Scalar SolverFDDPTpl<Scalar>::get_th_acceptminstep() const {
  return th_acceptminstep_;
}

template <typename Scalar>
Scalar SolverFDDPTpl<Scalar>::get_rho() const {
  return rho_;
}

template <typename Scalar>
Scalar SolverFDDPTpl<Scalar>::get_th_minfeas() const {
  return th_minfeas_;
}

template <typename Scalar>
Scalar SolverFDDPTpl<Scalar>::get_upsilon() const {
  return upsilon_;
}

template <typename Scalar>
Scalar SolverFDDPTpl<Scalar>::get_upsilon_decfactor() const {
  return upsilon_decfactor_;
}

template <typename Scalar>
bool SolverFDDPTpl<Scalar>::get_zero_upsilon() const {
  return zero_upsilon_;
}

template <typename Scalar>
const std::vector<std::size_t>& SolverFDDPTpl<Scalar>::get_Ts() const {
  return Ts_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::MatrixXs>&
SolverFDDPTpl<Scalar>::get_Vxx() const {
  return Vxx_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::VectorXs>&
SolverFDDPTpl<Scalar>::get_Vx() const {
  return Vx_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::MatrixXs>&
SolverFDDPTpl<Scalar>::get_Qxx() const {
  return Qxx_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::MatrixXs>&
SolverFDDPTpl<Scalar>::get_Qxu() const {
  return Qxu_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::MatrixXs>&
SolverFDDPTpl<Scalar>::get_Quu() const {
  return Quu_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::VectorXs>&
SolverFDDPTpl<Scalar>::get_Qx() const {
  return Qx_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::VectorXs>&
SolverFDDPTpl<Scalar>::get_Qu() const {
  return Qu_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::MatrixXsRowMajor>&
SolverFDDPTpl<Scalar>::get_K() const {
  return K_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::VectorXs>&
SolverFDDPTpl<Scalar>::get_k() const {
  return k_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::MatrixXs>&
SolverFDDPTpl<Scalar>::get_Vxc() const {
  return Vxc_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::MatrixXs>&
SolverFDDPTpl<Scalar>::get_Qxc() const {
  return Qxc_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::MatrixXs>&
SolverFDDPTpl<Scalar>::get_Quc() const {
  return Quc_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::MatrixXs>&
SolverFDDPTpl<Scalar>::get_dXc() const {
  return dXc_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::MatrixXs>&
SolverFDDPTpl<Scalar>::get_dUc() const {
  return dUc_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::MatrixXs>&
SolverFDDPTpl<Scalar>::get_Kc() const {
  return Kc_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs& SolverFDDPTpl<Scalar>::get_dHc()
    const {
  return dHc_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& SolverFDDPTpl<Scalar>::get_hc()
    const {
  return hc_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
SolverFDDPTpl<Scalar>::get_beta_plus() const {
  return beta_plus_;
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::set_reg_incfactor(const Scalar regfactor) {
  if (regfactor <= Scalar(1.)) {
    throw_pretty(
        "Invalid argument: " << "reg_incfactor value is higher than 1.");
  }
  reg_incfactor_ = regfactor;
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::set_reg_decfactor(const Scalar regfactor) {
  if (regfactor <= Scalar(1.)) {
    throw_pretty(
        "Invalid argument: " << "reg_decfactor value is higher than 1.");
  }
  reg_decfactor_ = regfactor;
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::set_th_grad(const Scalar th_grad) {
  if (Scalar(0.) > th_grad) {
    throw_pretty("Invalid argument: " << "th_grad value has to be positive.");
  }
  th_grad_ = th_grad;
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::set_th_noimprovement(
    const Scalar th_noimprovement) {
  if (Scalar(0.) > th_noimprovement) {
    throw_pretty(
        "Invalid argument: " << "th_noimprovement value has to be positive.");
  }
  th_noimprovement_ = th_noimprovement;
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::set_th_stepdec(const Scalar th_stepdec) {
  if (Scalar(0.) >= th_stepdec || th_stepdec > Scalar(1.)) {
    throw_pretty(
        "Invalid argument: " << "th_stepdec value should between 0 and 1.");
  }
  th_stepdec_ = th_stepdec;
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::set_th_stepinc(const Scalar th_stepinc) {
  if (Scalar(0.) >= th_stepinc || th_stepinc > Scalar(1.)) {
    throw_pretty(
        "Invalid argument: " << "th_stepinc value should between 0 and 1.");
  }
  th_stepinc_ = th_stepinc;
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::set_th_minimprove(const Scalar th_minimprove) {
  if (Scalar(0.) >= th_minimprove || th_minimprove > Scalar(100.)) {
    throw_pretty("Invalid argument: "
                 << "th_minimprove value should between 0 and 100.");
  }
  th_minimprove_ = th_minimprove;
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::set_th_acceptnegstep(
    const Scalar th_acceptnegstep) {
  if (Scalar(0.) > th_acceptnegstep) {
    throw_pretty(
        "Invalid argument: " << "th_acceptnegstep value has to be positive.");
  }
  th_acceptnegstep_ = th_acceptnegstep;
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::set_th_acceptminstep(
    const Scalar th_acceptminstep) {
  if (Scalar(0.) > th_acceptminstep || th_acceptminstep > Scalar(1.)) {
    throw_pretty("Invalid argument: "
                 << "th_acceptminstep value should be between 0 and 1.");
  }
  th_acceptminstep_ = th_acceptminstep;
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::set_rho(const Scalar rho) {
  if (Scalar(0.) >= rho || rho > Scalar(1.)) {
    throw_pretty("Invalid argument: " << "rho value should between 0 and 1.");
  }
  rho_ = rho;
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::set_th_minfeas(const Scalar th_minfeas) {
  th_minfeas_ = th_minfeas;
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::set_upsilon_decfactor(
    const Scalar upsilon_decfactor) {
  if (Scalar(0.) >= upsilon_decfactor || upsilon_decfactor > Scalar(1.)) {
    throw_pretty("Invalid argument: "
                 << "upsilon_decfactor value should between 0 and 1.");
  }
  upsilon_decfactor_ = upsilon_decfactor;
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::set_zero_upsilon(const bool zero_upsilon) {
  zero_upsilon_ = zero_upsilon;
}

}  // namespace crocoddyl
