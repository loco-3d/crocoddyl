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
    : SolverFDDPTpl(std::static_pointer_cast<ProblemAbstract>(problem),
                    dyn_solver, term_solver, AStateNone) {}

template <typename Scalar>
SolverFDDPTpl<Scalar>::SolverFDDPTpl(std::shared_ptr<ProblemAbstract> problem,
                                     const DynamicsSolverType dyn_solver,
                                     const EqualitySolverType term_solver,
                                     const ArrivalStateSolverType astate_solver)
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
      zero_upsilon_(false),
      astate_solver_(astate_solver),
      n_phases_(0) {
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
    // Keep virtual dispatch for parameter restoration and INTRO preprocessing.
    calcDir();
  }
  // Update the search direction associated with the batch's internal
  // constraints
  backwardPass();
  // Parameter backward pass (only when problem has phases)
  if (n_phases_ > 0) {
    parametrizedBackwardPass();
    // Multi-phase needs dxs at phase boundaries before paramsPass
    if (n_phases_ > 1) {
      linearRollout();
    }
    paramsPass();
  }
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
  // For parameter-estimation problems: update ps_try and push into models
  if (n_phases_ > 0) {
    acceptstep_ = false;
    const std::vector<std::shared_ptr<ParameterPhaseModel>>& params_models =
        problem_->get_paramsModel();
    const std::vector<std::shared_ptr<ParameterPhaseData>>& params_datas =
        problem_->get_paramsData();
    for (std::size_t i = 0; i < n_phases_; ++i) {
      ps_try_[i] = ps_[i] + steplength * dps_[i];
      problem_->update_p(ps_try_[i], i);
      const std::shared_ptr<ConstraintModelManager>& constraints =
          params_models[i]->get_constraints();
      if (constraints != nullptr) {
        if (params_datas[i]->constraints == nullptr || i >= qp_x0_.size() ||
            i >= qp_u0_.size()) {
          throw_pretty("Invalid data: parameter-constraint data is missing");
        }
        params_models[i]->calc(params_datas[i], qp_x0_[i], qp_u0_[i]);
      }
    }
  }
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
      const std::shared_ptr<ActionDataAbstract>& d =
          problem_->get_terminalData();
      Lxx_dx_.back().noalias() = d->Lxx * dxs_.back();
      DV_[1] -= dxs_.back().dot(d->Lx);
      DV_[2] -= dxs_.back().dot(Lxx_dx_.back());
      break;
  }
  if (n_phases_ > 0) {
    if (dyn_solver_ == DynamicsSolverType::SingleShoot) {
      const std::vector<std::size_t>& phase_starts = problem_->get_phase_idxs();
      for (std::size_t t = 0; t < T; ++t) {
        if (P_dp_[t].size() != 0) {
          DV_[1] -= P_dp_[t].dot(Qu_[t]);
          DV_[2] += P_dp_[t].dot(Quuk_[t]);
        }
      }
      for (std::size_t nph = 0; nph < n_phases_; ++nph) {
        const std::size_t t = phase_starts[nph];
        const VectorXs& dx = dxs_[t];
        if (nph == 0) {
          DV_[1] -= dx.dot(Vx_[0]);
          DV_[2] -= dx.dot(Vxx_[0] * dx);
        }
        DV_[1] -= dps_[nph].dot(Vp_phase_[nph]);
        DV_[2] -= Scalar(2.) * dps_[nph].dot(Vpx_phase_[nph] * dx);
        DV_[2] -= dps_[nph].dot(Vpp_phase_[nph] * dps_[nph]);
      }
    } else {
      const std::vector<std::size_t>& phase_ends = problem_->get_phase_edxs();
      const std::vector<std::shared_ptr<ActionDataAbstract>>& datas =
          problem_->get_runningDatas();
      std::size_t tstart = 0;
      for (std::size_t nph = 0; nph < n_phases_; ++nph) {
        const std::size_t tend = phase_ends[nph];
        for (std::size_t t = tstart; t < tend; ++t) {
          const std::shared_ptr<ActionDataAbstract>& dt = datas[t];
          Lpp_dp_[t].noalias() = dt->Lpp * dps_[nph];
          Lpx_dp_[t].noalias() = dt->Lpx.transpose() * dps_[nph];
          DV_[1] -= dps_[nph].dot(dt->Lp);
          DV_[2] -= dps_[nph].dot(Lpp_dp_[t]);
          DV_[2] -= Scalar(2.) * Lpx_dp_[t].dot(dxs_[t]);
          if (dt->Lu.size() > 0) {
            Lpu_dp_[t].noalias() = dt->Lpu.transpose() * dps_[nph];
            DV_[2] -= Scalar(2.) * Lpu_dp_[t].dot(dus_[t]);
          }
        }
        tstart = tend;
      }
      const std::shared_ptr<ActionDataAbstract>& d_T =
          problem_->get_terminalData();
      const std::size_t last = n_phases_ - 1;
      Lpp_dp_.back().noalias() = d_T->Lpp * dps_[last];
      Lpx_dp_.back().noalias() = d_T->Lpx.transpose() * dps_[last];
      DV_[1] -= dps_[last].dot(d_T->Lp);
      DV_[2] -= dps_[last].dot(Lpp_dp_.back());
      DV_[2] -= Scalar(2.) * Lpx_dp_.back().dot(dxs_.back());
    }
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
  if (n_phases_ == 0 && std::abs(dPhi_) <= th_noimprovement_ &&
      std::abs(dPhiexp_) <= th_noimprovement_) {
    // Preserve the legacy no-improvement acceptance for ordinary problems.
    // Parameterized problems follow PDDP's Armijo logic below.
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
  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
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
  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
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
    if (n_phases_ > 0) {
      Qpc_[t].conservativeResize(models[t]->get_np(), nh_T);
      Vpc_[t].conservativeResize(models[t]->get_np(), nh_T);
    }
  }
  Vxc_.back().conservativeResize(ndx, nh_T);
  dXc_.back().conservativeResize(ndx, nh_T);
  if (n_phases_ > 0) {
    Vpc_.back().conservativeResize(problem_->get_terminalModel()->get_np(),
                                   nh_T);
    Vpc_next_.conservativeResize(Vpc_next_.rows(), nh_T);
    for (std::size_t i = 0; i < n_phases_; ++i) {
      Vpc_phase_[i].conservativeResize(ps_[i].size(), nh_T);
      dPc_[i].conservativeResize(ps_[i].size(), nh_T);
      Kpc_[i].conservativeResize(ps_[i].size(), nh_T);
    }
  }
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
  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract>>& datas =
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
  const std::size_t nh_T = problem_->get_terminalModel()->get_nh_T();
  const std::vector<std::shared_ptr<ActionDataAbstract>>& datas =
      problem_->get_runningDatas();
  // ActionData can store the maximum running/terminal constraint layout. Only
  // the first nh_T rows belong to the terminal node.
  Vxc_.back() = -d_T->Hx.topRows(nh_T).transpose();
  if (n_phases_ > 0) {
    Vpc_.back().setZero();  // terminal node has no param-constraint coupling
  }
  if (n_phases_ > 0) {
    const std::vector<std::size_t>& phase_starts = problem_->get_phase_idxs();
    const std::vector<std::size_t>& phase_ends = problem_->get_phase_edxs();
    for (std::size_t phase = n_phases_; phase-- > 0;) {
      const std::size_t np =
          problem_->get_runningModels()[phase_starts[phase]]->get_np();
      Vpc_next_.setZero();
      for (std::size_t t = phase_ends[phase]; t-- > phase_starts[phase];) {
        // Update action-value function associated with the batch's
        // constraint-to-go conditions
        computeBatchActionValueFunction(t, datas[t]);
        // Update feed-forward policy associated with the batch's
        // constraint-to-go conditions
        computeBatchPolicy(t);
        // Update value function associated with the batch's constraint-to-go
        // conditions
        computeBatchValueFunction(t);
        Vpc_next_.topRows(np) = Vpc_[t];
      }
      Vpc_phase_[phase] = Vpc_[phase_starts[phase]];
    }
  } else {
    for (std::size_t t = problem_->get_T(); t-- > 0;) {
      // Update action-value function associated with the batch's
      // constraint-to-go conditions
      computeBatchActionValueFunction(t, datas[t]);
      // Update feed-forward policy associated with the batch's
      // constraint-to-go conditions
      computeBatchPolicy(t);
      // Update value function associated with the batch's constraint-to-go
      // conditions
      computeBatchValueFunction(t);
    }
  }
  dXc_[0].setZero();
  if (n_phases_ > 0) {
    paramsBatchPass();
  }
  // Phase index for forward pass (parameter coupling to dXc)
  const std::vector<std::size_t>* phase_starts_fwd =
      (n_phases_ > 0) ? &problem_->get_phase_idxs() : nullptr;
  std::size_t nph_fwd = 0;
  for (std::size_t t = 0; t < problem_->get_T(); ++t) {  // sequence
    const std::shared_ptr<ActionDataAbstract>& d = datas[t];
    dUc_[t] = -Kc_[t];
    dUc_[t].noalias() -= K_[t] * dXc_[t];
    if (n_phases_ > 0) {
      dUc_[t].noalias() -= P_[t] * dPc_[nph_fwd];
    }
    dXc_[t + 1].noalias() = d->Fx * dXc_[t];
    dXc_[t + 1].noalias() += d->Fu * dUc_[t];
    if (n_phases_ > 0) {
      dXc_[t + 1].noalias() += d->Fp * dPc_[nph_fwd];
      if (nph_fwd + 1 < n_phases_ &&
          (t + 1) == (*phase_starts_fwd)[nph_fwd + 1]) {
        ++nph_fwd;
      }
    }
  }
  STOP_PROFILER("SolverFDDP::batchPass");
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::updateDir() {
  START_PROFILER("SolverFDDP::updateDir");
  const std::shared_ptr<ActionDataAbstract>& d_T = problem_->get_terminalData();
  const std::size_t nh_T = problem_->get_terminalModel()->get_nh_T();
  // ActionData can store the maximum running/terminal constraint layout. Only
  // the first nh_T rows belong to the terminal node.
  const auto Hx_T = d_T->Hx.topRows(nh_T);
  dHc_.noalias() = Hx_T * dXc_.back();
  hc_ = d_T->h.head(nh_T);
  hc_.noalias() += Hx_T * dxs_.back();
  switch (term_solver_) {
    // For the LuNull and QrNull solvers, we compute terminal multiplier using
    // nullspace parametrization. Instead of parametrizing Hx, we opt to
    // equivalent parametrize dHc. This approach is much efficient.
    case LuNull: {
      dHc_lu_.compute(dHc_);
      dHc_rank_ = dHc_lu_.rank();
      YZc_.leftCols(dHc_rank_) << dHc_lu_.matrixLU().transpose();
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
  // Update parameter directions and k_ with terminal constraint multiplier
  if (n_phases_ > 0) {
    const std::vector<std::size_t>& phase_ends = problem_->get_phase_edxs();
    for (std::size_t i = 0; i < n_phases_; ++i) {
      kp_[i].noalias() -= Kpc_[i] * beta_plus_;
      dps_[i].noalias() -= dPc_[i] * beta_plus_;
    }
    // Rebuild P_dp contribution and update k_
    std::size_t tstart = 0;
    for (std::size_t i = 0; i < n_phases_; ++i) {
      for (std::size_t t = tstart; t < phase_ends[i]; ++t) {
        k_[t] -= P_dp_[t];
        P_dp_[t].noalias() = P_[t] * dps_[i];
        k_[t] += P_dp_[t];
      }
      tstart = phase_ends[i];
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
  if (n_phases_ > 0) {
    Qpc_[t].noalias() = data->Fp.transpose() * Vxc_[t + 1];
    Qpc_[t] += Vpc_next_.topRows(data->Fp.cols());
  }
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
  if (n_phases_ > 0) {
    Vpc_[t] = Qpc_[t];
    Vpc_[t].noalias() -= Qpu_[t] * Kc_[t];
  }
  STOP_PROFILER("SolverFDDP::computeBatchValueFunction");
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::linearRollout() {
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract>>& datas =
      problem_->get_runningDatas();
  dxs_[0] = fs_[0];
  // Phase tracking for parameter contribution
  const bool has_params = (n_phases_ > 0);
  const std::vector<std::size_t>* phase_starts =
      has_params ? &problem_->get_phase_idxs() : nullptr;
  std::size_t nph = 0;
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
    // Parameter contribution: Fp * dps
    if (has_params) {
      dxs_[t + 1].noalias() += d->Fp * dps_[nph];
      if (nph + 1 < n_phases_ && (t + 1) == (*phase_starts)[nph + 1]) {
        ++nph;
      }
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
  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract>>& datas =
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
    }
    m->calc(d, xs_try_[t], us_try_[t]);
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
    }
    m->calc(d, xs_try_[t], us_try_[t]);
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
  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract>>& datas =
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
      }
      m->calc(d, xs_try_[t], us_try_[t]);
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
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract>>& datas =
      problem_->get_runningDatas();
  if (n_phases_ > 0 && astate_solver_ != AStateNone &&
      astate_solver_ != AStateQP) {
    models[0]->get_state()->integrate(xs_[0], steplength * dxs_[0], xs_try_[0]);
  } else {
    xs_try_[0] = problem_->get_x0();
  }
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& m = models[t];
    const std::shared_ptr<ActionDataAbstract>& d = datas[t];
    if (m->get_nu() != 0) {
      m->get_state()->diff(xs_[t], xs_try_[t], dx_[t]);
      us_try_[t] = us_[t] - steplength * k_[t];
      us_try_[t].noalias() -= K_[t] * dx_[t];
    }
    m->calc(d, xs_try_[t], us_try_[t]);
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
  // Accept the parameter candidate on step acceptance
  if (n_phases_ > 0) {
    for (std::size_t i = 0; i < n_phases_; ++i) {
      ps_[i] = ps_try_[i];
    }
  }
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
  if (problem_->get_n_phases() != 0) {
    throw_pretty(
        "Invalid operation: parameterized problems cannot be cast by "
        "SolverFDDP.");
  }
  auto sp = std::dynamic_pointer_cast<ShootingProblemTpl<Scalar>>(problem_);
  if (sp == nullptr) {
    throw_pretty(
        "Invalid operation: parameterized problems cannot be cast by "
        "SolverFDDP.");
  }
  ReturnType ret(
      std::static_pointer_cast<ProblemAbstractTpl<NewScalar>>(
          std::make_shared<ProblemType>(sp->template cast<NewScalar>())),
      dyn_solver_, term_solver_, astate_solver_);
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
  std::size_t max_nu = 0;
  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    max_nu = std::max(max_nu, models[t]->get_nu());
  }
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
  // Parameter data -- only allocated when the problem has phases
  n_phases_ = problem_->get_n_phases();
  if (n_phases_ > 0) {
    const std::vector<std::size_t>& phase_starts = problem_->get_phase_idxs();
    const std::vector<std::size_t>& phase_ends = problem_->get_phase_edxs();
    if (phase_starts.size() != n_phases_ || phase_ends.size() != n_phases_) {
      throw_pretty(
          "Invalid argument: phase boundary count does not match n_phases.");
    }
    std::size_t max_np = 0;
    for (std::size_t i = 0; i < n_phases_; ++i) {
      if (phase_starts[i] >= phase_ends[i] || phase_ends[i] > T ||
          (i > 0 && phase_starts[i] != phase_ends[i - 1])) {
        throw_pretty("Invalid argument: inconsistent phase boundaries.");
      }
      const std::size_t np = models[phase_starts[i]]->get_np();
      for (std::size_t t = phase_starts[i]; t < phase_ends[i]; ++t) {
        if (models[t]->get_np() != np) {
          throw_pretty(
              "Invalid argument: all models in a phase must have the same "
              "parameter dimension.");
        }
      }
      max_np = std::max(max_np, np);
    }
    if (phase_starts.front() != 0 || phase_ends.back() != T ||
        problem_->get_terminalModel()->get_np() !=
            models[phase_starts.back()]->get_np()) {
      throw_pretty(
          "Invalid argument: phases must cover the horizon and the terminal "
          "parameter dimension must match the final phase.");
    }
    // Per-node arrays (T+1 and T)
    Vp_.resize(T + 1);
    Vpp_.resize(T + 1);
    Vpx_.resize(T + 1);
    Vpx_f_.resize(T + 1);
    Qp_.resize(T);
    Qpp_.resize(T);
    Qpx_.resize(T);
    Qpu_.resize(T);
    P_.resize(T);
    P_dp_.resize(T);
    Lpp_dp_.resize(T + 1);
    Lpx_dp_.resize(T + 1);
    Lpu_dp_.resize(T);
    Qpc_.resize(T);
    Vpc_.resize(T + 1);
    for (std::size_t t = 0; t < T; ++t) {
      const std::size_t nu = models[t]->get_nu();
      const std::size_t np = models[t]->get_np();
      Vp_[t] = VectorXs::Zero(np);
      Vpp_[t] = MatrixXs::Zero(np, np);
      Vpx_[t] = MatrixXs::Zero(np, ndx);
      Vpx_f_[t] = VectorXs::Zero(np);
      Qp_[t] = VectorXs::Zero(np);
      Qpp_[t] = MatrixXs::Zero(np, np);
      Qpx_[t] = MatrixXs::Zero(np, ndx);
      Qpu_[t] = MatrixXs::Zero(np, nu);
      P_[t] = MatrixXs::Zero(nu, np);
      P_dp_[t] = VectorXs::Zero(nu);
      Lpp_dp_[t] = VectorXs::Zero(np);
      Lpx_dp_[t] = VectorXs::Zero(ndx);
      Lpu_dp_[t] = VectorXs::Zero(nu);
      Qpc_[t] = MatrixXs::Zero(np, nh_T);
      Vpc_[t] = MatrixXs::Zero(np, nh_T);
    }
    const std::size_t terminal_np = problem_->get_terminalModel()->get_np();
    Vp_.back() = VectorXs::Zero(terminal_np);
    Vpp_.back() = MatrixXs::Zero(terminal_np, terminal_np);
    Vpx_.back() = MatrixXs::Zero(terminal_np, ndx);
    Vpx_f_.back() = VectorXs::Zero(terminal_np);
    Lpp_dp_.back() = VectorXs::Zero(terminal_np);
    Lpx_dp_.back() = VectorXs::Zero(ndx);
    Vpc_.back() = MatrixXs::Zero(terminal_np, nh_T);
    // Per-phase arrays
    ps_.resize(n_phases_);
    ps_try_.resize(n_phases_);
    dps_.resize(n_phases_);
    kp_.resize(n_phases_);
    Kp_.resize(n_phases_);
    Vp_phase_.resize(n_phases_);
    Vpp_phase_.resize(n_phases_);
    Vpx_phase_.resize(n_phases_);
    Vpx_f_phase_.resize(n_phases_);
    Vpp_llt_.resize(n_phases_);
    Vpc_phase_.resize(n_phases_);
    dPc_.resize(n_phases_);
    Kpc_.resize(n_phases_);
    // LuNull/QrNull data
    Vpp_rank_.resize(n_phases_);
    YZp_.resize(n_phases_);
    Vpy_.resize(n_phases_);
    Vyy_.resize(n_phases_);
    Vy_.resize(n_phases_);
    Vxy_.resize(n_phases_);
    Vyy_llt_.resize(n_phases_);
    kp_y_.resize(n_phases_);
    Kp_y_.resize(n_phases_);
    Vpp_svd_.resize(n_phases_);
    FxTVxx_param_ = MatrixXs::Zero(ndx, ndx);
    FpTVxx_param_ = MatrixXs::Zero(max_np, ndx);
    FuTVxx_param_ = MatrixXs::Zero(max_nu, ndx);
    Vpp_sym_ = MatrixXs::Zero(max_np, max_np);
    Vp_next_ = VectorXs::Zero(max_np);
    Vpp_next_ = MatrixXs::Zero(max_np, max_np);
    Vpx_next_ = MatrixXs::Zero(max_np, ndx);
    Vpx_f_next_ = VectorXs::Zero(max_np);
    Vpc_next_ = MatrixXs::Zero(max_np, nh_T);
    qp_c_.resize(n_phases_);
    qp_x0_.resize(n_phases_);
    qp_u0_.resize(n_phases_);
#ifdef CROCODDYL_WITH_ODYN
    qp_models_.resize(n_phases_);
    qp_datas_.resize(n_phases_);
    qp_solvers_.resize(n_phases_);
    qp_params_.resize(n_phases_);
#endif
    const std::vector<std::shared_ptr<ParameterPhaseModel>>& params_models =
        problem_->get_paramsModel();
    for (std::size_t i = 0; i < n_phases_; ++i) {
      const std::size_t np = models[phase_starts[i]]->get_np();
      ps_[i] = VectorXs::Zero(np);
      ps_try_[i] = VectorXs::Zero(np);
      dps_[i] = VectorXs::Zero(np);
      kp_[i] = VectorXs::Zero(np);
      Kp_[i] = MatrixXs::Zero(np, ndx);
      Vp_phase_[i] = VectorXs::Zero(np);
      Vpp_phase_[i] = MatrixXs::Zero(np, np);
      Vpx_phase_[i] = MatrixXs::Zero(np, ndx);
      Vpx_f_phase_[i] = VectorXs::Zero(np);
      Vpp_llt_[i] = Eigen::LLT<MatrixXs>(np);
      Vpc_phase_[i] = MatrixXs::Zero(np, nh_T);
      dPc_[i] = MatrixXs::Zero(np, nh_T);
      Kpc_[i] = MatrixXs::Zero(np, nh_T);
      Vpp_rank_[i] = np;
      YZp_[i] = MatrixXs::Zero(np, np);
      Vpy_[i] = MatrixXs::Zero(np, np);
      Vyy_[i] = MatrixXs::Zero(np, np);
      Vy_[i] = VectorXs::Zero(np);
      Vxy_[i] = MatrixXs::Zero(ndx, np);
      Vyy_llt_[i] = Eigen::LLT<MatrixXs>(np);
      kp_y_[i] = VectorXs::Zero(np);
      Kp_y_[i] = MatrixXs::Zero(np, ndx);
      Vpp_svd_[i] = Eigen::JacobiSVD<MatrixXs>(
          np, np, Eigen::ComputeFullU | Eigen::ComputeFullV);
      qp_c_[i] = VectorXs::Zero(np);
      const std::shared_ptr<ConstraintModelManager>& constraints =
          params_models[i]->get_constraints();
      if (constraints != nullptr) {
        qp_x0_[i] = constraints->get_state()->zero();
        qp_u0_[i] = VectorXs::Zero(constraints->get_nu());
#ifdef CROCODDYL_WITH_ODYN
        const std::size_t nh = constraints->get_nh();
        const std::size_t ng = constraints->get_ng();
        if (np != 0 && (nh != 0 || ng != 0)) {
          qp_models_[i] =
              std::make_shared<odyn::DenseModelTpl<Scalar>>(np, nh, 2 * ng);
          qp_datas_[i] =
              std::make_shared<odyn::DenseDataTpl<Scalar>>(*qp_models_[i]);
          qp_solvers_[i] = std::make_shared<odyn::DenseQPTpl<Scalar>>();
          qp_params_[i] = std::make_shared<odyn::ParamsTpl<Scalar>>();
        }
#endif
      } else {
        qp_x0_[i] = VectorXs::Zero(0);
        qp_u0_[i] = VectorXs::Zero(0);
      }
    }
    // Arrival-state correction matrices
    Vxx0_ = MatrixXs::Zero(ndx, ndx);
    Vx0_ = VectorXs::Zero(ndx);
    Vxx0_llt_ = Eigen::LLT<MatrixXs>(ndx);
  }
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

template <typename Scalar>
bool SolverFDDPTpl<Scalar>::solve(const std::vector<VectorXs>& init_xs,
                                  const std::vector<VectorXs>& init_us,
                                  const std::vector<VectorXs>& init_ps,
                                  const std::size_t maxiter,
                                  const bool is_feasible,
                                  const Scalar reg_init) {
  // Initialise parameter vectors from init_ps (one per phase)
  if (!init_ps.empty()) {
    if (init_ps.size() != n_phases_) {
      throw_pretty(
          "Invalid argument: init_ps must contain one vector per phase.");
    }
    for (std::size_t i = 0; i < n_phases_; ++i) {
      if (init_ps[i].size() != ps_[i].size()) {
        throw_pretty("Invalid argument: init_ps[" << i
                                                  << "] has wrong dimension.");
      }
      ps_[i] = init_ps[i];
    }
  }
  for (std::size_t i = 0; i < n_phases_; ++i) {
    ps_try_[i] = ps_[i];
    problem_->update_p(ps_[i], i);
  }
  // Delegate to the base solver loop after initializing the parameters.
  return SolverAbstract::solve(init_xs, init_us, DefaultVector<Scalar>::value,
                               maxiter, is_feasible, reg_init);
}

// Parametrized backward pass

template <typename Scalar>
void SolverFDDPTpl<Scalar>::parametrizedBackwardPass() {
  const std::shared_ptr<ActionDataAbstract>& d_T = problem_->get_terminalData();
  const std::size_t terminal_np = d_T->Lp.size();
  Vp_.back() = d_T->Lp;
  Vpx_.back() = d_T->Lpx;
  Vpp_.back() = d_T->Lpp;
  if (preg_ != Scalar(0.)) {
    for (std::size_t i = 0; i < terminal_np; ++i) {
      Vpp_.back()(i, i) += preg_;
    }
  }
  Vpx_f_.back().noalias() = Vpx_.back() * fs_.back();

  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract>>& datas =
      problem_->get_runningDatas();
  const std::vector<std::size_t>& phase_starts = problem_->get_phase_idxs();
  const std::vector<std::size_t>& phase_ends = problem_->get_phase_edxs();

  for (std::size_t phase = n_phases_; phase-- > 0;) {
    const std::size_t np = models[phase_starts[phase]]->get_np();
    Vp_next_.setZero();
    Vpp_next_.setZero();
    Vpx_next_.setZero();
    Vpx_f_next_.setZero();
    if (phase + 1 == n_phases_) {
      Vp_next_.head(np) = Vp_.back();
      Vpp_next_.topLeftCorner(np, np) = Vpp_.back();
      Vpx_next_.topRows(np) = Vpx_.back();
      Vpx_f_next_.head(np) = Vpx_f_.back();
    }
    for (std::size_t t = phase_ends[phase]; t-- > phase_starts[phase];) {
      const std::shared_ptr<ActionModelAbstract>& model = models[t];
      parametrizedActionValueFunction(t, model, datas[t]);
      parametrizedPolicy(t);
      parametrizedValueFunction(t, model);
      Vp_next_.head(np) = Vp_[t];
      Vpp_next_.topLeftCorner(np, np) = Vpp_[t];
      Vpx_next_.topRows(np) = Vpx_[t];
      Vpx_f_next_.head(np) = Vpx_f_[t];
    }
    const std::size_t start = phase_starts[phase];
    Vp_phase_[phase] = Vp_[start];
    Vpp_phase_[phase] = Vpp_[start];
    Vpx_phase_[phase] = Vpx_[start];
    Vpx_f_phase_[phase] = Vpx_f_[start];
  }
}

// Parametrized action-value function

template <typename Scalar>
void SolverFDDPTpl<Scalar>::parametrizedActionValueFunction(
    const std::size_t t, const std::shared_ptr<ActionModelAbstract>& model,
    const std::shared_ptr<ActionDataAbstract>& data) {
  const std::size_t nu = model->get_nu();
  const std::size_t np = model->get_np();

  // Vx_{t+1} + Vxx_{t+1} * f_{t+1}
  VectorXs& Vx_p = Vx_[t + 1];
  MatrixXs& Vxx_p = Vxx_[t + 1];

  Vx_p += Vxx_f_[t + 1];  // temporarily add gap term

  FxTVxx_param_.noalias() = data->Fx.transpose() * Vxx_p;
  FpTVxx_param_.topRows(np).noalias() = data->Fp.transpose() * Vxx_p;

  Qp_[t].noalias() = data->Lp + Vp_next_.head(np) + Vpx_f_next_.head(np) +
                     data->Fp.transpose() * Vx_p;

  Qpx_[t].noalias() = data->Lpx + Vpx_next_.topRows(np) * data->Fx;
  Qpx_[t].noalias() += data->Fp.transpose() * FxTVxx_param_.transpose();

  Qpp_[t].noalias() = data->Lpp + Vpp_next_.topLeftCorner(np, np);
  Qpp_[t].noalias() +=
      Scalar(2.) * data->Fp.transpose() * Vpx_next_.topRows(np).transpose();
  Qpp_[t].noalias() += FpTVxx_param_.topRows(np) * data->Fp;

  if (nu != 0) {
    FuTVxx_param_.topRows(static_cast<Eigen::Index>(nu)).noalias() =
        data->Fu.transpose() * Vxx_p;
    Qpu_[t].noalias() = data->Lpu + Vpx_next_.topRows(np) * data->Fu;
    Qpu_[t].noalias() +=
        data->Fp.transpose() *
        FuTVxx_param_.topRows(static_cast<Eigen::Index>(nu)).transpose();
  }

  // Restore
  Vx_p -= Vxx_f_[t + 1];
}

// Parametrized policy

template <typename Scalar>
void SolverFDDPTpl<Scalar>::parametrizedPolicy(const std::size_t t) {
  const std::size_t nu = problem_->get_runningModels()[t]->get_nu();
  if (nu != 0) {
    // P = Quu^{-1} Qpu^T   (already have Quu_llt_ from backwardPass)
    P_[t] = Qpu_[t].transpose();
    Quu_llt_[t].solveInPlace(P_[t]);
  }
}

// Parametrized value function

template <typename Scalar>
void SolverFDDPTpl<Scalar>::parametrizedValueFunction(
    const std::size_t t, const std::shared_ptr<ActionModelAbstract>& model) {
  const std::size_t nu = model->get_nu();
  Vp_[t] = Qp_[t];
  Vpx_[t] = Qpx_[t];
  Vpp_[t] = Qpp_[t];
  if (nu != 0) {
    Vp_[t].noalias() -= P_[t].transpose() * Qu_[t];
    Vpp_[t].noalias() -= Qpu_[t] * P_[t];
    Vpx_[t].noalias() -= Qpu_[t] * K_[t];
  }
  const std::size_t np = model->get_np();
  Vpp_sym_.topLeftCorner(np, np).noalias() =
      Scalar(0.5) * (Vpp_[t] + Vpp_[t].transpose());
  Vpp_[t] = Vpp_sym_.topLeftCorner(np, np);
  Vpx_f_[t].noalias() = Vpx_[t] * fs_[t];
}

// Parameter solve pass

template <typename Scalar>
void SolverFDDPTpl<Scalar>::paramsPass() {
  if (n_phases_ == 0) return;

  const std::vector<std::size_t>& phase_starts = problem_->get_phase_idxs();

  switch (astate_solver_) {
    case AStateSchur:
    case AStateNone: {
      for (std::size_t i = 0; i < n_phases_; ++i) {
        if (ps_[i].size() == 0) {
          continue;
        }
        Vpp_llt_[i].compute(Vpp_phase_[i]);
        if (Vpp_llt_[i].info() != Eigen::Success) {
          throw_pretty("parameter backward error: Vpp_phase is not PD");
        }
        kp_[i] = Vp_phase_[i] + Vpx_f_phase_[i];
        Vpp_llt_[i].solveInPlace(kp_[i]);
        Kp_[i] = Vpx_phase_[i];
        Vpp_llt_[i].solveInPlace(Kp_[i]);
      }
      break;
    }
    case AStateLuNull:
    case AStateQrNull: {
      for (std::size_t i = 0; i < n_phases_; ++i) {
        const std::size_t np = static_cast<std::size_t>(ps_[i].size());
        if (np == 0) {
          continue;
        }
        Eigen::JacobiSVD<MatrixXs>& svd = Vpp_svd_[i];
        svd.compute(Vpp_phase_[i], Eigen::ComputeFullU | Eigen::ComputeFullV);
        const Eigen::Index sv_count = svd.singularValues().size();
        const typename Eigen::NumTraits<Scalar>::Real tol =
            static_cast<typename Eigen::NumTraits<Scalar>::Real>(
                std::max(Vpp_phase_[i].rows(), Vpp_phase_[i].cols())) *
            Eigen::NumTraits<Scalar>::epsilon() *
            (sv_count > 0 ? svd.singularValues()(0)
                          : typename Eigen::NumTraits<Scalar>::Real(0.));
        Eigen::Index rank_idx = 0;
        while (rank_idx < sv_count && svd.singularValues()(rank_idx) > tol) {
          ++rank_idx;
        }
        const Eigen::Index np_i = static_cast<Eigen::Index>(np);
        const Eigen::Index nullity = np_i - rank_idx;
        YZp_[i].setZero();
        if (rank_idx > 0) {
          YZp_[i].leftCols(rank_idx) = svd.matrixU().leftCols(rank_idx);
        }
        if (nullity > 0) {
          YZp_[i].middleCols(rank_idx, nullity) =
              svd.matrixV().rightCols(nullity);
        }
        Vpp_rank_[i] = static_cast<std::size_t>(rank_idx);
        const std::size_t rank = Vpp_rank_[i];
        const Eigen::Index rank_eig = static_cast<Eigen::Index>(rank);
        // Build reduced system
        Vpy_[i].setZero();
        Vyy_[i].setZero();
        if (rank_eig > 0) {
          Vpy_[i].leftCols(rank_eig).noalias() =
              Vpp_phase_[i] * YZp_[i].leftCols(rank_eig);
          Vyy_[i].topLeftCorner(rank_eig, rank_eig).noalias() =
              YZp_[i].leftCols(rank_eig).transpose() *
              Vpy_[i].leftCols(rank_eig);
        }
        if (rank_eig < np_i) {
          Vyy_[i]
              .bottomRightCorner(np_i - rank_eig, np_i - rank_eig)
              .setIdentity();
        }
        Vyy_llt_[i].compute(Vyy_[i]);
        if (Vyy_llt_[i].info() != Eigen::Success) {
          throw_pretty("parameter backward error: Vyy is not PD");
        }
        Vy_[i].setZero();
        Vxy_[i].setZero();
        if (rank_eig > 0) {
          Vy_[i].head(rank_eig).noalias() =
              YZp_[i].leftCols(rank_eig).transpose() * Vp_phase_[i];
          Vxy_[i].leftCols(rank_eig).noalias() =
              Vpx_phase_[i].transpose() * YZp_[i].leftCols(rank_eig);
        }
        kp_y_[i] = Vy_[i];
        Vyy_llt_[i].solveInPlace(kp_y_[i]);
        kp_[i].noalias() = YZp_[i] * kp_y_[i];
        Kp_y_[i].setZero();
        if (rank_eig > 0) {
          Kp_y_[i].topRows(rank_eig) = Vxy_[i].leftCols(rank_eig).transpose();
        }
        Vyy_llt_[i].solveInPlace(Kp_y_[i]);
        Kp_[i].noalias() = YZp_[i] * Kp_y_[i];
      }
      break;
    }
    case AStateQP: {
#ifdef CROCODDYL_WITH_ODYN
      START_PROFILER("SolverFDDP::paramsPassQP");
      const std::vector<std::shared_ptr<ParameterPhaseModel>>& params_models =
          problem_->get_paramsModel();
      const std::vector<std::shared_ptr<ParameterPhaseData>>& params_datas =
          problem_->get_paramsData();
      for (std::size_t i = 0; i < n_phases_; ++i) {
        const std::size_t np = static_cast<std::size_t>(ps_[i].size());
        if (np == 0) {
          continue;
        }
        Vpp_llt_[i].compute(Vpp_phase_[i]);
        if (Vpp_llt_[i].info() != Eigen::Success) {
          STOP_PROFILER("SolverFDDP::paramsPassQP");
          throw_pretty("parameter backward error: Vpp_phase is not PD");
        }
        Kp_[i] = Vpx_phase_[i];
        Vpp_llt_[i].solveInPlace(Kp_[i]);

        qp_c_[i].noalias() = Vp_phase_[i] + Vpx_f_phase_[i];
        const std::shared_ptr<ConstraintModelManager>& constraints_model =
            params_models[i]->get_constraints();
        const std::shared_ptr<ConstraintDataManager>& constraints_data =
            params_datas[i]->constraints;
        const bool has_constraints = constraints_model != nullptr &&
                                     constraints_data != nullptr &&
                                     (constraints_model->get_nh() != 0 ||
                                      constraints_model->get_ng() != 0);

        if (!has_constraints) {
          kp_[i] = qp_c_[i];
          Vpp_llt_[i].solveInPlace(kp_[i]);
          continue;
        }

        const std::size_t nh = constraints_model->get_nh();
        const std::size_t ng = constraints_model->get_ng();
        if (qp_x0_[i].size() != static_cast<Eigen::Index>(
                                    constraints_model->get_state()->get_nx()) ||
            qp_u0_[i].size() !=
                static_cast<Eigen::Index>(constraints_model->get_nu())) {
          STOP_PROFILER("SolverFDDP::paramsPassQP");
          throw_pretty(
              "parameter backward error: parameter constraint dimensions "
              "changed; recreate the solver");
        }
        params_models[i]->calc(params_datas[i], qp_x0_[i], qp_u0_[i]);
        params_models[i]->calcDiff(params_datas[i], qp_x0_[i], qp_u0_[i]);
        if (!qp_models_[i] || !qp_datas_[i] || !qp_solvers_[i] ||
            !qp_params_[i]) {
          STOP_PROFILER("SolverFDDP::paramsPassQP");
          throw_pretty(
              "parameter backward error: missing preallocated QP data");
        }
        odyn::DenseModelTpl<Scalar>& qp_model = *qp_models_[i];
        odyn::DenseDataTpl<Scalar>& qp_data = *qp_datas_[i];
        odyn::DenseQPTpl<Scalar>& qp_solver = *qp_solvers_[i];
        odyn::ParamsTpl<Scalar>& qp_params = *qp_params_[i];
        if (qp_model.n != np || qp_model.m != nh || qp_model.p != 2 * ng) {
          STOP_PROFILER("SolverFDDP::paramsPassQP");
          throw_pretty(
              "parameter backward error: parameter constraint dimensions "
              "changed; recreate the solver");
        }

        qp_model.Q = Vpp_phase_[i];
        qp_model.c = qp_c_[i];
        qp_model.l.setConstant(-std::numeric_limits<Scalar>::infinity());
        qp_model.u.setConstant(std::numeric_limits<Scalar>::infinity());
        if (nh != 0) {
          qp_model.A = constraints_data->Hp;
          qp_model.b = -constraints_data->h;
        }
        if (ng != 0) {
          qp_model.G.topRows(ng) = constraints_data->Gp;
          qp_model.G.bottomRows(ng) = -constraints_data->Gp;
          qp_model.h.head(ng) =
              constraints_model->get_ub().head(ng) - constraints_data->g;
          qp_model.h.tail(ng) =
              constraints_data->g - constraints_model->get_lb().head(ng);
        }

        const odyn::Status status = qp_solver.solve(
            qp_model, qp_data, qp_params, odyn::VerboseLevel::Silent);
        if (status == odyn::Status::PrimalInfeasible ||
            status == odyn::Status::DualInfeasible ||
            status == odyn::Status::Unsolved) {
          STOP_PROFILER("SolverFDDP::paramsPassQP");
          throw_pretty("parameter backward error: arrival-node QP failed");
        }
        kp_[i].noalias() = -qp_data.x;
      }
      STOP_PROFILER("SolverFDDP::paramsPassQP");
      break;
#else
      throw_pretty(
          "parameter backward error: AStateQP requires CROCODDYL_WITH_ODYN");
#endif
    }
  }

  // Compute dps = -kp (feedforward) - Kp * dx_phase_start (feedback)
  for (std::size_t i = 0; i < n_phases_; ++i) {
    dps_[i] = -kp_[i];
  }
  // For phases > 0, couple the parameter direction to the state direction
  // at the phase start (dxs_[phase_starts[nph]])
  for (std::size_t i = 1; i < n_phases_; ++i) {
    const std::size_t t = phase_starts[i];
    dps_[i].noalias() -= Kp_[i] * dxs_[t];
  }

  // Arrival-state solver: marginalise the initial state
  if (astate_solver_ != AStateNone && astate_solver_ != AStateQP) {
    Vxx0_ = Vxx_[0];
    Vxx0_.noalias() -= Vpx_phase_[0].transpose() * Kp_[0];
    Vx0_ = Vx_[0];
    Vx0_.noalias() -= Vpx_phase_[0].transpose() * kp_[0];
    Vxx0_llt_.compute(Vxx0_);
    if (Vxx0_llt_.info() != Eigen::Success) {
      throw_pretty("parameter backward error: Vxx0 is not PD");
    }
    dxs_[0] = -Vx0_;
    Vxx0_llt_.solveInPlace(dxs_[0]);
    dps_[0].noalias() -= Kp_[0] * dxs_[0];
  }

  // Update k_ with P * dps contribution so control policy uses params
  const std::vector<std::size_t>& phase_ends = problem_->get_phase_edxs();
  std::size_t t0 = 0;
  for (std::size_t i = 0; i < n_phases_; ++i) {
    for (std::size_t t = t0; t < phase_ends[i]; ++t) {
      P_dp_[t].noalias() = P_[t] * dps_[i];
      k_[t] += P_dp_[t];
    }
    t0 = phase_ends[i];
  }
}

// paramsBatchPass

template <typename Scalar>
void SolverFDDPTpl<Scalar>::paramsBatchPass() {
  if (n_phases_ == 0) return;
  for (std::size_t i = 0; i < n_phases_; ++i) {
    if (ps_[i].size() == 0) {
      continue;
    }
    Kpc_[i] = Vpc_phase_[i];
    Vpp_llt_[i].solveInPlace(Kpc_[i]);
    dPc_[i] = -Kpc_[i];
  }
}

// calcDir override (restore p on non-acceptance)

template <typename Scalar>
void SolverFDDPTpl<Scalar>::calcDir() {
  START_PROFILER("SolverFDDP::calcDir");
  if (!acceptstep_ && n_phases_ > 0) {
    // Restore the last accepted parameters so calcDiff sees correct values
    for (std::size_t i = 0; i < n_phases_; ++i) {
      problem_->update_p(ps_[i], i);
    }
    if (problem_->has_parameter_constraints()) {
      const std::vector<std::shared_ptr<ParameterPhaseModel>>& params_models =
          problem_->get_paramsModel();
      const std::vector<std::shared_ptr<ParameterPhaseData>>& params_datas =
          problem_->get_paramsData();
      for (std::size_t i = 0; i < params_models.size(); ++i) {
        const std::shared_ptr<ConstraintModelManager>& constraints =
            params_models[i]->get_constraints();
        if (constraints == nullptr) {
          continue;
        }
        if (params_datas[i]->constraints == nullptr) {
          throw_pretty("Invalid data: parameter-constraint data is missing");
        }
        if (i >= qp_x0_.size() || i >= qp_u0_.size() ||
            constraints->get_state() == nullptr ||
            qp_x0_[i].size() !=
                static_cast<Eigen::Index>(constraints->get_state()->get_nx()) ||
            qp_u0_[i].size() !=
                static_cast<Eigen::Index>(constraints->get_nu())) {
          throw_pretty(
              "Invalid data: parameter-constraint workspace is invalid");
        }
        params_models[i]->calc(params_datas[i], qp_x0_[i], qp_u0_[i]);
      }
    }
  }
  SolverAbstract::calcDir();
  if (n_phases_ > 0 && astate_solver_ != AStateNone) {
    // PDDP treats the arrival state as a decision variable, not as an initial
    // dynamics defect.
    fs_[0].setZero();
    ffeas_ = computeFeasibility(fs_);
    feas_ = ffeas_ + gfeas_ + hfeas_;
  }
  STOP_PROFILER("SolverFDDP::calcDir");
}

// Getters

template <typename Scalar>
ArrivalStateSolverType SolverFDDPTpl<Scalar>::get_astate_solver() const {
  return astate_solver_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::VectorXs>&
SolverFDDPTpl<Scalar>::get_ps() const {
  return ps_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::VectorXs>&
SolverFDDPTpl<Scalar>::get_ps_try() const {
  return ps_try_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::VectorXs>&
SolverFDDPTpl<Scalar>::get_dps() const {
  return dps_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::VectorXs>&
SolverFDDPTpl<Scalar>::get_kp() const {
  return kp_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::MatrixXs>&
SolverFDDPTpl<Scalar>::get_Kp() const {
  return Kp_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::VectorXs>&
SolverFDDPTpl<Scalar>::get_Vp() const {
  return Vp_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::MatrixXs>&
SolverFDDPTpl<Scalar>::get_Vpp() const {
  return Vpp_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::MatrixXs>&
SolverFDDPTpl<Scalar>::get_Vpx() const {
  return Vpx_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::VectorXs>&
SolverFDDPTpl<Scalar>::get_Vp_phase() const {
  return Vp_phase_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::MatrixXs>&
SolverFDDPTpl<Scalar>::get_Vpp_phase() const {
  return Vpp_phase_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::MatrixXs>&
SolverFDDPTpl<Scalar>::get_Vpx_phase() const {
  return Vpx_phase_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::VectorXs>&
SolverFDDPTpl<Scalar>::get_Qp() const {
  return Qp_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::MatrixXs>&
SolverFDDPTpl<Scalar>::get_Qpp() const {
  return Qpp_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::MatrixXs>&
SolverFDDPTpl<Scalar>::get_Qpx() const {
  return Qpx_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::MatrixXs>&
SolverFDDPTpl<Scalar>::get_Qpu() const {
  return Qpu_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::MatrixXs>&
SolverFDDPTpl<Scalar>::get_P() const {
  return P_;
}

template <typename Scalar>
void SolverFDDPTpl<Scalar>::set_astate_solver(
    const ArrivalStateSolverType type) {
  astate_solver_ = type;
}

}  // namespace crocoddyl
