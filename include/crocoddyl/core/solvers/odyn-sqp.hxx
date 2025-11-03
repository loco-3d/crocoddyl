///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2025-2025, Heriot-Watt University
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
      zero_upsilon_(false) {
  // Allocating the solver's data
  allocateData();
  // Create the Odyn solver and parameters
  // params.max_iter = 200;
  // params.stop_dinf = 1e-7;
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::computeDirection(const bool recalc) {
  START_PROFILER("SolverOdynSQP::computeDirection");
  // Update the batch's derivatives
  if (recalc) {
    SolverAbstract::calcDir();
  }
  computeQuadraticModel();
  // // Update the QP model and resize its data
  // model_.update(Q_, c_, A_, b_, G_, h_);
  // data_.conservativeResize(model_);
  // // Solve the QP problem using Odyn
  // // self.params.stop_abs = 1e-4
  // solver.solve(model_, data_, params_, odyn::VerboseLevel::Silent);
  // // Unpack primal into dx/du
  // x_ = model_.get_x(data_);
  // extractQpDirection(x_);
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
void SolverOdynSQPTpl<Scalar>::computeQuadraticModel() {}

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
template <typename NewScalar>
SolverOdynSQPTpl<NewScalar> SolverOdynSQPTpl<Scalar>::cast() const {
  typedef SolverOdynSQPTpl<NewScalar> ReturnType;
  typedef ShootingProblemTpl<NewScalar> ProblemType;
  ReturnType ret(
      std::make_shared<ProblemType>(problem_->template cast<NewScalar>()));
  // // Setting the abstract parameters
  // ret.setCallbacks(vector_cast<NewScalar>(callbacks_));
  // ret.set_th_acceptstep(scalar_cast<NewScalar>(th_acceptstep_));
  // ret.set_th_stop(
  //     std::sqrt(std::numeric_limits<NewScalar>::epsilon()) <
  //     NewScalar(th_stop_)
  //         ? scalar_cast<NewScalar>(th_stop_)
  //         : std::sqrt(
  //               std::numeric_limits<NewScalar>::
  //                   epsilon()));  // Stopping threshold shouldn't be lower
  //                   than
  //                                 // square root of the machine precision
  // // Setting the FDDP parameters
  // ret.set_alphas(vector_cast<NewScalar>(alphas_));
  // ret.set_reg_incfactor(scalar_cast<NewScalar>(reg_incfactor_));
  // ret.set_reg_decfactor(scalar_cast<NewScalar>(reg_decfactor_));
  // ret.set_reg_min(
  //     ScaleNumerics<Scalar>(1e-9) < NewScalar(reg_min_)
  //         ? scalar_cast<NewScalar>(reg_min_)
  //         : ScaleNumerics<NewScalar>(
  //               1e-9));  // Minimum regularization value shouldn't be lower
  //               than
  //                        // 1e-9 or 1e-5 for doubles or floats
  // ret.set_reg_max(
  //     ScaleNumerics<Scalar>(1e9, 1e-4) > NewScalar(reg_max_)
  //         ? scalar_cast<NewScalar>(reg_max_)
  //         : ScaleNumerics<NewScalar>(
  //               1e9, 1e-4));  // Maximum regularization value shouldn't be
  //                             // higher than 1e9 or 1e5 for doubles or floats
  // ret.set_th_grad(scalar_cast<NewScalar>(ScaleNumerics<NewScalar>(th_grad_)));
  // ret.set_th_noimprovement(scalar_cast<NewScalar>(th_noimprovement_));
  // ret.set_th_stepdec(scalar_cast<NewScalar>(th_stepdec_));
  // ret.set_th_stepinc(scalar_cast<NewScalar>(th_stepinc_));
  // ret.set_th_minimprove(scalar_cast<NewScalar>(th_minimprove_));
  // ret.set_th_acceptnegstep(scalar_cast<NewScalar>(th_acceptnegstep_));
  // ret.set_th_acceptminstep(scalar_cast<NewScalar>(th_acceptminstep_));
  // ret.set_rho(scalar_cast<NewScalar>(rho_));
  // ret.set_th_minfeas(scalar_cast<NewScalar>(th_minfeas_));
  // ret.set_upsilon_decfactor(scalar_cast<NewScalar>(upsilon_decfactor_));
  // ret.set_zero_upsilon(zero_upsilon_);
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
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
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
  const std::size_t nh_T = model_T->get_nh();
  const std::size_t ng_T = model_T->get_ng();
  n_ += ndx;
  m_ += nh_T;
  p_ += ng_T + ndx;
  p_ *= 2;
  // Store xs and us indeces for decision variables
  updateStateAndControlIndex();
  // Store the QP sparse matrices and vectors

  // Q_ = sp.lil_matrix((self.n, self.n));
  // c_ = np.zeros(self.n);
  // A_ = sp.lil_matrix((self.m, self.n));
  // b_ = np.zeros(self.m);
  // G_ = sp.lil_matrix((self.p, self.n));
  // h_ = np.zeros(self.p);
  // // Create the Odyn's sparse QP model
  // model_ = odyn.SparseModel(self.Q.tocsc(), self.c, self.A.tocsc(), self.b,
  // self.G.tocsc(), self.h) data_ = self.model.createData()
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::resizeRunningData() {
  START_PROFILER("SolverOdynSQP::resizeRunningData");
  SolverAbstract::resizeRunningData();
  const std::size_t T = problem_->get_T();
  const std::size_t ndx = problem_->get_ndx();
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    Luu_du_[t].conservativeResize(nu);
    Lxu_du_[t].conservativeResize(nu);
  }
  STOP_PROFILER("SolverOdynSQP::resizeRunningData");
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::resizeTerminalData() {
  START_PROFILER("SolverOdynSQP::resizeTerminalData");
  // const std::size_t T = problem_->get_T();
  // const std::size_t ndx = problem_->get_ndx();
  // const std::size_t nh_T = problem_->get_terminalModel()->get_nh_T();
  // const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
  //     problem_->get_runningModels();
  // for (std::size_t t = 0; t < T; ++t) {
  //   const std::shared_ptr<ActionModelAbstract>& model = models[t];
  //   const std::size_t nu = model->get_nu();
  // }
  STOP_PROFILER("SolverOdynSQP::resizeTerminalData");
}

template <typename Scalar>
void SolverOdynSQPTpl<Scalar>::updateStateAndControlIndex() {
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
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
