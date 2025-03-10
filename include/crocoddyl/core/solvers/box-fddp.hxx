///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
SolverBoxFDDPTpl<Scalar>::SolverBoxFDDPTpl(
    std::shared_ptr<ShootingProblem> problem,
    const DynamicsSolverType dyn_solver, const EqualitySolverType term_solver)
    : SolverFDDP(problem, dyn_solver, term_solver),
      qp_(problem->get_runningModels()[0]->get_nu(), 100, Scalar(0.1),
          Scalar(1e-5), Scalar(0.)) {
  allocateData();
  const std::size_t n_alphas = 10;
  alphas_.resize(n_alphas);
  for (std::size_t n = 0; n < n_alphas; ++n) {
    alphas_[n] = Scalar(1.) / pow(Scalar(2.), static_cast<Scalar>(n));
  }
  // Change the default convergence tolerance since the gradient of the
  // Lagrangian is smaller than an unconstrained OC problem (i.e. gradient = Qu
  // - mu^T * C where mu > 0 and C defines the inequality matrix that bounds the
  // control); and we don't have access to mu from the box QP.
  th_stop_ = Scalar(5e-5);
}

template <typename Scalar>
void SolverBoxFDDPTpl<Scalar>::resizeRunningData() {
  START_PROFILER("SolverBoxFDDP::resizeRunningData");
  SolverFDDP::resizeRunningData();
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    Quu_inv_[t].conservativeResize(nu, nu);
    du_lb_[t].conservativeResize(nu);
    du_ub_[t].conservativeResize(nu);
  }
  STOP_PROFILER("SolverBoxFDDP::resizeRunningData");
}

template <typename Scalar>
void SolverBoxFDDPTpl<Scalar>::allocateData() {
  const std::size_t T = problem_->get_T();
  Quu_inv_.resize(T);
  du_lb_.resize(T);
  du_ub_.resize(T);
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    Quu_inv_[t] = MatrixXs::Zero(nu, nu);
    du_lb_[t] = VectorXs::Zero(nu);
    du_ub_[t] = VectorXs::Zero(nu);
  }
  const std::size_t ndx = problem_->get_ndx();
  xnext_ = VectorXs::Zero(ndx);
}

template <typename Scalar>
void SolverBoxFDDPTpl<Scalar>::computePolicy(const std::size_t t) {
  const std::size_t nu = problem_->get_runningModels()[t]->get_nu();
  if (nu > 0) {
    if (!problem_->get_runningModels()[t]->get_has_control_limits() ||
        !is_feasible_) {
      // No control limits on this model: Use vanilla DDP
      SolverFDDP::computePolicy(t);
      return;
    }
    du_lb_[t] = problem_->get_runningModels()[t]->get_u_lb() - us_[t];
    du_ub_[t] = problem_->get_runningModels()[t]->get_u_ub() - us_[t];
    const BoxQPSolution& boxqp_sol =
        qp_.solve(Quu_[t], Qu_[t], du_lb_[t], du_ub_[t], k_[t]);
    // Compute controls
    Quu_inv_[t].setZero();
    for (std::size_t i = 0; i < boxqp_sol.free_idx.size(); ++i) {
      for (std::size_t j = 0; j < boxqp_sol.free_idx.size(); ++j) {
        Quu_inv_[t](boxqp_sol.free_idx[i], boxqp_sol.free_idx[j]) =
            boxqp_sol.Hff_inv(i, j);
      }
    }
    K_[t].noalias() = Quu_inv_[t] * Qxu_[t].transpose();
    k_[t] = -boxqp_sol.x;
    // The box-QP clamped the gradient direction; this is important for
    // accounting the algorithm advancement (i.e. stopping criteria)
    for (std::size_t i = 0; i < boxqp_sol.clamped_idx.size(); ++i) {
      Qu_[t](boxqp_sol.clamped_idx[i]) = Scalar(0.);
    }
  }
}

template <typename Scalar>
void SolverBoxFDDPTpl<Scalar>::forwardPass(const Scalar steplength) {
  if (steplength > Scalar(1.) || steplength < Scalar(0.)) {
    throw_pretty("Invalid argument: "
                 << "invalid step length, value is between 0. to 1.");
  }
  cost_try_ = Scalar(0.);
  xnext_ = problem_->get_x0();
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
      problem_->get_runningDatas();
  if ((is_feasible_) || (steplength == 1)) {
    for (std::size_t t = 0; t < T; ++t) {
      const std::shared_ptr<ActionModelAbstract>& m = models[t];
      const std::shared_ptr<ActionDataAbstract>& d = datas[t];
      const std::size_t nu = m->get_nu();
      xs_try_[t] = xnext_;
      m->get_state()->diff(xs_[t], xs_try_[t], dx_[t]);
      if (nu != 0) {
        us_try_[t].noalias() = us_[t] - k_[t] * steplength - K_[t] * dx_[t];
        if (m->get_has_control_limits()) {  // clamp control
          us_try_[t] =
              us_try_[t].cwiseMax(m->get_u_lb()).cwiseMin(m->get_u_ub());
        }
        m->calc(d, xs_try_[t], us_try_[t]);
      } else {
        m->calc(d, xs_try_[t]);
      }
      xnext_ = d->xnext;
      cost_try_ += d->cost;
      if (raiseIfNaN(cost_try_)) {
        throw_pretty("forward_error");
      }
      if (raiseIfNaN(xnext_.template lpNorm<Eigen::Infinity>())) {
        throw_pretty("forward_error");
      }
    }
    const std::shared_ptr<ActionModelAbstract>& m =
        problem_->get_terminalModel();
    const std::shared_ptr<ActionDataAbstract>& d = problem_->get_terminalData();
    xs_try_.back() = xnext_;
    m->calc(d, xs_try_.back());
    cost_try_ += d->cost;
    if (raiseIfNaN(cost_try_)) {
      throw_pretty("forward_error");
    }
  } else {
    for (std::size_t t = 0; t < T; ++t) {
      const std::shared_ptr<ActionModelAbstract>& m = models[t];
      const std::shared_ptr<ActionDataAbstract>& d = datas[t];
      const std::size_t nu = m->get_nu();
      m->get_state()->integrate(xnext_, fs_[t] * (steplength - 1), xs_try_[t]);
      m->get_state()->diff(xs_[t], xs_try_[t], dx_[t]);
      if (nu != 0) {
        us_try_[t].noalias() = us_[t] - k_[t] * steplength - K_[t] * dx_[t];
        if (m->get_has_control_limits()) {  // clamp control
          us_try_[t] =
              us_try_[t].cwiseMax(m->get_u_lb()).cwiseMin(m->get_u_ub());
        }
        m->calc(d, xs_try_[t], us_try_[t]);
      } else {
        m->calc(d, xs_try_[t]);
      }
      xnext_ = d->xnext;
      cost_try_ += d->cost;
      if (raiseIfNaN(cost_try_)) {
        throw_pretty("forward_error");
      }
      if (raiseIfNaN(xnext_.template lpNorm<Eigen::Infinity>())) {
        throw_pretty("forward_error");
      }
    }
    const std::shared_ptr<ActionModelAbstract>& m =
        problem_->get_terminalModel();
    const std::shared_ptr<ActionDataAbstract>& d = problem_->get_terminalData();
    m->get_state()->integrate(xnext_, fs_.back() * (steplength - Scalar(1.)),
                              xs_try_.back());
    m->calc(d, xs_try_.back());
    cost_try_ += d->cost;
    if (raiseIfNaN(cost_try_)) {
      throw_pretty("forward_error");
    }
  }
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::MatrixXs>&
SolverBoxFDDPTpl<Scalar>::get_Quu_inv() const {
  return Quu_inv_;
}

}  // namespace crocoddyl
