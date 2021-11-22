///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, CNRS-LAAS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "crocoddyl/core/solvers/box-ddp.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

SolverBoxDDP::SolverBoxDDP(boost::shared_ptr<ShootingProblem> problem)
    : SolverDDP(problem), qp_(problem->get_runningModels()[0]->get_nu(), 100, 0.1, 1e-5, 0.) {
  allocateData();

  const std::size_t n_alphas = 10;
  alphas_.resize(n_alphas);
  for (std::size_t n = 0; n < n_alphas; ++n) {
    alphas_[n] = 1. / pow(2., static_cast<double>(n));
  }
  // Change the default convergence tolerance since the gradient of the Lagrangian is smaller
  // than an unconstrained OC problem (i.e. gradient = Qu - mu^T * C where mu > 0 and C defines
  // the inequality matrix that bounds the control); and we don't have access to mu from the
  // box QP.
  th_stop_ = 5e-5;
}

SolverBoxDDP::~SolverBoxDDP() {}

void SolverBoxDDP::resizeData() {
  START_PROFILER("SolverBoxDDP::resizeData");
  SolverDDP::resizeData();

  const std::size_t T = problem_->get_T();
  const std::vector<boost::shared_ptr<ActionModelAbstract> >& models = problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const boost::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    Quu_inv_[t].conservativeResize(nu, nu);
    du_lb_[t].conservativeResize(nu);
    du_ub_[t].conservativeResize(nu);
  }
  STOP_PROFILER("SolverBoxDDP::resizeData");
}

void SolverBoxDDP::allocateData() {
  SolverDDP::allocateData();

  const std::size_t T = problem_->get_T();
  Quu_inv_.resize(T);
  du_lb_.resize(T);
  du_ub_.resize(T);
  const std::vector<boost::shared_ptr<ActionModelAbstract> >& models = problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const boost::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    Quu_inv_[t] = Eigen::MatrixXd::Zero(nu, nu);
    du_lb_[t] = Eigen::VectorXd::Zero(nu);
    du_ub_[t] = Eigen::VectorXd::Zero(nu);
  }
}

void SolverBoxDDP::computeGains(const std::size_t t) {
  START_PROFILER("SolverBoxDDP::computeGains");
  const std::size_t nu = problem_->get_runningModels()[t]->get_nu();
  if (nu > 0) {
    if (!problem_->get_runningModels()[t]->get_has_control_limits() || !is_feasible_) {
      // No control limits on this model: Use vanilla DDP
      SolverDDP::computeGains(t);
      return;
    }

    du_lb_[t] = problem_->get_runningModels()[t]->get_u_lb() - us_[t];
    du_ub_[t] = problem_->get_runningModels()[t]->get_u_ub() - us_[t];

    START_PROFILER("SolverBoxDDP::boxQP");
    const BoxQPSolution& boxqp_sol = qp_.solve(Quu_[t], Qu_[t], du_lb_[t], du_ub_[t], k_[t]);
    START_PROFILER("SolverBoxDDP::boxQP");

    // Compute controls
    START_PROFILER("SolverBoxDDP::Quu_invproj");
    Quu_inv_[t].setZero();
    for (std::size_t i = 0; i < boxqp_sol.free_idx.size(); ++i) {
      for (std::size_t j = 0; j < boxqp_sol.free_idx.size(); ++j) {
        Quu_inv_[t](boxqp_sol.free_idx[i], boxqp_sol.free_idx[j]) = boxqp_sol.Hff_inv(i, j);
      }
    }
    STOP_PROFILER("SolverBoxDDP::Quu_invproj");
    START_PROFILER("SolverBoxDDP::Quu_invproj_Qxu");
    K_[t].noalias() = Quu_inv_[t] * Qxu_[t].transpose();
    STOP_PROFILER("SolverBoxDDP::Quu_invproj_Qxu");
    k_[t] = -boxqp_sol.x;

    // The box-QP clamped the gradient direction; this is important for accounting
    // the algorithm advancement (i.e. stopping criteria)
    START_PROFILER("SolverBoxDDP::Qu_proj");
    for (std::size_t i = 0; i < boxqp_sol.clamped_idx.size(); ++i) {
      Qu_[t](boxqp_sol.clamped_idx[i]) = 0.;
    }
    STOP_PROFILER("SolverBoxDDP::Qu_proj");
  }
  STOP_PROFILER("SolverBoxDDP::computeGains");
}

void SolverBoxDDP::forwardPass(double steplength) {
  if (steplength > 1. || steplength < 0.) {
    throw_pretty("Invalid argument: "
                 << "invalid step length, value is between 0. to 1.");
  }
  START_PROFILER("SolverBoxDDP::forwardPass");
  cost_try_ = 0.;
  xnext_ = problem_->get_x0();
  const std::size_t T = problem_->get_T();
  const std::vector<boost::shared_ptr<ActionModelAbstract> >& models = problem_->get_runningModels();
  const std::vector<boost::shared_ptr<ActionDataAbstract> >& datas = problem_->get_runningDatas();
  for (std::size_t t = 0; t < T; ++t) {
    const boost::shared_ptr<ActionModelAbstract>& m = models[t];
    const boost::shared_ptr<ActionDataAbstract>& d = datas[t];
    const std::size_t nu = m->get_nu();

    xs_try_[t] = xnext_;
    m->get_state()->diff(xs_[t], xs_try_[t], dx_[t]);
    if (nu != 0) {
      us_try_[t].noalias() = us_[t] - k_[t] * steplength - K_[t] * dx_[t];
      if (m->get_has_control_limits()) {  // clamp control
        us_try_[t] = us_try_[t].cwiseMax(m->get_u_lb()).cwiseMin(m->get_u_ub());
      }
      m->calc(d, xs_try_[t], us_try_[t]);
    } else {
      m->calc(d, xs_try_[t]);
    }
    xnext_ = d->xnext;
    cost_try_ += d->cost;

    if (raiseIfNaN(cost_try_)) {
      STOP_PROFILER("SolverBoxDDP::forwardPass");
      throw_pretty("forward_error");
    }
    if (raiseIfNaN(xnext_.lpNorm<Eigen::Infinity>())) {
      STOP_PROFILER("SolverBoxDDP::forwardPass");
      throw_pretty("forward_error");
    }
  }

  const boost::shared_ptr<ActionModelAbstract>& m = problem_->get_terminalModel();
  const boost::shared_ptr<ActionDataAbstract>& d = problem_->get_terminalData();
  if ((is_feasible_) || (steplength == 1)) {
    xs_try_.back() = xnext_;
  } else {
    m->get_state()->integrate(xnext_, fs_.back() * (steplength - 1), xs_try_.back());
  }
  m->calc(d, xs_try_.back());
  cost_try_ += d->cost;

  if (raiseIfNaN(cost_try_)) {
    STOP_PROFILER("SolverBoxDDP::forwardPass");
    throw_pretty("forward_error");
  }
}

const std::vector<Eigen::MatrixXd>& SolverBoxDDP::get_Quu_inv() const { return Quu_inv_; }

}  // namespace crocoddyl
