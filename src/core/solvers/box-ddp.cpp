///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019, CNRS-LAAS, The University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include "crocoddyl/core/solvers/box-ddp.hpp"

namespace crocoddyl {

SolverBoxDDP::SolverBoxDDP(ShootingProblem& problem) : SolverDDP(problem) {
  allocateData();

  const unsigned int& n_alphas = 10;
  alphas_.resize(n_alphas);
  for (unsigned int n = 0; n < n_alphas; ++n) {
    alphas_[n] = 1. / pow(2., static_cast<double>(n));
  }
}

SolverBoxDDP::~SolverBoxDDP() {}

void SolverBoxDDP::allocateData() {
  SolverDDP::allocateData();

  unsigned int nu_max = 0;
  unsigned int const& T = problem_.get_T();
  Quu_inv_.resize(T);
  for (unsigned int t = 0; t < T; ++t) {
    ActionModelAbstract* model = problem_.running_models_[t];
    unsigned int const& nu = model->get_nu();

    // Store the largest number of controls across all models to allocate u_ll_, u_hl_
    if (nu > nu_max) nu_max = nu;

    Quu_inv_[t] = Eigen::MatrixXd::Zero(nu, nu);
  }

  u_ll_.resize(nu_max);
  u_hl_.resize(nu_max);
}

void SolverBoxDDP::computeGains(const unsigned int& t) {
  if (problem_.running_models_[t]->get_nu() > 0) {
    if (!problem_.running_models_[t]->get_has_control_limits()) {
      // No control limits on this model: Use vanilla DDP
      SolverDDP::computeGains(t);
      return;
    }

    u_ll_ = problem_.running_models_[t]->get_u_lb() - us_[t];
    u_hl_ = problem_.running_models_[t]->get_u_ub() - us_[t];

    BoxQPSolution boxqp_sol = BoxQP(Quu_[t], Qu_[t], u_ll_, u_hl_, us_[t], 0.1, 100, 1e-5, ureg_);

    Quu_inv_[t].setZero();
    for (size_t i = 0; i < boxqp_sol.free_idx.size(); ++i)
      for (size_t j = 0; j < boxqp_sol.free_idx.size(); ++j)
        Quu_inv_[t](boxqp_sol.free_idx[i], boxqp_sol.free_idx[j]) = boxqp_sol.Hff_inv(i, j);

    // Compute controls
    K_[t].noalias() = Quu_inv_[t] * Qxu_[t].transpose();
    k_[t].noalias() = -boxqp_sol.x;

    for (size_t j = 0; j < boxqp_sol.clamped_idx.size(); ++j) K_[t](boxqp_sol.clamped_idx[j]) = 0.0;
  }
}

void SolverBoxDDP::forwardPass(const double& steplength) {
  assert(steplength <= 1. && "Step length has to be <= 1.");
  assert(steplength >= 0. && "Step length has to be >= 0.");
  cost_try_ = 0.;
  xnext_ = problem_.get_x0();
  unsigned int const& T = problem_.get_T();
  for (unsigned int t = 0; t < T; ++t) {
    ActionModelAbstract* m = problem_.running_models_[t];
    boost::shared_ptr<ActionDataAbstract>& d = problem_.running_datas_[t];
    if ((is_feasible_) || (steplength == 1)) {
      xs_try_[t] = xnext_;
    } else {
      m->get_state().integrate(xnext_, gaps_[t] * (steplength - 1), xs_try_[t]);
    }
    m->get_state().diff(xs_[t], xs_try_[t], dx_[t]);
    us_try_[t].noalias() = us_[t] - k_[t] * steplength - K_[t] * dx_[t];

    // Clamp!
    if (m->get_has_control_limits()) {
      us_try_[t] = us_try_[t].cwiseMax(m->get_u_lb()).cwiseMin(m->get_u_ub());
    }

    m->calc(d, xs_try_[t], us_try_[t]);
    xnext_ = d->xnext;
    cost_try_ += d->cost;

    if (raiseIfNaN(cost_try_)) {
      throw "forward_error";
    }
    if (raiseIfNaN(xnext_.lpNorm<Eigen::Infinity>())) {
      throw "forward_error";
    }
  }

  ActionModelAbstract* m = problem_.terminal_model_;
  boost::shared_ptr<ActionDataAbstract>& d = problem_.terminal_data_;

  if ((is_feasible_) || (steplength == 1)) {
    xs_try_.back() = xnext_;
  } else {
    m->get_state().integrate(xnext_, gaps_.back() * (steplength - 1), xs_try_.back());
  }
  m->calc(d, xs_try_.back());
  cost_try_ += d->cost;

  if (raiseIfNaN(cost_try_)) {
    throw "forward_error";
  }
}

}  // namespace crocoddyl
