///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019, The University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include "crocoddyl/core/solvers/box-ddp.hpp"

namespace crocoddyl {

SolverBoxDDP::SolverBoxDDP(ShootingProblem& problem) : SolverFDDP(problem) {
  allocateData();

  const unsigned int& n_alphas = 10;
  alphas_.resize(n_alphas);
  for (unsigned int n = 0; n < n_alphas; ++n) {
    alphas_[n] = 1. / pow(2., static_cast<double>(n));
  }
}

SolverBoxDDP::~SolverBoxDDP() {}

void SolverBoxDDP::allocateData() {
  SolverFDDP::allocateData();
  
  const unsigned int& T = problem_.get_T();
  Quu_inv_.resize(T);
  for (unsigned int t = 0; t < T; ++t) {
    ActionModelAbstract* model = problem_.running_models_[t];
    const unsigned int& nu = model->get_nu();
   
    Quu_inv_[t] = Eigen::MatrixXd::Zero(nu, nu);
  }
}

void SolverBoxDDP::computeGains(const unsigned int& t) {
  if (problem_.running_models_[t]->get_nu() > 0) {
    if (!problem_.running_models_[t]->get_has_control_limits()) {
      std::cerr << "NOT LIMITED!!" << problem_.running_models_[t]->get_u_lower_limit() << std::endl;
      SolverFDDP::computeGains(t);
      return;
    }
    Eigen::VectorXd low_limit = problem_.running_models_[t]->get_u_lower_limit() - us_[t],
                    high_limit = problem_.running_models_[t]->get_u_upper_limit() - us_[t];

    // std::cout << "[" << t << "] low_limit: " << low_limit.transpose() << std::endl;
    // std::cout << "[" << t << "] high_limit: " << high_limit.transpose() << std::endl;
    // std::cout << "[" << t << "] us_[t]: " << us_[t].transpose() << std::endl;

    const double regularisation = 0.001;
    exotica::BoxQPSolution boxqp_sol = exotica::BoxQP(Quu_[t], Qu_[t], low_limit, high_limit, us_[t], 0.1, 100, 1e-5, regularisation);

    Quu_inv_[t].setZero();
    for (size_t i = 0; i < boxqp_sol.free_idx.size(); ++i)
      for (size_t j = 0; j < boxqp_sol.free_idx.size(); ++j)
        Quu_inv_[t](boxqp_sol.free_idx[i], boxqp_sol.free_idx[j]) = boxqp_sol.Hff_inv(i, j);

    // Compute controls
    K_[t] = Quu_inv_[t] * Qxu_[t].transpose();
    k_[t] = - boxqp_sol.x;

    // if (boxqp_sol.clamped_idx.size() > 0)
    //   std::cout << "clamped_idx.size() = " << boxqp_sol.clamped_idx.size() << std::endl;

    for (size_t j = 0; j < boxqp_sol.clamped_idx.size(); ++j)
      K_[t](boxqp_sol.clamped_idx[j]) = 0.0;

    // Compare with good old unconstrained
    // std::cout << "[Box-QP]: K_["<<t<<"]:" << K_[t] .transpose()<<std::endl;
    // Quu_llt_[t].compute(Quu_[t]);
    // K_[t] = Qxu_[t].transpose();
    // Quu_llt_[t].solveInPlace(K_[t]);
    // k_[t] = Qu_[t];
    // Quu_llt_[t].solveInPlace(k_[t]);
    // std::cout << "[Inverse]: K_["<<t<<"]:" << K_[t] .transpose()<<std::endl;

    // std::cout << "K_[" << t << "]: " << K_[t] << std::endl;
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
      // us_try_[t].noalias() = us_try_[t].cwiseMax(m->get_u_lower_limit()).cwiseMin(m->get_u_upper_limit());
    }

    m->calc(d, xs_try_[t], us_try_[t]);
    xnext_ = d->get_xnext();
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
