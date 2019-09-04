///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS, New York University, Max Planck Gesellschaft
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////


#include "crocoddyl/core/solvers/kkt.hpp"

namespace crocoddyl {

    SolverKKT::SolverKKT(ShootingProblem& problem)
    : SolverAbstract(problem),
      regfactor_(10.),
      regmin_(1e-9),
      regmax_(1e9),
      cost_try_(0.),
      th_grad_(1e-12),
      th_step_(0.5),
      was_feasible_(false) {
  allocateData();

  const unsigned int& n_alphas = 10;
  alphas_.resize(n_alphas);
  for (unsigned int n = 0; n < n_alphas; ++n) {
    alphas_[n] = 1. / pow(2., (double)n);
  }
}

SolverKKT::~SolverKKT() {}


void SolverKKT::allocateData() {
  const long unsigned int& T = problem_.get_T();
 
  gaps_.resize(T + 1);

  xs_try_.resize(T + 1);
  us_try_.resize(T);
  dx_.resize(T);

  for (long unsigned int t = 0; t < T; ++t) {
    ActionModelAbstract* model = problem_.running_models_[t];
    const int& nx = model->get_nx();
    const int& ndx = model->get_ndx();
    const int& nu = model->get_nu();

    if (t == 0) {
      xs_try_[t] = problem_.get_x0();
    } else {
      xs_try_[t] = Eigen::VectorXd::Constant(nx, NAN);
    }
    us_try_[t] = Eigen::VectorXd::Constant(nu, NAN);
    dx_[t] = Eigen::VectorXd::Zero(ndx);
  }

}






}  // namespace crocoddyl