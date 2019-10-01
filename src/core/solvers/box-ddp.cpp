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
  
  const unsigned int& T = problem_.get_T();
  Quu_inv_.resize(T);
  for (unsigned int t = 0; t < T; ++t) {
    ActionModelAbstract* model = problem_.running_models_[t];
    const unsigned int& nu = model->get_nu();
   
    Quu_inv_[t] = Eigen::MatrixXd::Zero(nu, nu);
  }
}

bool SolverBoxDDP::solve(const std::vector<Eigen::VectorXd>& init_xs, const std::vector<Eigen::VectorXd>& init_us,
                      const unsigned int& maxiter, const bool& is_feasible, const double& reginit) {
  return SolverDDP::solve(init_xs, init_us, maxiter, is_feasible, reginit);

//   if (std::isnan(reginit)) {
//     xreg_ = regmin_;
//     ureg_ = regmin_;
//   } else {
//     xreg_ = reginit;
//     ureg_ = reginit;
//   }
//   was_feasible_ = false;

//   bool recalc = true;
//   for (iter_ = 0; iter_ < maxiter; ++iter_) {
//     while (true) {
//       try {
//         computeDirection(recalc);
//       } catch (const char* msg) {
//         recalc = false;
//         increaseRegularization();
//         if (xreg_ == regmax_) {
//           return false;
//         } else {
//           continue;
//         }
//       }
//       break;
//     }
//     expectedImprovement();

//     // We need to recalculate the derivatives when the step length passes
//     recalc = false;
//     for (std::vector<double>::const_iterator it = alphas_.begin(); it != alphas_.end(); ++it) {
//       steplength_ = *it;

//       try {
//         dV_ = tryStep(steplength_);
//       } catch (const char* msg) {
//         continue;
//       }
//       dVexp_ = steplength_ * (d_[0] + 0.5 * steplength_ * d_[1]);

//       if (d_[0] < th_grad_ || !is_feasible_ || dV_ > th_acceptstep_ * dVexp_) {
//         was_feasible_ = is_feasible_;
//         setCandidate(xs_try_, us_try_, true);
//         cost_ = cost_try_;
//         recalc = true;
//         break;
//       }
//     }

//     if (steplength_ > th_step_) {
//       decreaseRegularization();
//     }
//     if (steplength_ == alphas_.back()) {
//       increaseRegularization();
//       if (xreg_ == regmax_) {
//         return false;
//       }
//     }
//     stoppingCriteria();

//     unsigned int const& n_callbacks = static_cast<unsigned int>(callbacks_.size());
//     for (unsigned int c = 0; c < n_callbacks; ++c) {
//       CallbackAbstract& callback = *callbacks_[c];
//       callback(*this);
//     }

//     if (was_feasible_ && stop_ < th_stop_) {
//       return true;
//     }
//   }
//   return false;
}

void SolverBoxDDP::computeDirection(const bool& recalc) {
  if (recalc) {
    calc();
  }
  backwardPass();
}

void SolverBoxDDP::backwardPass() {
  boost::shared_ptr<ActionDataAbstract>& d_T = problem_.terminal_data_;
  Vxx_.back() = d_T->get_Lxx();
  Vx_.back() = d_T->get_Lx();

  x_reg_.fill(xreg_);
  if (!std::isnan(xreg_)) {
    Vxx_.back().diagonal() += x_reg_;
  }

  if (!is_feasible_) {
    Vx_.back().noalias() += Vxx_.back() * gaps_.back();
  }

  for (int t = static_cast<int>(problem_.get_T()) - 1; t >= 0; --t) {
    ActionModelAbstract* m = problem_.running_models_[t];
    boost::shared_ptr<ActionDataAbstract>& d = problem_.running_datas_[t];
    const Eigen::MatrixXd& Vxx_p = Vxx_[t + 1];
    const Eigen::VectorXd& Vx_p = Vx_[t + 1];

    FxTVxx_p_.noalias() = d->get_Fx().transpose() * Vxx_p;
    FuTVxx_p_[t].noalias() = d->get_Fu().transpose() * Vxx_p;
    Qxx_[t].noalias() = d->get_Lxx() + FxTVxx_p_ * d->get_Fx();
    Qxu_[t].noalias() = d->get_Lxu() + FxTVxx_p_ * d->get_Fu();
    Quu_[t].noalias() = d->get_Luu() + FuTVxx_p_[t] * d->get_Fu();
    Qx_[t].noalias() = d->get_Lx() + d->get_Fx().transpose() * Vx_p;
    Qu_[t].noalias() = d->get_Lu() + d->get_Fu().transpose() * Vx_p;

    if (!std::isnan(ureg_)) {
      unsigned int const& nu = m->get_nu();
      Quu_[t].diagonal() += Eigen::VectorXd::Constant(nu, ureg_);
    }

    computeGains(t);

    if (std::isnan(ureg_)) {
      Vx_[t].noalias() = Qx_[t] - K_[t].transpose() * Qu_[t];
    } else {
      Quuk_[t].noalias() = Quu_[t] * k_[t];
      Vx_[t].noalias() = Qx_[t] + K_[t].transpose() * Quuk_[t] - 2 * K_[t].transpose() * Qu_[t];
    }
    Vxx_[t].noalias() = Qxx_[t] - Qxu_[t] * K_[t];
    Vxx_[t] = 0.5 * (Vxx_[t] + Vxx_[t].transpose()).eval();  // TODO(cmastalli): as suggested by Nicolas

    if (!std::isnan(xreg_)) {
      Vxx_[t].diagonal() += x_reg_;
    }

    // Compute and store the Vx gradient at end of the interval (rollout state)
    if (!is_feasible_) {
      Vx_[t].noalias() += Vxx_[t] * gaps_[t];
    }

    if (raiseIfNaN(Vx_[t].lpNorm<Eigen::Infinity>())) {
      throw "backward_error";
    }
    if (raiseIfNaN(Vxx_[t].lpNorm<Eigen::Infinity>())) {
      throw "backward_error";
    }
  }
}

void SolverBoxDDP::computeGains(const unsigned int& t) {
  if (problem_.running_models_[t]->get_nu() > 0) {
      if (!problem_.running_models_[t]->get_has_control_limits()) {
          std::cerr << "NOT LIMITED!!" << std::endl;
      }
    Eigen::VectorXd low_limit = problem_.running_models_[t]->get_u_lower_limit() - us_[t],
                    high_limit = problem_.running_models_[t]->get_u_upper_limit() - us_[t];

    // std::cout << "[" << t << "] low_limit: " << low_limit.transpose() << std::endl;
    // std::cout << "[" << t << "] high_limit: " << high_limit.transpose() << std::endl;
    // std::cout << "[" << t << "] us_[t]: " << us_[t].transpose() << std::endl;

    exotica::BoxQPSolution boxqp_sol = exotica::BoxQP(Quu_[t], Qu_[t], low_limit, high_limit, us_[t]); //, 0.1, 100, 1e-5, 0.0001);

    Quu_inv_[t].setZero();
    for (size_t i = 0; i < boxqp_sol.free_idx.size(); ++i)
        for (size_t j = 0; j < boxqp_sol.free_idx.size(); ++j)
            Quu_inv_[t](boxqp_sol.free_idx[i], boxqp_sol.free_idx[j]) = boxqp_sol.Hff_inv(i, j);

    // Compute controls
    K_[t] = -Quu_inv_[t] * Qxu_[t];
    k_[t] = boxqp_sol.x;

    for (size_t j = 0; j < boxqp_sol.clamped_idx.size(); ++j)
        K_[t](boxqp_sol.clamped_idx[j]) = 0.0;

    // std::cout << "K_[" << t << "]: " << K_[t] << std::endl;
  }
}

}  // namespace crocoddyl
