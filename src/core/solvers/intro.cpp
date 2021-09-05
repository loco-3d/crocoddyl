///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "crocoddyl/core/solvers/intro.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/utils/stop-watch.hpp"

namespace crocoddyl {

SolverIntro::SolverIntro(boost::shared_ptr<ShootingProblem> problem)
    : SolverDDP(problem), rho_(0.3), dPhi_(0.), hfeas_try_(0.), upsilon_(0.) {
  reg_incfactor_ = 1e6;

  const std::size_t T = problem_->get_T();
  k_hat_.resize(T);
  K_hat_.resize(T);
  Quu_hat_.resize(T);
  QuuinvHuT_.resize(T);
  Quu_hat_llt_.resize(T);

  const std::size_t ndx = problem_->get_ndx();
  const std::size_t nu = problem_->get_nu_max();
  const std::vector<boost::shared_ptr<ActionModelAbstract> >& models = problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const boost::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nh = model->get_nh();

    k_hat_[t] = Eigen::VectorXd::Zero(nu);
    K_hat_[t] = Eigen::MatrixXd::Zero(nh, ndx);
    Quu_hat_[t] = Eigen::MatrixXd::Zero(nh, nh);
    QuuinvHuT_[t] = Eigen::MatrixXd::Zero(nu, nh);
    Quu_hat_llt_[t] = Eigen::LLT<Eigen::MatrixXd>(nh);
  }
}

SolverIntro::~SolverIntro() {}

bool SolverIntro::solve(const std::vector<Eigen::VectorXd>& init_xs, const std::vector<Eigen::VectorXd>& init_us,
                        const std::size_t maxiter, const bool is_feasible, const double reginit) {
  START_PROFILER("SolverIntro::solve");
  // xs_try_[0] = problem_->get_x0();  // it is needed in case that init_xs[0] is infeasible
  setCandidate(init_xs, init_us, is_feasible);

  if (std::isnan(reginit)) {
    xreg_ = reg_min_;
    ureg_ = reg_min_;
  } else {
    xreg_ = reginit;
    ureg_ = reginit;
  }
  was_feasible_ = false;
  upsilon_ = 0.;

  bool recalcDiff = true;
  for (iter_ = 0; iter_ < maxiter; ++iter_) {
    while (true) {
      try {
        computeDirection(recalcDiff);
      } catch (std::exception& e) {
        recalcDiff = false;
        increaseRegularization();
        if (xreg_ == reg_max_) {
          return false;
        } else {
          continue;
        }
      }
      break;
    }
    expectedImprovement();

    // Update the penalty parameter for computing the merit function and its directional derivative
    // For more details see Section 3 of "An Interior Point Algorithm for Large Scale Nonlinear Programming"
    if (hfeas_ != 0) {
      upsilon_ = std::max(upsilon_, (d_[0] + .5 * d_[1]) / ((1 - rho_) * hfeas_));
    }

    // We need to recalculate the derivatives when the step length passes
    for (std::vector<double>::const_iterator it = alphas_.begin(); it != alphas_.end(); ++it) {
      steplength_ = *it;

      try {
        dV_ = tryStep(steplength_);
        dPhi_ = dV_ + upsilon_ * (hfeas_ - hfeas_try_);
      } catch (std::exception& e) {
        continue;
      }
      dVexp_ = steplength_ * (d_[0] + 0.5 * steplength_ * d_[1]);
      dPhiexp_ = dVexp_ + steplength_ * upsilon_ * hfeas_;

      if (abs(d_[0]) < th_grad_ || !is_feasible || dPhi_ > th_acceptstep_ * dPhiexp_) {
        was_feasible_ = is_feasible_;
        setCandidate(xs_try_, us_try_, true);
        cost_ = cost_try_;
        hfeas_ = hfeas_try_;
        break;
      }
    }

    stoppingCriteria();
    const std::size_t n_callbacks = callbacks_.size();
    for (std::size_t c = 0; c < n_callbacks; ++c) {
      CallbackAbstract& callback = *callbacks_[c];
      callback(*this);
    }

    if (steplength_ > th_stepdec_ && dV_ >= 0.) {
      decreaseRegularization();
    }
    if (steplength_ <= th_stepinc_) {
      increaseRegularization();
      if (xreg_ == reg_max_) {
        STOP_PROFILER("SolverIntro::solve");
        return false;
      }
    }

    if (was_feasible_ && stop_ < th_stop_) {
      STOP_PROFILER("SolverIntro::solve");
      return true;
    }
  }
  STOP_PROFILER("SolverIntro::solve");
  return false;
}

double SolverIntro::tryStep(const double steplength) {
  forwardPass(steplength);
  hfeas_try_ = computeEqualityFeasibility();
  return cost_ - cost_try_;
}

double SolverIntro::stoppingCriteria() {
  stop_ = std::max(hfeas_, abs(d_[0] + 0.5 * d_[1]));
  return stop_;
}

void SolverIntro::computeGains(const std::size_t t) {
  SolverDDP::computeGains(t);
  const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = problem_->get_runningModels()[t];
  const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = problem_->get_runningDatas()[t];

  const std::size_t nu = model->get_nu();
  if (nu > 0 && model->get_nh() > 0) {
    QuuinvHuT_[t] = data->Hu.transpose();
    Quu_llt_[t].solveInPlace(QuuinvHuT_[t]);
    Quu_hat_[t].noalias() = data->Hu * QuuinvHuT_[t];
    Quu_hat_llt_[t].compute(Quu_hat_[t]);
    const Eigen::ComputationInfo& info = Quu_hat_llt_[t].info();
    if (info != Eigen::Success) {
      throw_pretty("backward error");
    }
    Eigen::Transpose<Eigen::MatrixXd> HuQuuinv = QuuinvHuT_[t].transpose();
    Quu_hat_llt_[t].solveInPlace(HuQuuinv);

    k_hat_[t] = data->h;
    k_hat_[t].noalias() -= data->Hu * k_[t];
    K_hat_[t] = data->Hx;
    K_hat_[t].noalias() -= data->Hu * K_[t];
    k_[t].noalias() += (k_hat_[t].transpose() * HuQuuinv).transpose();
    K_[t].noalias() += (K_hat_[t].transpose() * HuQuuinv).transpose();
  }
}

double SolverIntro::get_rho() const { return rho_; }

double SolverIntro::get_dPhi() const { return dPhi_; }

double SolverIntro::get_dPhiexp() const { return dPhiexp_; }

double SolverIntro::get_upsilon() const { return upsilon_; }

void SolverIntro::set_rho(const double rho) {
  if (0. >= rho || rho > 1.) {
    throw_pretty("Invalid argument: "
                 << "rho value should between 0 and 1.");
  }
  rho_ = rho;
}

}  // namespace crocoddyl
