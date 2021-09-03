///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "crocoddyl/core/solvers/ddp.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/utils/stop-watch.hpp"

namespace crocoddyl {

SolverIntro::SolverIntro(boost::shared_ptr<ShootingProblem> problem)
    : SolverDDP(problem),
    dPhi_(0.),
    rho_(0.3),
    hfeas_try_(0.),
    upsilon_(0.) {
  reg_incFactor = 1e6;
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
      upsilon_ = std::max(upsilon_, (d[0] + .5 * d[1]) / ((1 - rho_) * hfeas_));
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

      if (abs(d[0]) > th_grad_ || !is_feasible || dPhi_ > th_acceptstep_ * dPhiexp_)
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
  SolverDDP::tryStep();
  computeEqualityFeasibility();
  return cost_ - cost_try_;
}

double SolverIntro::stoppingCriteria() {
  stop_ = std::max(hfeas_, abs(d_[0] + 0.5 * d_[1]));
  return stop_;
}

void SolverIntro::computeGains(const std::size_t t) {
  SolverDDP::computeGains(t);
}

double SolverIntro::get_rho() const { return rho_; }

void SolverIntro::set_rho(const double rho) {
  if (0. >= rho || rho > 1.) {
    throw_pretty("Invalid argument: "
                 << "rho value should between 0 and 1.");
  }
  rho_ = rho;
}

}  // namespace crocoddyl
