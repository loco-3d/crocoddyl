///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2022, LAAS-CNRS, University of Edinburgh
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifdef CROCODDYL_WITH_MULTITHREADING
#include <omp.h>
#endif  // CROCODDYL_WITH_MULTITHREADING

#include "crocoddyl/core/solvers/fddp.hpp"

namespace crocoddyl {

SolverFDDP::SolverFDDP(std::shared_ptr<ShootingProblem> problem)
    : SolverDDP(problem), dg_(0), dq_(0), dv_(0), th_acceptnegstep_(2) {}

SolverFDDP::~SolverFDDP() {}

bool SolverFDDP::solve(const std::vector<Eigen::VectorXd>& init_xs,
                       const std::vector<Eigen::VectorXd>& init_us,
                       const std::size_t maxiter, const bool is_feasible,
                       const double init_reg) {
  START_PROFILER("SolverFDDP::solve");
  if (problem_->is_updated()) {
    resizeData();
  }
  xs_try_[0] =
      problem_->get_x0();  // it is needed in case that init_xs[0] is infeasible
  setCandidate(init_xs, init_us, is_feasible);

  if (std::isnan(init_reg)) {
    preg_ = reg_min_;
    dreg_ = reg_min_;
  } else {
    preg_ = init_reg;
    dreg_ = init_reg;
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
        if (preg_ == reg_max_) {
          return false;
        } else {
          continue;
        }
      }
      break;
    }
    updateExpectedImprovement();

    // We need to recalculate the derivatives when the step length passes
    recalcDiff = false;
    for (std::vector<double>::const_iterator it = alphas_.begin();
         it != alphas_.end(); ++it) {
      steplength_ = *it;

      try {
        dV_ = tryStep(steplength_);
      } catch (std::exception& e) {
        continue;
      }
      expectedImprovement();
      dVexp_ = steplength_ * (d_[0] + 0.5 * steplength_ * d_[1]);

      if (dVexp_ >= 0) {  // descend direction
        if (std::abs(d_[0]) < th_grad_ || dV_ > th_acceptstep_ * dVexp_) {
          was_feasible_ = is_feasible_;
          setCandidate(xs_try_, us_try_, (was_feasible_) || (steplength_ == 1));
          cost_ = cost_try_;
          recalcDiff = true;
          break;
        }
      } else {  // reducing the gaps by allowing a small increment in the cost
                // value
        if (!is_feasible_ && dV_ > th_acceptnegstep_ * dVexp_) {
          was_feasible_ = is_feasible_;
          setCandidate(xs_try_, us_try_, (was_feasible_) || (steplength_ == 1));
          cost_ = cost_try_;
          recalcDiff = true;
          break;
        }
      }
    }

    if (steplength_ > th_stepdec_) {
      decreaseRegularization();
    }
    if (steplength_ <= th_stepinc_) {
      increaseRegularization();
      if (preg_ == reg_max_) {
        STOP_PROFILER("SolverFDDP::solve");
        return false;
      }
    }
    stoppingCriteria();

    const std::size_t n_callbacks = callbacks_.size();
    for (std::size_t c = 0; c < n_callbacks; ++c) {
      CallbackAbstract& callback = *callbacks_[c];
      callback(*this);
    }

    if (was_feasible_ && stop_ < th_stop_) {
      STOP_PROFILER("SolverFDDP::solve");
      return true;
    }
  }
  STOP_PROFILER("SolverFDDP::solve");
  return false;
}

const Eigen::Vector2d& SolverFDDP::expectedImprovement() {
  dv_ = 0;
  const std::size_t T = this->problem_->get_T();
  if (!is_feasible_) {
    // NB: The dimension of vectors xs_try_ and xs_ are T+1, whereas the
    // dimension of dx_ is T. Here, we are re-using the final element of dx_ for
    // the computation of the difference at the terminal node. Using the access
    // iterator back() this re-use of the final element is fine. Cf. the
    // discussion at https://github.com/loco-3d/crocoddyl/issues/1022
    problem_->get_terminalModel()->get_state()->diff(xs_try_.back(), xs_.back(),
                                                     dx_.back());
    fTVxx_p_.noalias() = Vxx_.back() * dx_.back();
    dv_ -= fs_.back().dot(fTVxx_p_);
    const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
        problem_->get_runningModels();

    for (std::size_t t = 0; t < T; ++t) {
      models[t]->get_state()->diff(xs_try_[t], xs_[t], dx_[t]);
      fTVxx_p_.noalias() = Vxx_[t] * dx_[t];
      dv_ -= fs_[t].dot(fTVxx_p_);
    }
  }
  d_[0] = dg_ + dv_;
  d_[1] = dq_ - 2 * dv_;
  return d_;
}

void SolverFDDP::updateExpectedImprovement() {
  dg_ = 0;
  dq_ = 0;
  const std::size_t T = this->problem_->get_T();
  if (!is_feasible_) {
    dg_ -= Vx_.back().dot(fs_.back());
    fTVxx_p_.noalias() = Vxx_.back() * fs_.back();
    dq_ += fs_.back().dot(fTVxx_p_);
  }
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::size_t nu = models[t]->get_nu();
    if (nu != 0) {
      dg_ += Qu_[t].dot(k_[t]);
      dq_ -= k_[t].dot(Quuk_[t]);
    }
    if (!is_feasible_) {
      dg_ -= Vx_[t].dot(fs_[t]);
      fTVxx_p_.noalias() = Vxx_[t] * fs_[t];
      dq_ += fs_[t].dot(fTVxx_p_);
    }
  }
}

void SolverFDDP::forwardPass(const double steplength) {
  if (steplength > 1. || steplength < 0.) {
    throw_pretty("Invalid argument: "
                 << "invalid step length, value is between 0. to 1.");
  }
  START_PROFILER("SolverFDDP::forwardPass");
  cost_try_ = 0.;
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
        m->calc(d, xs_try_[t], us_try_[t]);
      } else {
        m->calc(d, xs_try_[t]);
      }
      xnext_ = d->xnext;
      cost_try_ += d->cost;

      if (raiseIfNaN(cost_try_)) {
        STOP_PROFILER("SolverFDDP::forwardPass");
        throw_pretty("forward_error");
      }
      if (raiseIfNaN(xnext_.lpNorm<Eigen::Infinity>())) {
        STOP_PROFILER("SolverFDDP::forwardPass");
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
      STOP_PROFILER("SolverFDDP::forwardPass");
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
        m->calc(d, xs_try_[t], us_try_[t]);
      } else {
        m->calc(d, xs_try_[t]);
      }
      xnext_ = d->xnext;
      cost_try_ += d->cost;

      if (raiseIfNaN(cost_try_)) {
        STOP_PROFILER("SolverFDDP::forwardPass");
        throw_pretty("forward_error");
      }
      if (raiseIfNaN(xnext_.lpNorm<Eigen::Infinity>())) {
        STOP_PROFILER("SolverFDDP::forwardPass");
        throw_pretty("forward_error");
      }
    }

    const std::shared_ptr<ActionModelAbstract>& m =
        problem_->get_terminalModel();
    const std::shared_ptr<ActionDataAbstract>& d = problem_->get_terminalData();
    m->get_state()->integrate(xnext_, fs_.back() * (steplength - 1),
                              xs_try_.back());
    m->calc(d, xs_try_.back());
    cost_try_ += d->cost;

    if (raiseIfNaN(cost_try_)) {
      STOP_PROFILER("SolverFDDP::forwardPass");
      throw_pretty("forward_error");
    }
  }
  STOP_PROFILER("SolverFDDP::forwardPass");
}

double SolverFDDP::get_th_acceptnegstep() const { return th_acceptnegstep_; }

void SolverFDDP::set_th_acceptnegstep(const double th_acceptnegstep) {
  if (0. > th_acceptnegstep) {
    throw_pretty(
        "Invalid argument: " << "th_acceptnegstep value has to be positive.");
  }
  th_acceptnegstep_ = th_acceptnegstep;
}

}  // namespace crocoddyl
