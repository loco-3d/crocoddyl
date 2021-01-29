///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/solver-base.hpp"

namespace crocoddyl {

SolverAbstract::SolverAbstract(boost::shared_ptr<ShootingProblem> problem)
    : problem_(problem),
      is_feasible_(false),
      cost_(0.),
      stop_(0.),
      xreg_(NAN),
      ureg_(NAN),
      steplength_(1.),
      dV_(0.),
      dVexp_(0.),
      th_acceptstep_(0.1),
      th_stop_(1e-9),
      iter_(0) {
  // Allocate common data
  const std::size_t& T = problem_->get_T();
  xs_.resize(T + 1);
  us_.resize(T);
  for (std::size_t t = 0; t < T; ++t) {
    const boost::shared_ptr<ActionModelAbstract>& model = problem_->get_runningModels()[t];

    xs_[t] = model->get_state()->zero();
    us_[t] = Eigen::VectorXd::Zero(problem_->get_nu_max());
  }
  xs_.back() = problem_->get_terminalModel()->get_state()->zero();
}

SolverAbstract::~SolverAbstract() {}

void SolverAbstract::setCandidate(const std::vector<Eigen::VectorXd>& xs_warm,
                                  const std::vector<Eigen::VectorXd>& us_warm, const bool& is_feasible) {
  const std::size_t& T = problem_->get_T();

  if (xs_warm.size() == 0) {
    for (std::size_t t = 0; t < T; ++t) {
      xs_[t] = problem_->get_runningModels()[t]->get_state()->zero();
    }
    xs_.back() = problem_->get_terminalModel()->get_state()->zero();
  } else {
    if (xs_warm.size() != T + 1) {
      throw_pretty("Warm start state has wrong dimension, got " << xs_warm.size() << " expecting " << (T + 1));
    }
    for (std::size_t t = 0; t < T; ++t) {
      const std::size_t& nx = problem_->get_runningModels()[t]->get_state()->get_nx();
      if (static_cast<std::size_t>(xs_warm[t].size()) != nx) {
        throw_pretty("Invalid argument: "
                     << "xs_init[" + std::to_string(t) + "] has wrong dimension (it should be " + std::to_string(nx) +
                            ")");
      }
    }
    const std::size_t& nx = problem_->get_terminalModel()->get_state()->get_nx();
    if (static_cast<std::size_t>(xs_warm[T].size()) != nx) {
      throw_pretty("Invalid argument: "
                   << "xs_init[" + std::to_string(T) + "] has wrong dimension (it should be " + std::to_string(nx) +
                          ")");
    }
    std::copy(xs_warm.begin(), xs_warm.end(), xs_.begin());
  }

  if (us_warm.size() == 0) {
    for (std::size_t t = 0; t < T; ++t) {
      us_[t] = Eigen::VectorXd::Zero(problem_->get_nu_max());
    }
  } else {
    if (us_warm.size() != T) {
      throw_pretty("Warm start control has wrong dimension, got " << us_warm.size() << " expecting " << T);
    }
    const std::size_t& nu = problem_->get_nu_max();
    for (std::size_t t = 0; t < T; ++t) {
      if (static_cast<std::size_t>(us_warm[t].size()) > nu) {
        throw_pretty("Invalid argument: "
                     << "us_init[" + std::to_string(t) + "] has wrong dimension (it should be lower than " +
                            std::to_string(nu) + ")");
      }
    }
    std::copy(us_warm.begin(), us_warm.end(), us_.begin());
  }
  is_feasible_ = is_feasible;
}

void SolverAbstract::setCallbacks(const std::vector<boost::shared_ptr<CallbackAbstract> >& callbacks) {
  callbacks_ = callbacks;
}

const std::vector<boost::shared_ptr<CallbackAbstract> >& SolverAbstract::getCallbacks() const { return callbacks_; }

const boost::shared_ptr<ShootingProblem>& SolverAbstract::get_problem() const { return problem_; }

const std::vector<Eigen::VectorXd>& SolverAbstract::get_xs() const { return xs_; }

const std::vector<Eigen::VectorXd>& SolverAbstract::get_us() const { return us_; }

const bool& SolverAbstract::get_is_feasible() const { return is_feasible_; }

const double& SolverAbstract::get_cost() const { return cost_; }

const double& SolverAbstract::get_stop() const { return stop_; }

const Eigen::Vector2d& SolverAbstract::get_d() const { return d_; }

const double& SolverAbstract::get_xreg() const { return xreg_; }

const double& SolverAbstract::get_ureg() const { return ureg_; }

const double& SolverAbstract::get_steplength() const { return steplength_; }

const double& SolverAbstract::get_dV() const { return dV_; }

const double& SolverAbstract::get_dVexp() const { return dVexp_; }

const double& SolverAbstract::get_th_acceptstep() const { return th_acceptstep_; }

const double& SolverAbstract::get_th_stop() const { return th_stop_; }

const std::size_t& SolverAbstract::get_iter() const { return iter_; }

void SolverAbstract::set_xs(const std::vector<Eigen::VectorXd>& xs) {
  const std::size_t& T = problem_->get_T();
  if (xs.size() != T + 1) {
    throw_pretty("Invalid argument: "
                 << "xs list has to be " + std::to_string(T + 1));
  }

  const std::size_t& nx = problem_->get_nx();
  for (std::size_t t = 0; t < T; ++t) {
    if (static_cast<std::size_t>(xs[t].size()) != nx) {
      throw_pretty("Invalid argument: "
                   << "xs[" + std::to_string(t) + "] has wrong dimension (it should be " + std::to_string(nx) + ")")
    }
  }
  if (static_cast<std::size_t>(xs[T].size()) != nx) {
    throw_pretty("Invalid argument: "
                 << "xs[" + std::to_string(T) + "] has wrong dimension (it should be " + std::to_string(nx) + ")")
  }
  xs_ = xs;
}

void SolverAbstract::set_us(const std::vector<Eigen::VectorXd>& us) {
  const std::size_t& T = problem_->get_T();
  if (us.size() != T) {
    throw_pretty("Invalid argument: "
                 << "us list has to be " + std::to_string(T));
  }

  const std::size_t& nu = problem_->get_nu_max();
  for (std::size_t t = 0; t < T; ++t) {
    if (static_cast<std::size_t>(us[t].size()) != nu) {
      throw_pretty("Invalid argument: "
                   << "us[" + std::to_string(t) + "] has wrong dimension (it should be " + std::to_string(nu) + ")")
    }
  }
  us_ = us;
}

void SolverAbstract::set_xreg(const double& xreg) {
  if (xreg < 0.) {
    throw_pretty("Invalid argument: "
                 << "xreg value has to be positive.");
  }
  xreg_ = xreg;
}

void SolverAbstract::set_ureg(const double& ureg) {
  if (ureg < 0.) {
    throw_pretty("Invalid argument: "
                 << "ureg value has to be positive.");
  }
  ureg_ = ureg;
}

void SolverAbstract::set_th_acceptstep(const double& th_acceptstep) {
  if (0. >= th_acceptstep || th_acceptstep > 1) {
    throw_pretty("Invalid argument: "
                 << "th_acceptstep value should between 0 and 1.");
  }
  th_acceptstep_ = th_acceptstep;
}

void SolverAbstract::set_th_stop(const double& th_stop) {
  if (th_stop <= 0.) {
    throw_pretty("Invalid argument: "
                 << "th_stop value has to higher than 0.");
  }
  th_stop_ = th_stop;
}

bool raiseIfNaN(const double& value) {
  if (std::isnan(value) || std::isinf(value) || value >= 1e30) {
    return true;
  } else {
    return false;
  }
}

}  // namespace crocoddyl
