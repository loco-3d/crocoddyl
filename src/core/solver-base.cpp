///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifdef CROCODDYL_WITH_MULTITHREADING
#include <omp.h>
#endif  // CROCODDYL_WITH_MULTITHREADING

#include "crocoddyl/core/solver-base.hpp"

namespace crocoddyl {

SolverAbstract::SolverAbstract(std::shared_ptr<ShootingProblem> problem)
    : problem_(problem),
      is_feasible_(false),
      was_feasible_(false),
      cost_(0.),
      merit_(0.),
      stop_(0.),
      dV_(0.),
      dPhi_(0.),
      dVexp_(0.),
      dPhiexp_(0.),
      dfeas_(0.),
      feas_(0.),
      ffeas_(0.),
      gfeas_(0.),
      hfeas_(0.),
      ffeas_try_(0.),
      gfeas_try_(0.),
      hfeas_try_(0.),
      preg_(0.),
      dreg_(0.),
      steplength_(1.),
      th_acceptstep_(0.1),
      th_stop_(1e-9),
      th_gaptol_(1e-16),
      feasnorm_(LInf),
      iter_(0),
      tmp_feas_(0.) {
  // Allocate common data
  const std::size_t ndx = problem_->get_ndx();
  const std::size_t T = problem_->get_T();
  const std::size_t ng_T = problem_->get_terminalModel()->get_ng_T();
  xs_.resize(T + 1);
  us_.resize(T);
  fs_.resize(T + 1);
  g_adj_.resize(T + 1);
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    const std::size_t ng = model->get_ng();
    xs_[t] = model->get_state()->zero();
    us_[t] = Eigen::VectorXd::Zero(nu);
    fs_[t] = Eigen::VectorXd::Zero(ndx);
    g_adj_[t] = Eigen::VectorXd::Zero(ng);
  }
  xs_.back() = problem_->get_terminalModel()->get_state()->zero();
  fs_.back() = Eigen::VectorXd::Zero(ndx);
  g_adj_.back() = Eigen::VectorXd::Zero(ng_T);
}

SolverAbstract::~SolverAbstract() {}

void SolverAbstract::resizeData() {
  START_PROFILER("SolverAbstract::resizeData");
  const std::size_t T = problem_->get_T();
  const std::size_t ng_T = problem_->get_terminalModel()->get_ng_T();
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    const std::size_t ng = model->get_ng();
    us_[t].conservativeResize(nu);
    g_adj_[t].conservativeResize(ng);
  }

  g_adj_.back().conservativeResize(ng_T);

  STOP_PROFILER("SolverAbstract::resizeData");
}

double SolverAbstract::computeDynamicFeasibility() {
  tmp_feas_ = 0.;
  const std::size_t T = problem_->get_T();
  const Eigen::VectorXd& x0 = problem_->get_x0();
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
      problem_->get_runningDatas();

  models[0]->get_state()->diff(xs_[0], x0, fs_[0]);
#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(problem_->get_nthreads())
#endif
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& m = models[t];
    const std::shared_ptr<ActionDataAbstract>& d = datas[t];
    m->get_state()->diff(xs_[t + 1], d->xnext, fs_[t + 1]);
  }
  switch (feasnorm_) {
    case LInf:
      tmp_feas_ = std::max(tmp_feas_, fs_[0].lpNorm<Eigen::Infinity>());
      for (std::size_t t = 0; t < T; ++t) {
        tmp_feas_ = std::max(tmp_feas_, fs_[t + 1].lpNorm<Eigen::Infinity>());
      }
      break;
    case L1:
      tmp_feas_ = fs_[0].lpNorm<1>();
      for (std::size_t t = 0; t < T; ++t) {
        tmp_feas_ += fs_[t + 1].lpNorm<1>();
      }
      break;
  }
  return tmp_feas_;
}

double SolverAbstract::computeInequalityFeasibility() {
  tmp_feas_ = 0.;
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
      problem_->get_runningDatas();

  switch (feasnorm_) {
    case LInf:
      for (std::size_t t = 0; t < T; ++t) {
        if (models[t]->get_ng() > 0) {
          g_adj_[t] = datas[t]
                          ->g.cwiseMax(models[t]->get_g_lb())
                          .cwiseMin(models[t]->get_g_ub());
          tmp_feas_ = std::max(
              tmp_feas_, (datas[t]->g - g_adj_[t]).lpNorm<Eigen::Infinity>());
        }
      }
      if (problem_->get_terminalModel()->get_ng_T() > 0) {
        g_adj_.back() =
            problem_->get_terminalData()
                ->g.cwiseMax(problem_->get_terminalModel()->get_g_lb())
                .cwiseMin(problem_->get_terminalModel()->get_g_ub());
        tmp_feas_ += (problem_->get_terminalData()->g - g_adj_.back())
                         .lpNorm<Eigen::Infinity>();
      }
      break;
    case L1:
      for (std::size_t t = 0; t < T; ++t) {
        if (models[t]->get_ng() > 0) {
          g_adj_[t] = datas[t]
                          ->g.cwiseMax(models[t]->get_g_lb())
                          .cwiseMin(models[t]->get_g_ub());
          tmp_feas_ =
              std::max(tmp_feas_, (datas[t]->g - g_adj_[t]).lpNorm<1>());
        }
      }
      if (problem_->get_terminalModel()->get_ng_T() > 0) {
        g_adj_.back() =
            problem_->get_terminalData()
                ->g.cwiseMax(problem_->get_terminalModel()->get_g_lb())
                .cwiseMin(problem_->get_terminalModel()->get_g_ub());
        tmp_feas_ +=
            (problem_->get_terminalData()->g - g_adj_.back()).lpNorm<1>();
      }
      break;
  }
  return tmp_feas_;
}

double SolverAbstract::computeEqualityFeasibility() {
  tmp_feas_ = 0.;
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
      problem_->get_runningDatas();
  switch (feasnorm_) {
    case LInf:
      for (std::size_t t = 0; t < T; ++t) {
        if (models[t]->get_nh() > 0) {
          tmp_feas_ =
              std::max(tmp_feas_, datas[t]->h.lpNorm<Eigen::Infinity>());
        }
      }
      if (problem_->get_terminalModel()->get_nh_T() > 0) {
        tmp_feas_ =
            std::max(tmp_feas_,
                     problem_->get_terminalData()->h.lpNorm<Eigen::Infinity>());
      }
      break;
    case L1:
      for (std::size_t t = 0; t < T; ++t) {
        if (models[t]->get_nh() > 0) {
          tmp_feas_ += datas[t]->h.lpNorm<1>();
        }
      }
      if (problem_->get_terminalModel()->get_nh_T() > 0) {
        tmp_feas_ += problem_->get_terminalData()->h.lpNorm<1>();
      }
      break;
  }
  return tmp_feas_;
}

void SolverAbstract::setCandidate(const std::vector<Eigen::VectorXd>& xs_warm,
                                  const std::vector<Eigen::VectorXd>& us_warm,
                                  bool is_feasible) {
  const std::size_t T = problem_->get_T();

  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  if (xs_warm.size() == 0) {
    for (std::size_t t = 0; t < T; ++t) {
      const std::shared_ptr<ActionModelAbstract>& model = models[t];
      xs_[t] = model->get_state()->zero();
    }
    xs_.back() = problem_->get_terminalModel()->get_state()->zero();
  } else {
    if (xs_warm.size() != T + 1) {
      throw_pretty("Warm start state vector has wrong dimension, got "
                   << xs_warm.size() << " expecting " << (T + 1));
    }
    for (std::size_t t = 0; t < T; ++t) {
      const std::size_t nx = models[t]->get_state()->get_nx();
      if (static_cast<std::size_t>(xs_warm[t].size()) != nx) {
        throw_pretty("Invalid argument: "
                     << "xs_init[" + std::to_string(t) +
                            "] has wrong dimension ("
                     << xs_warm[t].size()
                     << " provided - it should be equal to " +
                            std::to_string(nx) + "). ActionModel: "
                     << *models[t]);
      }
    }
    const std::size_t nx = problem_->get_terminalModel()->get_state()->get_nx();
    if (static_cast<std::size_t>(xs_warm[T].size()) != nx) {
      throw_pretty("Invalid argument: "
                   << "xs_init[" + std::to_string(T) +
                          "] (terminal state) has wrong dimension ("
                   << xs_warm[T].size()
                   << " provided - it should be equal to " +
                          std::to_string(nx) + "). ActionModel: "
                   << *problem_->get_terminalModel());
    }
    std::copy(xs_warm.begin(), xs_warm.end(), xs_.begin());
  }

  if (us_warm.size() == 0) {
    for (std::size_t t = 0; t < T; ++t) {
      const std::shared_ptr<ActionModelAbstract>& model = models[t];
      const std::size_t nu = model->get_nu();
      us_[t] = Eigen::VectorXd::Zero(nu);
    }
  } else {
    if (us_warm.size() != T) {
      throw_pretty("Warm start control has wrong dimension, got "
                   << us_warm.size() << " expecting " << T);
    }
    for (std::size_t t = 0; t < T; ++t) {
      const std::shared_ptr<ActionModelAbstract>& model = models[t];
      const std::size_t nu = model->get_nu();
      if (static_cast<std::size_t>(us_warm[t].size()) != nu) {
        throw_pretty("Invalid argument: "
                     << "us_init[" + std::to_string(t) +
                            "] has wrong dimension ("
                     << us_warm[t].size()
                     << " provided - it should be equal to " +
                            std::to_string(nu) + "). ActionModel: "
                     << *model);
      }
    }
    std::copy(us_warm.begin(), us_warm.end(), us_.begin());
  }
  is_feasible_ = is_feasible;
}

void SolverAbstract::setCallbacks(
    const std::vector<std::shared_ptr<CallbackAbstract> >& callbacks) {
  callbacks_ = callbacks;
}

const std::vector<std::shared_ptr<CallbackAbstract> >&
SolverAbstract::getCallbacks() const {
  return callbacks_;
}

const std::shared_ptr<ShootingProblem>& SolverAbstract::get_problem() const {
  return problem_;
}

const std::vector<Eigen::VectorXd>& SolverAbstract::get_xs() const {
  return xs_;
}

const std::vector<Eigen::VectorXd>& SolverAbstract::get_us() const {
  return us_;
}

const std::vector<Eigen::VectorXd>& SolverAbstract::get_fs() const {
  return fs_;
}

bool SolverAbstract::get_is_feasible() const { return is_feasible_; }

double SolverAbstract::get_cost() const { return cost_; }

double SolverAbstract::get_merit() const { return merit_; }

double SolverAbstract::get_stop() const { return stop_; }

const Eigen::Vector2d& SolverAbstract::get_d() const { return d_; }

double SolverAbstract::get_dV() const { return dV_; }

double SolverAbstract::get_dPhi() const { return dPhi_; }

double SolverAbstract::get_dVexp() const { return dVexp_; }

double SolverAbstract::get_dPhiexp() const { return dPhiexp_; }

double SolverAbstract::get_dfeas() const { return dfeas_; }

double SolverAbstract::get_feas() const { return feas_; }

double SolverAbstract::get_ffeas() const { return ffeas_; }

double SolverAbstract::get_gfeas() const { return gfeas_; }

double SolverAbstract::get_hfeas() const { return hfeas_; }

double SolverAbstract::get_ffeas_try() const { return ffeas_try_; }

double SolverAbstract::get_gfeas_try() const { return gfeas_try_; }

double SolverAbstract::get_hfeas_try() const { return hfeas_try_; }

double SolverAbstract::get_preg() const { return preg_; }

double SolverAbstract::get_dreg() const { return dreg_; }

DEPRECATED(
    "Use get_preg for gettting the primal-dual regularization",
    double SolverAbstract::get_xreg() const { return preg_; })

DEPRECATED(
    "Use get_preg for gettting the primal-dual regularization",
    double SolverAbstract::get_ureg() const { return preg_; })

double SolverAbstract::get_steplength() const { return steplength_; }

double SolverAbstract::get_th_acceptstep() const { return th_acceptstep_; }

double SolverAbstract::get_th_stop() const { return th_stop_; }

double SolverAbstract::get_th_gaptol() const { return th_gaptol_; }

FeasibilityNorm SolverAbstract::get_feasnorm() const { return feasnorm_; }

std::size_t SolverAbstract::get_iter() const { return iter_; }

void SolverAbstract::set_xs(const std::vector<Eigen::VectorXd>& xs) {
  const std::size_t T = problem_->get_T();
  if (xs.size() != T + 1) {
    throw_pretty("Invalid argument: " << "xs list has to be of length " +
                                             std::to_string(T + 1));
  }

  const std::size_t nx = problem_->get_nx();
  for (std::size_t t = 0; t < T; ++t) {
    if (static_cast<std::size_t>(xs[t].size()) != nx) {
      throw_pretty("Invalid argument: "
                   << "xs[" + std::to_string(t) + "] has wrong dimension ("
                   << xs[t].size()
                   << " provided - it should be " + std::to_string(nx) + ")")
    }
  }
  if (static_cast<std::size_t>(xs[T].size()) != nx) {
    throw_pretty("Invalid argument: "
                 << "xs[" + std::to_string(T) +
                        "] (terminal state) has wrong dimension ("
                 << xs[T].size()
                 << " provided - it should be " + std::to_string(nx) + ")")
  }
  xs_ = xs;
}

void SolverAbstract::set_us(const std::vector<Eigen::VectorXd>& us) {
  const std::size_t T = problem_->get_T();
  if (us.size() != T) {
    throw_pretty("Invalid argument: " << "us list has to be of length " +
                                             std::to_string(T));
  }

  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    if (static_cast<std::size_t>(us[t].size()) != nu) {
      throw_pretty("Invalid argument: "
                   << "us[" + std::to_string(t) + "] has wrong dimension ("
                   << us[t].size()
                   << " provided - it should be " + std::to_string(nu) + ")")
    }
  }
  us_ = us;
}

void SolverAbstract::set_preg(const double preg) {
  if (preg < 0.) {
    throw_pretty("Invalid argument: " << "preg value has to be positive.");
  }
  preg_ = preg;
}

void SolverAbstract::set_dreg(const double dreg) {
  if (dreg < 0.) {
    throw_pretty("Invalid argument: " << "dreg value has to be positive.");
  }
  dreg_ = dreg;
}

DEPRECATED(
    "Use set_preg for gettting the primal-variable regularization",
    void SolverAbstract::set_xreg(const double xreg) {
      if (xreg < 0.) {
        throw_pretty("Invalid argument: " << "xreg value has to be positive.");
      }
      xreg_ = xreg;
      preg_ = xreg;
    })

DEPRECATED(
    "Use set_preg for gettting the primal-variable regularization",
    void SolverAbstract::set_ureg(const double ureg) {
      if (ureg < 0.) {
        throw_pretty("Invalid argument: " << "ureg value has to be positive.");
      }
      ureg_ = ureg;
      preg_ = ureg;
    })

void SolverAbstract::set_th_acceptstep(const double th_acceptstep) {
  if (0. >= th_acceptstep || th_acceptstep > 1) {
    throw_pretty(
        "Invalid argument: " << "th_acceptstep value should between 0 and 1.");
  }
  th_acceptstep_ = th_acceptstep;
}

void SolverAbstract::set_th_stop(const double th_stop) {
  if (th_stop <= 0.) {
    throw_pretty("Invalid argument: " << "th_stop value has to higher than 0.");
  }
  th_stop_ = th_stop;
}

void SolverAbstract::set_th_gaptol(const double th_gaptol) {
  if (0. > th_gaptol) {
    throw_pretty("Invalid argument: " << "th_gaptol value has to be positive.");
  }
  th_gaptol_ = th_gaptol;
}

void SolverAbstract::set_feasnorm(const FeasibilityNorm feasnorm) {
  feasnorm_ = feasnorm;
}

bool raiseIfNaN(const double value) {
  if (std::isnan(value) || std::isinf(value) || value >= 1e30) {
    return true;
  } else {
    return false;
  }
}

}  // namespace crocoddyl
