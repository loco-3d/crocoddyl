///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/solvers/ddp.hpp"

namespace crocoddyl {

SolverDDP::SolverDDP(boost::shared_ptr<ShootingProblem> problem)
    : SolverAbstract(problem),
      regfactor_(10.),
      regmin_(1e-9),
      regmax_(1e9),
      cost_try_(0.),
      th_grad_(1e-12),
      th_stepdec_(0.5),
      th_stepinc_(0.01),
      was_feasible_(false) {
  allocateData();

  const std::size_t& n_alphas = 10;
  alphas_.resize(n_alphas);
  for (std::size_t n = 0; n < n_alphas; ++n) {
    alphas_[n] = 1. / pow(2., static_cast<double>(n));
  }
  if (th_stepinc_ < alphas_[n_alphas - 1]) {
    th_stepinc_ = alphas_[n_alphas - 1];
    std::cerr << "Warning: th_stepinc has higher value than lowest alpha value, set to "
              << std::to_string(alphas_[n_alphas - 1]) << std::endl;
  }
}

SolverDDP::~SolverDDP() {}

bool SolverDDP::solve(const std::vector<Eigen::VectorXd>& init_xs, const std::vector<Eigen::VectorXd>& init_us,
                      const std::size_t& maxiter, const bool& is_feasible, const double& reginit) {
  xs_try_[0] = problem_->get_x0();  // it is needed in case that init_xs[0] is infeasible
  setCandidate(init_xs, init_us, is_feasible);

  if (std::isnan(reginit)) {
    xreg_ = regmin_;
    ureg_ = regmin_;
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
        if (xreg_ == regmax_) {
          return false;
        } else {
          continue;
        }
      }
      break;
    }
    expectedImprovement();

    // We need to recalculate the derivatives when the step length passes
    recalcDiff = false;
    for (std::vector<double>::const_iterator it = alphas_.begin(); it != alphas_.end(); ++it) {
      steplength_ = *it;

      try {
        dV_ = tryStep(steplength_);
      } catch (std::exception& e) {
        continue;
      }
      dVexp_ = steplength_ * (d_[0] + 0.5 * steplength_ * d_[1]);

      if (dVexp_ >= 0) {  // descend direction
        if (d_[0] < th_grad_ || !is_feasible_ || dV_ > th_acceptstep_ * dVexp_) {
          was_feasible_ = is_feasible_;
          setCandidate(xs_try_, us_try_, true);
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
      if (xreg_ == regmax_) {
        return false;
      }
    }
    stoppingCriteria();

    const std::size_t& n_callbacks = callbacks_.size();
    for (std::size_t c = 0; c < n_callbacks; ++c) {
      CallbackAbstract& callback = *callbacks_[c];
      callback(*this);
    }

    if (was_feasible_ && stop_ < th_stop_) {
      return true;
    }
  }
  return false;
}

void SolverDDP::computeDirection(const bool& recalcDiff) {
  if (recalcDiff) {
    calcDiff();
  }
  backwardPass();
}

double SolverDDP::tryStep(const double& steplength) {
  forwardPass(steplength);
  return cost_ - cost_try_;
}

double SolverDDP::stoppingCriteria() {
  stop_ = 0.;
  const std::size_t& T = this->problem_->get_T();
  for (std::size_t t = 0; t < T; ++t) {
    stop_ += Qu_[t].squaredNorm();
  }
  return stop_;
}

const Eigen::Vector2d& SolverDDP::expectedImprovement() {
  d_.fill(0);
  const std::size_t& T = this->problem_->get_T();
  for (std::size_t t = 0; t < T; ++t) {
    d_[0] += Qu_[t].dot(k_[t]);
    d_[1] -= k_[t].dot(Quuk_[t]);
  }
  return d_;
}

double SolverDDP::calcDiff() {
  if (iter_ == 0) problem_->calc(xs_, us_);
  cost_ = problem_->calcDiff(xs_, us_);
  if (!is_feasible_) {
    const Eigen::VectorXd& x0 = problem_->get_x0();
    problem_->get_runningModels()[0]->get_state()->diff(xs_[0], x0, fs_[0]);

    const std::size_t& T = problem_->get_T();
    const std::vector<boost::shared_ptr<ActionModelAbstract> >& models = problem_->get_runningModels();
    const std::vector<boost::shared_ptr<ActionDataAbstract> >& datas = problem_->get_runningDatas();
    for (std::size_t t = 0; t < T; ++t) {
      const boost::shared_ptr<ActionModelAbstract>& model = models[t];
      const boost::shared_ptr<ActionDataAbstract>& d = datas[t];
      model->get_state()->diff(xs_[t + 1], d->xnext, fs_[t + 1]);
    }
  } else if (!was_feasible_) {  // closing the gaps
    for (std::vector<Eigen::VectorXd>::iterator it = fs_.begin(); it != fs_.end(); ++it) {
      it->setZero();
    }
  }
  return cost_;
}

void SolverDDP::backwardPass() {
  const boost::shared_ptr<ActionDataAbstract>& d_T = problem_->get_terminalData();
  Vxx_.back() = d_T->Lxx;
  Vx_.back() = d_T->Lx;

  if (!std::isnan(xreg_)) {
    Vxx_.back().diagonal().array() += xreg_;
  }

  if (!is_feasible_) {
    Vx_.back().noalias() += Vxx_.back() * fs_.back();
  }

  const std::vector<boost::shared_ptr<ActionDataAbstract> >& datas = problem_->get_runningDatas();
  for (int t = static_cast<int>(problem_->get_T()) - 1; t >= 0; --t) {
    const boost::shared_ptr<ActionDataAbstract>& d = datas[t];
    const Eigen::MatrixXd& Vxx_p = Vxx_[t + 1];
    const Eigen::VectorXd& Vx_p = Vx_[t + 1];

    Qxx_[t] = d->Lxx;
    Qxu_[t] = d->Lxu;
    Quu_[t] = d->Luu;
    Qx_[t] = d->Lx;
    Qu_[t] = d->Lu;
    FxTVxx_p_.noalias() = d->Fx.transpose() * Vxx_p;
    FuTVxx_p_[t].noalias() = d->Fu.transpose() * Vxx_p;
    Qxx_[t].noalias() += FxTVxx_p_ * d->Fx;
    Qxu_[t].noalias() += FxTVxx_p_ * d->Fu;
    Quu_[t].noalias() += FuTVxx_p_[t] * d->Fu;
    Qx_[t].noalias() += d->Fx.transpose() * Vx_p;
    Qu_[t].noalias() += d->Fu.transpose() * Vx_p;

    if (!std::isnan(ureg_)) {
      Quu_[t].diagonal().array() += ureg_;
    }

    computeGains(t);

    Vx_[t] = Qx_[t];
    if (std::isnan(ureg_)) {
      Vx_[t].noalias() -= K_[t].transpose() * Qu_[t];
    } else {
      Quuk_[t].noalias() = Quu_[t] * k_[t];
      Vx_[t].noalias() += K_[t].transpose() * Quuk_[t];
      Vx_[t].noalias() -= 2 * (K_[t].transpose() * Qu_[t]);
    }
    Vxx_[t] = Qxx_[t];
    Vxx_[t].noalias() -= Qxu_[t] * K_[t];
    Vxx_[t] = 0.5 * (Vxx_[t] + Vxx_[t].transpose()).eval();

    if (!std::isnan(xreg_)) {
      Vxx_[t].diagonal().array() += xreg_;
    }

    // Compute and store the Vx gradient at end of the interval (rollout state)
    if (!is_feasible_) {
      Vx_[t].noalias() += Vxx_[t] * fs_[t];
    }

    if (raiseIfNaN(Vx_[t].lpNorm<Eigen::Infinity>())) {
      throw_pretty("backward_error");
    }
    if (raiseIfNaN(Vxx_[t].lpNorm<Eigen::Infinity>())) {
      throw_pretty("backward_error");
    }
  }
}

void SolverDDP::forwardPass(const double& steplength) {
  if (steplength > 1. || steplength < 0.) {
    throw_pretty("Invalid argument: "
                 << "invalid step length, value is between 0. to 1.");
  }
  cost_try_ = 0.;
  const std::size_t& T = problem_->get_T();
  const std::vector<boost::shared_ptr<ActionModelAbstract> >& models = problem_->get_runningModels();
  const std::vector<boost::shared_ptr<ActionDataAbstract> >& datas = problem_->get_runningDatas();
  for (std::size_t t = 0; t < T; ++t) {
    const boost::shared_ptr<ActionModelAbstract>& m = models[t];
    const boost::shared_ptr<ActionDataAbstract>& d = datas[t];

    m->get_state()->diff(xs_[t], xs_try_[t], dx_[t]);
    us_try_[t].noalias() = us_[t];
    us_try_[t].noalias() -= k_[t] * steplength;
    us_try_[t].noalias() -= K_[t] * dx_[t];
    m->calc(d, xs_try_[t], us_try_[t]);
    xs_try_[t + 1] = d->xnext;
    cost_try_ += d->cost;

    if (raiseIfNaN(cost_try_)) {
      throw_pretty("forward_error");
    }
    if (raiseIfNaN(xs_try_[t + 1].lpNorm<Eigen::Infinity>())) {
      throw_pretty("forward_error");
    }
  }

  const boost::shared_ptr<ActionModelAbstract>& m = problem_->get_terminalModel();
  const boost::shared_ptr<ActionDataAbstract>& d = problem_->get_terminalData();
  m->calc(d, xs_try_.back());
  cost_try_ += d->cost;

  if (raiseIfNaN(cost_try_)) {
    throw_pretty("forward_error");
  }
}

void SolverDDP::computeGains(const std::size_t& t) {
  if (problem_->get_runningModels()[t]->get_nu() > 0) {
    Quu_llt_[t].compute(Quu_[t]);
    const Eigen::ComputationInfo& info = Quu_llt_[t].info();
    if (info != Eigen::Success) {
      throw_pretty("backward_error");
    }
    K_[t] = Qxu_[t].transpose();
    Quu_llt_[t].solveInPlace(K_[t]);
    k_[t] = Qu_[t];
    Quu_llt_[t].solveInPlace(k_[t]);
  }
}

void SolverDDP::increaseRegularization() {
  xreg_ *= regfactor_;
  if (xreg_ > regmax_) {
    xreg_ = regmax_;
  }
  ureg_ = xreg_;
}

void SolverDDP::decreaseRegularization() {
  xreg_ /= regfactor_;
  if (xreg_ < regmin_) {
    xreg_ = regmin_;
  }
  ureg_ = xreg_;
}

void SolverDDP::allocateData() {
  const std::size_t& T = problem_->get_T();
  Vxx_.resize(T + 1);
  Vx_.resize(T + 1);
  Qxx_.resize(T);
  Qxu_.resize(T);
  Quu_.resize(T);
  Qx_.resize(T);
  Qu_.resize(T);
  K_.resize(T);
  k_.resize(T);
  fs_.resize(T + 1);

  xs_try_.resize(T + 1);
  us_try_.resize(T);
  dx_.resize(T);

  FuTVxx_p_.resize(T);
  Quu_llt_.resize(T);
  Quuk_.resize(T);

  for (std::size_t t = 0; t < T; ++t) {
    const boost::shared_ptr<ActionModelAbstract>& model = problem_->get_runningModels()[t];
    const std::size_t& nx = model->get_state()->get_nx();
    const std::size_t& ndx = model->get_state()->get_ndx();
    const std::size_t& nu = model->get_nu();

    Vxx_[t] = Eigen::MatrixXd::Zero(ndx, ndx);
    Vx_[t] = Eigen::VectorXd::Zero(ndx);
    Qxx_[t] = Eigen::MatrixXd::Zero(ndx, ndx);
    Qxu_[t] = Eigen::MatrixXd::Zero(ndx, nu);
    Quu_[t] = Eigen::MatrixXd::Zero(nu, nu);
    Qx_[t] = Eigen::VectorXd::Zero(ndx);
    Qu_[t] = Eigen::VectorXd::Zero(nu);
    K_[t] = Eigen::MatrixXd::Zero(nu, ndx);
    k_[t] = Eigen::VectorXd::Zero(nu);
    fs_[t] = Eigen::VectorXd::Zero(ndx);

    if (t == 0) {
      xs_try_[t] = problem_->get_x0();
    } else {
      xs_try_[t] = Eigen::VectorXd::Constant(nx, NAN);
    }
    us_try_[t] = Eigen::VectorXd::Constant(nu, NAN);
    dx_[t] = Eigen::VectorXd::Zero(ndx);

    FuTVxx_p_[t] = Eigen::MatrixXd::Zero(nu, ndx);
    Quu_llt_[t] = Eigen::LLT<Eigen::MatrixXd>(nu);
    Quuk_[t] = Eigen::VectorXd(nu);
  }
  const std::size_t& ndx = problem_->get_terminalModel()->get_state()->get_ndx();
  Vxx_.back() = Eigen::MatrixXd::Zero(ndx, ndx);
  Vx_.back() = Eigen::VectorXd::Zero(ndx);
  xs_try_.back() = problem_->get_terminalModel()->get_state()->zero();
  fs_.back() = Eigen::VectorXd::Zero(ndx);

  FxTVxx_p_ = Eigen::MatrixXd::Zero(ndx, ndx);
  fTVxx_p_ = Eigen::VectorXd::Zero(ndx);
}

const double& SolverDDP::get_regfactor() const { return regfactor_; }

const double& SolverDDP::get_regmin() const { return regmin_; }

const double& SolverDDP::get_regmax() const { return regmax_; }

const std::vector<double>& SolverDDP::get_alphas() const { return alphas_; }

const double& SolverDDP::get_th_stepdec() const { return th_stepdec_; }

const double& SolverDDP::get_th_stepinc() const { return th_stepinc_; }

const double& SolverDDP::get_th_grad() const { return th_grad_; }

const std::vector<Eigen::MatrixXd>& SolverDDP::get_Vxx() const { return Vxx_; }

const std::vector<Eigen::VectorXd>& SolverDDP::get_Vx() const { return Vx_; }

const std::vector<Eigen::MatrixXd>& SolverDDP::get_Qxx() const { return Qxx_; }

const std::vector<Eigen::MatrixXd>& SolverDDP::get_Qxu() const { return Qxu_; }

const std::vector<Eigen::MatrixXd>& SolverDDP::get_Quu() const { return Quu_; }

const std::vector<Eigen::VectorXd>& SolverDDP::get_Qx() const { return Qx_; }

const std::vector<Eigen::VectorXd>& SolverDDP::get_Qu() const { return Qu_; }

const std::vector<Eigen::MatrixXd>& SolverDDP::get_K() const { return K_; }

const std::vector<Eigen::VectorXd>& SolverDDP::get_k() const { return k_; }

const std::vector<Eigen::VectorXd>& SolverDDP::get_fs() const { return fs_; }

void SolverDDP::set_regfactor(const double& regfactor) {
  if (regfactor <= 1.) {
    throw_pretty("Invalid argument: "
                 << "regfactor value is higher than 1.");
  }
  regfactor_ = regfactor;
}

void SolverDDP::set_regmin(const double& regmin) {
  if (0. > regmin) {
    throw_pretty("Invalid argument: "
                 << "regmin value has to be positive.");
  }
  regmin_ = regmin;
}

void SolverDDP::set_regmax(const double& regmax) {
  if (0. > regmax) {
    throw_pretty("Invalid argument: "
                 << "regmax value has to be positive.");
  }
  regmax_ = regmax;
}

void SolverDDP::set_alphas(const std::vector<double>& alphas) {
  double prev_alpha = alphas[0];
  if (prev_alpha != 1.) {
    std::cerr << "Warning: alpha[0] should be 1" << std::endl;
  }
  for (std::size_t i = 1; i < alphas.size(); ++i) {
    double alpha = alphas[i];
    if (0. >= alpha) {
      throw_pretty("Invalid argument: "
                   << "alpha values has to be positive.");
    }
    if (alpha >= prev_alpha) {
      throw_pretty("Invalid argument: "
                   << "alpha values are monotonously decreasing.");
    }
    prev_alpha = alpha;
  }
  alphas_ = alphas;
}

void SolverDDP::set_th_stepdec(const double& th_stepdec) {
  if (0. >= th_stepdec || th_stepdec > 1.) {
    throw_pretty("Invalid argument: "
                 << "th_stepdec value should between 0 and 1.");
  }
  th_stepdec_ = th_stepdec;
}

void SolverDDP::set_th_stepinc(const double& th_stepinc) {
  if (0. >= th_stepinc || th_stepinc > 1.) {
    throw_pretty("Invalid argument: "
                 << "th_stepinc value should between 0 and 1.");
  }
  th_stepinc_ = th_stepinc;
}

void SolverDDP::set_th_grad(const double& th_grad) {
  if (0. > th_grad) {
    throw_pretty("Invalid argument: "
                 << "th_grad value has to be positive.");
  }
  th_grad_ = th_grad;
}

}  // namespace crocoddyl
