///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/solvers/ddp.hpp"

namespace crocoddyl {

SolverDDP::SolverDDP(boost::shared_ptr<ShootingProblem> problem)
    : SolverAbstract(problem),
      regfactor_(10.),
      regmin_(1e-9),
      regmax_(1e9),
      cost_try_(0.),
      th_grad_(1e-12),
      th_step_(0.5),
      was_feasible_(false) {
  allocateData();

  const std::size_t& n_alphas = 10;
  alphas_.resize(n_alphas);
  for (std::size_t n = 0; n < n_alphas; ++n) {
    alphas_[n] = 1. / pow(2., static_cast<double>(n));
  }
}

SolverDDP::~SolverDDP() {}

bool SolverDDP::solve(const std::vector<Eigen::VectorXd>& init_xs, const std::vector<Eigen::VectorXd>& init_us,
                      const std::size_t& maxiter, const bool& is_feasible, const double& reginit) {
  setCandidate(init_xs, init_us, is_feasible);

  if (std::isnan(reginit)) {
    xreg_ = regmin_;
    ureg_ = regmin_;
  } else {
    xreg_ = reginit;
    ureg_ = reginit;
  }
  was_feasible_ = false;

  bool recalc = true;
  for (iter_ = 0; iter_ < maxiter; ++iter_) {
    while (true) {
      try {
        computeDirection(recalc);
      } catch (CrocoddylException& e) {
        recalc = false;
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
    recalc = false;
    for (std::vector<double>::const_iterator it = alphas_.begin(); it != alphas_.end(); ++it) {
      steplength_ = *it;

      try {
        dV_ = tryStep(steplength_);
      } catch (CrocoddylException& e) {
        continue;
      }
      dVexp_ = steplength_ * (d_[0] + 0.5 * steplength_ * d_[1]);

      if (d_[0] < th_grad_ || !is_feasible_ || dV_ > th_acceptstep_ * dVexp_) {
        was_feasible_ = is_feasible_;
        setCandidate(xs_try_, us_try_, true);
        cost_ = cost_try_;
        recalc = true;
        break;
      }
    }

    if (steplength_ > th_step_) {
      decreaseRegularization();
    }
    if (steplength_ == alphas_.back()) {
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

void SolverDDP::computeDirection(const bool& recalc) {
  if (recalc) {
    calc();
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

double SolverDDP::calc() {
  cost_ = problem_->calcDiff(xs_, us_);
  if (!is_feasible_) {
    const Eigen::VectorXd& x0 = problem_->get_x0();
    problem_->get_runningModels()[0]->get_state()->diff(xs_[0], x0, gaps_[0]);

    const std::size_t& T = problem_->get_T();
    for (std::size_t t = 0; t < T; ++t) {
      const boost::shared_ptr<ActionModelAbstract>& model = problem_->get_runningModels()[t];
      const boost::shared_ptr<ActionDataAbstract>& d = problem_->get_runningDatas()[t];
      model->get_state()->diff(xs_[t + 1], d->xnext, gaps_[t + 1]);
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
    Vx_.back().noalias() += Vxx_.back() * gaps_.back();
  }

  for (int t = static_cast<int>(problem_->get_T()) - 1; t >= 0; --t) {
    const boost::shared_ptr<ActionDataAbstract>& d = problem_->get_runningDatas()[t];
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
      Vx_[t].noalias() += Vxx_[t] * gaps_[t];
    }

    if (raiseIfNaN(Vx_[t].lpNorm<Eigen::Infinity>())) {
      throw CrocoddylException("backward_error");
    }
    if (raiseIfNaN(Vxx_[t].lpNorm<Eigen::Infinity>())) {
      throw CrocoddylException("backward_error");
    }
  }
}

void SolverDDP::forwardPass(const double& steplength) {
  if (steplength > 1. || steplength < 0.) {
    throw CrocoddylException("invalid step length, value is between 0. to 1.");
  }
  cost_try_ = 0.;
  const std::size_t& T = problem_->get_T();
  for (std::size_t t = 0; t < T; ++t) {
    const boost::shared_ptr<ActionModelAbstract>& m = problem_->get_runningModels()[t];
    const boost::shared_ptr<ActionDataAbstract>& d = problem_->get_runningDatas()[t];

    m->get_state()->diff(xs_[t], xs_try_[t], dx_[t]);
    us_try_[t].noalias() = us_[t] - k_[t] * steplength - K_[t] * dx_[t];
    m->calc(d, xs_try_[t], us_try_[t]);
    xs_try_[t + 1] = d->xnext;
    cost_try_ += d->cost;

    if (raiseIfNaN(cost_try_)) {
      throw CrocoddylException("forward_error");
    }
    if (raiseIfNaN(xs_try_[t + 1].lpNorm<Eigen::Infinity>())) {
      throw CrocoddylException("forward_error");
    }
  }

  const boost::shared_ptr<ActionModelAbstract>& m = problem_->get_terminalModel();
  const boost::shared_ptr<ActionDataAbstract>& d = problem_->get_terminalData();
  m->calc(d, xs_try_.back());
  cost_try_ += d->cost;

  if (raiseIfNaN(cost_try_)) {
    throw CrocoddylException("forward_error");
  }
}

void SolverDDP::computeGains(const std::size_t& t) {
  if (problem_->get_runningModels()[t]->get_nu() > 0) {
    Quu_llt_[t].compute(Quu_[t]);
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
  gaps_.resize(T + 1);

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
    gaps_[t] = Eigen::VectorXd::Zero(ndx);

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
  gaps_.back() = Eigen::VectorXd::Zero(ndx);

  FxTVxx_p_ = Eigen::MatrixXd::Zero(ndx, ndx);
  fTVxx_p_ = Eigen::VectorXd::Zero(ndx);
}

const double& SolverDDP::get_regfactor() const { return regfactor_; }

const double& SolverDDP::get_regmin() const { return regmin_; }

const double& SolverDDP::get_regmax() const { return regmax_; }

const std::vector<double>& SolverDDP::get_alphas() const { return alphas_; }

const double& SolverDDP::get_th_step() const { return th_step_; }

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

const std::vector<Eigen::VectorXd>& SolverDDP::get_gaps() const { return gaps_; }

void SolverDDP::set_regfactor(double regfactor) { regfactor_ = regfactor; }

void SolverDDP::set_regmin(double regmin) { regmin_ = regmin; }

void SolverDDP::set_regmax(double regmax) { regmax_ = regmax; }

void SolverDDP::set_alphas(const std::vector<double>& alphas) { alphas_ = alphas; }

void SolverDDP::set_th_step(double th_step) { th_step_ = th_step; }

void SolverDDP::set_th_grad(double th_grad) { th_grad_ = th_grad; }

}  // namespace crocoddyl
