#include <crocoddyl/core/solvers/ddp.hpp>

namespace crocoddyl {

SolverDDP::SolverDDP(ShootingProblem& problem)
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

SolverDDP::~SolverDDP() {}

bool SolverDDP::solve(const std::vector<Eigen::VectorXd>& init_xs, const std::vector<Eigen::VectorXd>& init_us,
                      const unsigned int& maxiter, const bool& is_feasible, const double& reginit) {
  setCandidate(init_xs, init_us, is_feasible);

  if (std::isnan(reginit)) {
    xreg_ = regmin_;
    ureg_ = regmin_;
  } else {
    xreg_ = reginit;
    ureg_ = reginit;
  }
  was_feasible_ = false;

  for (iter_ = 0; iter_ < maxiter; ++iter_) {
    bool recalc = true;
    while (true) {
      try {
        computeDirection(recalc);
      } catch (const char* msg) {
        recalc = false;
        if (xreg_ == regmax_) {
          return false;
        } else {
          continue;
        }
      }
      break;
    }
    expectedImprovement();

    for (std::vector<double>::const_iterator it = alphas_.begin(); it != alphas_.end(); ++it) {
      steplength_ = *it;

      try {
        dV_ = tryStep(steplength_);
      } catch (const char* msg) {
        continue;
      }
      dVexp_ = steplength_ * (d_[0] + 0.5 * steplength_ * d_[1]);

      if (d_[0] < th_grad_ || !is_feasible_ || dV_ > th_acceptstep_ * dVexp_) {
        was_feasible_ = is_feasible_;
        setCandidate(xs_try_, us_try_, true);
        cost_ = cost_try_;
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

    const long unsigned int& n_callbacks = callbacks_.size();
    if (n_callbacks != 0) {
      for (long unsigned int c = 0; c < n_callbacks; ++c) {
        CallbackAbstract& callback = *callbacks_[c];
        callback(this);
      }
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
  const long unsigned int& T = this->problem_.get_T();
  for (long unsigned int t = 0; t < T; ++t) {
    stop_ += Qu_[t].squaredNorm();
  }
  return stop_;
}

const Eigen::Vector2d& SolverDDP::expectedImprovement() {
  d_ = Eigen::Vector2d::Zero();
  const long unsigned int& T = this->problem_.get_T();
  for (long unsigned int t = 0; t < T; ++t) {
    d_[0] += Qu_[t].dot(k_[t]);
    d_[1] -= k_[t].dot(Quu_[t] * k_[t]);
  }
  return d_;
}

double SolverDDP::calc() {
  cost_ = problem_.calcDiff(xs_, us_);
  if (!is_feasible_) {
    const Eigen::VectorXd& x0 = problem_.get_x0();
    problem_.running_models_[0]->get_state()->diff(xs_[0], x0, gaps_[0]);

    const long unsigned int& T = problem_.get_T();
    for (unsigned long int t = 0; t < T; ++t) {
      ActionModelAbstract* model = problem_.running_models_[t];
      std::shared_ptr<ActionDataAbstract>& d = problem_.running_datas_[t];
      model->get_state()->diff(xs_[t + 1], d->get_xnext(), gaps_[t + 1]);
    }
  }
  return cost_;
}

void SolverDDP::backwardPass() {
  std::shared_ptr<ActionDataAbstract>& d_T = problem_.terminal_data_;
  Vxx_.back() = d_T->get_Lxx();
  Vx_.back() = d_T->get_Lx();

  const int& ndx = problem_.terminal_model_->get_ndx();
  const Eigen::VectorXd& xreg = Eigen::VectorXd::Constant(ndx, xreg_);
  if (!std::isnan(xreg_)) {
    Vxx_.back().diagonal() += xreg;
  }

  for (int t = (int)problem_.get_T() - 1; t >= 0; --t) {
    ActionModelAbstract* m = problem_.running_models_[t];
    std::shared_ptr<ActionDataAbstract>& d = problem_.running_datas_[t];
    const Eigen::MatrixXd& Vxx_p = Vxx_[t + 1];
    const Eigen::VectorXd& Vx_p = Vx_[t + 1];
    const Eigen::VectorXd& gap_p = gaps_[t + 1];

    const Eigen::MatrixXd& FxTVxx_p = d->get_Fx().transpose() * Vxx_p;
    Qxx_[t] = d->get_Lxx() + FxTVxx_p * d->get_Fx();
    Qxu_[t] = d->get_Lxu() + FxTVxx_p * d->get_Fu();
    Quu_[t].noalias() = d->get_Luu() + d->get_Fu().transpose() * Vxx_p * d->get_Fu();
    if (!is_feasible_) {
      // In case the xt+1 are not f(xt,ut) i.e warm start not obtained from roll-out.
      const Eigen::VectorXd& relinearization = Vxx_p * gap_p;
      Qx_[t] = d->get_Lx() + d->get_Fx().transpose() * Vx_p + d->get_Fx().transpose() * relinearization;
      Qu_[t] = d->get_Lu() + d->get_Fu().transpose() * Vx_p + d->get_Fu().transpose() * relinearization;
    } else {
      Qx_[t] = d->get_Lx() + d->get_Fx().transpose() * Vx_p;
      Qu_[t] = d->get_Lu() + d->get_Fu().transpose() * Vx_p;
    }

    if (!std::isnan(ureg_)) {
      const int& nu = m->get_nu();
      Quu_[t].diagonal() += Eigen::VectorXd::Constant(nu, ureg_);
    }

    computeGains(t);

    if (std::isnan(ureg_)) {
      Vx_[t] = Qx_[t] - K_[t].transpose() * Qu_[t];
    } else {
      Vx_[t] = Qx_[t] + K_[t].transpose() * (Quu_[t] * k_[t] - 2. * Qu_[t]);
    }
    Vxx_[t] = Qxx_[t] - Qxu_[t] * K_[t];
    Vxx_[t] = 0.5 * (Vxx_[t] + Vxx_[t].transpose());  // TODO: as suggested by Nicolas

    if (!std::isnan(xreg_)) {
      Vxx_[t].diagonal() += xreg;
    }

    const double& Vx_value = Vx_[t].sum();
    const double& Vxx_value = Vxx_[t].sum();
    if (std::isnan(Vx_value) || std::isnan(Vxx_value)) {
      throw "backward error";
    }
  }
}

void SolverDDP::forwardPass(const double& steplength) {
  cost_try_ = 0.;
  const long unsigned int& T = problem_.get_T();
  for (long unsigned int t = 0; t < T; ++t) {
    ActionModelAbstract* m = problem_.running_models_[t];
    std::shared_ptr<ActionDataAbstract>& d = problem_.running_datas_[t];

    m->get_state()->diff(xs_[t], xs_try_[t], dx_[t]);
    us_try_[t] = us_[t] - k_[t] * steplength - K_[t] * dx_[t];
    m->calc(d, xs_try_[t], us_try_[t]);
    xs_try_[t + 1] = d->get_xnext();
    cost_try_ += d->cost;

    const double& value = xs_try_[t + 1].sum();
    if (std::isnan(value) || std::isinf(value) || std::isnan(cost_try_) || std::isnan(cost_try_)) {
      throw "forward error";
    }
  }

  ActionModelAbstract* m = problem_.terminal_model_;
  std::shared_ptr<ActionDataAbstract>& d = problem_.terminal_data_;
  m->calc(d, xs_try_.back());
  cost_try_ += d->cost;

  if (std::isnan(cost_try_) || std::isnan(cost_try_)) {
    throw "forward error";
  }
}

void SolverDDP::computeGains(const long unsigned int& t) {
  const Eigen::LLT<Eigen::MatrixXd>& Lb = Quu_[t].llt();
  K_[t] = Lb.solve(Qxu_[t].transpose());
  k_[t] = Lb.solve(Qu_[t]);
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
  const long unsigned int& T = problem_.get_T();
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

  xs_.resize(T + 1);
  us_.resize(T);
  xs_try_.resize(T + 1);
  us_try_.resize(T);
  dx_.resize(T);

  for (long unsigned int t = 0; t < T; ++t) {
    ActionModelAbstract* model = problem_.running_models_[t];
    const int& nx = model->get_nx();
    const int& ndx = model->get_ndx();
    const int& nu = model->get_nu();

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

    xs_[t] = model->get_state()->zero();
    us_[t] = Eigen::VectorXd::Zero(nu);
    if (t == 0) {
      xs_try_[t] = problem_.get_x0();
    } else {
      xs_try_[t] = Eigen::VectorXd::Constant(nx, NAN);
    }
    us_try_[t] = Eigen::VectorXd::Constant(nu, NAN);
    dx_[t] = Eigen::VectorXd::Zero(ndx);
  }
  const int& ndx = problem_.terminal_model_->get_ndx();
  Vxx_.back() = Eigen::MatrixXd::Zero(ndx, ndx);
  Vx_.back() = Eigen::VectorXd::Zero(ndx);
  xs_.back() = problem_.terminal_model_->get_state()->zero();
  xs_try_.back() = problem_.terminal_model_->get_state()->zero();
  gaps_.back() = Eigen::VectorXd::Zero(ndx);
}

}  // namespace crocoddyl
