///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, New York University, Max Planck Gesellschaft,
//                          University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/solvers/kkt.hpp"

namespace crocoddyl {

SolverKKT::SolverKKT(boost::shared_ptr<ShootingProblem> problem)
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
  xreg_ = 0.;
  ureg_ = 0.;
  alphas_.resize(n_alphas);
  for (unsigned int n = 0; n < n_alphas; ++n) {
    alphas_[n] = 1. / pow(2., (double)n);
  }
}

SolverKKT::~SolverKKT() {}

bool SolverKKT::solve(const std::vector<Eigen::VectorXd>& init_xs, const std::vector<Eigen::VectorXd>& init_us,
                      const std::size_t& maxiter, const bool& is_feasible, const double&) {
  setCandidate(init_xs, init_us, is_feasible);
  bool recalc = true;
  for (iter_ = 0; iter_ < maxiter; ++iter_) {
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
      dVexp_ = steplength_ * d_[0] + 0.5 * steplength_ * steplength_ * d_[1];
      if (d_[0] < th_grad_ || !is_feasible_ || dV_ > th_acceptstep_ * dVexp_) {
        was_feasible_ = is_feasible_;
        setCandidate(xs_try_, us_try_, true);
        cost_ = cost_try_;
        break;
      }
    }
    stoppingCriteria();
    const std::size_t& n_callbacks = callbacks_.size();
    if (n_callbacks != 0) {
      for (std::size_t c = 0; c < n_callbacks; ++c) {
        CallbackAbstract& callback = *callbacks_[c];
        callback(*this);
      }
    }
    if (was_feasible_ && stop_ < th_stop_) {
      return true;
    }
  }
  return false;
}

void SolverKKT::computeDirection(const bool& recalc) {
  const std::size_t& T = problem_->get_T();
  if (recalc) {
    calc();
  }
  computePrimalDual();
  const Eigen::VectorBlock<Eigen::VectorXd, Eigen::Dynamic> p_x = primal_.segment(0, ndx_);
  const Eigen::VectorBlock<Eigen::VectorXd, Eigen::Dynamic> p_u = primal_.segment(ndx_, nu_);

  std::size_t ix = 0;
  std::size_t iu = 0;
  for (std::size_t t = 0; t < T; ++t) {
    const std::size_t& ndxi = problem_->running_models_[t]->get_state()->get_ndx();
    const std::size_t& nui = problem_->running_models_[t]->get_nu();
    dxs_[t] = p_x.segment(ix, ndxi);
    dus_[t] = p_u.segment(iu, nui);
    lambdas_[t] = dual_.segment(ix, ndxi);
    ix += ndxi;
    iu += nui;
  }
  const std::size_t& ndxi = problem_->terminal_model_->get_state()->get_ndx();
  dxs_.back() = p_x.segment(ix, ndxi);
  lambdas_.back() = dual_.segment(ix, ndxi);
}

double SolverKKT::tryStep(const double& steplength) {
  const std::size_t& T = problem_->get_T();
  for (std::size_t t = 0; t < T; ++t) {
    const boost::shared_ptr<ActionModelAbstract>& m = problem_->running_models_[t];

    m->get_state()->integrate(xs_[t], steplength * dxs_[t], xs_try_[t]);
    us_try_[t] = us_[t];
    us_try_[t] += steplength * dus_[t];
  }
  const boost::shared_ptr<ActionModelAbstract> m = problem_->terminal_model_;
  m->get_state()->integrate(xs_[T], steplength * dxs_[T], xs_try_[T]);
  cost_try_ = problem_->calc(xs_try_, us_try_);
  return cost_ - cost_try_;
}

double SolverKKT::stoppingCriteria() {
  const std::size_t& T = problem_->get_T();
  std::size_t ix = 0;
  std::size_t iu = 0;
  for (std::size_t t = 0; t < T; ++t) {
    const boost::shared_ptr<ActionDataAbstract>& d = problem_->running_datas_[t];
    const std::size_t& ndxi = problem_->running_models_[t]->get_state()->get_ndx();
    const std::size_t& nui = problem_->running_models_[t]->get_nu();

    dF.segment(ix, ndxi) = lambdas_[t];
    dF.segment(ix, ndxi).noalias() -= d->Fx.transpose() * lambdas_[t + 1];
    dF.segment(ndx_ + iu, nui).noalias() = -lambdas_[t + 1].transpose() * d->Fu;
    ix += ndxi;
    iu += nui;
  }
  const std::size_t& ndxi = problem_->terminal_model_->get_state()->get_ndx();
  dF.segment(ix, ndxi) = lambdas_.back();
  stop_ = (kktref_.segment(0, ndx_ + nu_) + dF).squaredNorm() + kktref_.segment(ndx_ + nu_, ndx_).squaredNorm();
  return stop_;
}

const Eigen::Vector2d& SolverKKT::expectedImprovement() {
  d_ = Eigen::Vector2d::Zero();
  // -grad^T.primal
  d_(0) = -kktref_.segment(0, ndx_ + nu_).dot(primal_);
  // -(hessian.primal)^T.primal
  kkt_primal_.noalias() = kkt_.block(0, 0, ndx_ + nu_, ndx_ + nu_) * primal_;
  d_(1) = -kkt_primal_.dot(primal_);
  return d_;
}

const Eigen::MatrixXd& SolverKKT::get_kkt() const { return kkt_; }

const Eigen::VectorXd& SolverKKT::get_kktref() const { return kktref_; }

const Eigen::VectorXd& SolverKKT::get_primaldual() const { return primaldual_; }

const std::vector<Eigen::VectorXd>& SolverKKT::get_dxs() const { return dxs_; }

const std::vector<Eigen::VectorXd>& SolverKKT::get_dus() const { return dus_; }

const std::size_t& SolverKKT::get_nx() const { return nx_; }

const std::size_t& SolverKKT::get_ndx() const { return ndx_; }

const std::size_t& SolverKKT::get_nu() const { return nu_; }

double SolverKKT::calc() {
  cost_ = problem_->calcDiff(xs_, us_);

  // offset on constraint xnext = f(x,u) due to x0 = ref.
  const std::size_t& cx0 = problem_->get_runningModels()[0]->get_state()->get_ndx();

  std::size_t ix = 0;
  std::size_t iu = 0;
  const std::size_t& T = problem_->get_T();
  kkt_.block(ndx_ + nu_, 0, ndx_, ndx_) = Eigen::MatrixXd::Identity(ndx_, ndx_);
  for (std::size_t t = 0; t < T; ++t) {
    const boost::shared_ptr<ActionModelAbstract>& m = problem_->running_models_[t];
    const boost::shared_ptr<ActionDataAbstract>& d = problem_->running_datas_[t];
    const std::size_t& ndxi = m->get_state()->get_ndx();
    const std::size_t& nui = m->get_nu();

    // Computing the gap at the initial state
    if (t == 0) {
      m->get_state()->diff(problem_->get_x0(), xs_[0], kktref_.segment(ndx_ + nu_, ndxi));
    }

    // Filling KKT matrix
    kkt_.block(ix, ix, ndxi, ndxi) = d->Lxx;
    kkt_.block(ix, ndx_ + iu, ndxi, nui) = d->Lxu;
    kkt_.block(ndx_ + iu, ix, nui, ndxi) = d->Lxu.transpose();
    kkt_.block(ndx_ + iu, ndx_ + iu, nui, nui) = d->Luu;
    kkt_.block(ndx_ + nu_ + cx0 + ix, ix, ndxi, ndxi) = -d->Fx;
    kkt_.block(ndx_ + nu_ + cx0 + ix, ndx_ + iu, ndxi, nui) = -d->Fu;

    // Filling KKT vector
    kktref_.segment(ix, ndxi) = d->Lx;
    kktref_.segment(ndx_ + iu, nui) = d->Lu;
    m->get_state()->diff(d->xnext, xs_[t + 1], kktref_.segment(ndx_ + nu_ + cx0 + ix, ndxi));

    ix += ndxi;
    iu += nui;
  }
  boost::shared_ptr<ActionDataAbstract>& df = problem_->terminal_data_;
  const std::size_t& ndxf = problem_->terminal_model_->get_state()->get_ndx();
  kkt_.block(ix, ix, ndxf, ndxf) = df->Lxx;
  kktref_.segment(ix, ndxf) = df->Lx;
  kkt_.block(0, ndx_ + nu_, ndx_ + nu_, ndx_) = kkt_.block(ndx_ + nu_, 0, ndx_, ndx_ + nu_).transpose();
  return cost_;
}

void SolverKKT::computePrimalDual() {
  primaldual_ = kkt_.lu().solve(-kktref_);
  primal_ = primaldual_.segment(0, ndx_ + nu_);
  dual_ = primaldual_.segment(ndx_ + nu_, ndx_);
}

void SolverKKT::increaseRegularization() {
  xreg_ *= regfactor_;
  if (xreg_ > regmax_) {
    xreg_ = regmax_;
  }
  ureg_ = xreg_;
}

void SolverKKT::decreaseRegularization() {
  xreg_ /= regfactor_;
  if (xreg_ < regmin_) {
    xreg_ = regmin_;
  }
  ureg_ = xreg_;
}

void SolverKKT::allocateData() {
  const std::size_t& T = problem_->get_T();
  dxs_.resize(T + 1);
  dus_.resize(T);
  lambdas_.resize(T + 1);
  xs_try_.resize(T + 1);
  us_try_.resize(T);

  nx_ = 0;
  ndx_ = 0;
  nu_ = 0;
  for (std::size_t t = 0; t < T; ++t) {
    const boost::shared_ptr<ActionModelAbstract>& model = problem_->running_models_[t];
    const std::size_t& nx = model->get_state()->get_nx();
    const std::size_t& ndx = model->get_state()->get_ndx();
    const std::size_t& nu = model->get_nu();
    if (t == 0) {
      xs_try_[t] = problem_->get_x0();
    } else {
      xs_try_[t] = Eigen::VectorXd::Constant(nx, NAN);
    }
    us_try_[t] = Eigen::VectorXd::Constant(nu, NAN);
    dxs_[t] = Eigen::VectorXd::Zero(ndx);
    dus_[t] = Eigen::VectorXd::Zero(nu);
    lambdas_[t] = Eigen::VectorXd::Zero(ndx);
    nx_ += nx;
    ndx_ += ndx;
    nu_ += nu;
  }
  const boost::shared_ptr<ActionModelAbstract>& model = problem_->get_terminalModel();
  nx_ += model->get_state()->get_nx();
  ndx_ += model->get_state()->get_ndx();
  xs_try_.back() = problem_->terminal_model_->get_state()->zero();
  dxs_.back() = Eigen::VectorXd::Zero(model->get_state()->get_ndx());
  lambdas_.back() = Eigen::VectorXd::Zero(model->get_state()->get_ndx());

  // Set dimensions for kkt matrix and kkt_ref vector
  kkt_.resize(2 * ndx_ + nu_, 2 * ndx_ + nu_);
  kkt_.setZero();
  kktref_.resize(2 * ndx_ + nu_);
  kktref_.setZero();
  primaldual_.resize(2 * ndx_ + nu_);
  primaldual_.setZero();
  primal_.resize(ndx_ + nu_);
  primal_.setZero();
  kkt_primal_.resize(ndx_ + nu_);
  kkt_primal_.setZero();
  dual_.resize(ndx_);
  dual_.setZero();
  dF.resize(ndx_ + nu_);
  dF.setZero();
}

}  // namespace crocoddyl
