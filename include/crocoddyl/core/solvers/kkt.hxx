///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, New York University,
//                          Max Planck Gesellschaft, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
SolverKKTTpl<Scalar>::SolverKKTTpl(std::shared_ptr<ShootingProblem> problem)
    : SolverAbstract(problem),
      reg_incfactor_(Scalar(10.)),
      reg_decfactor_(Scalar(10.)),
      reg_min_(Scalar(1e-9)),
      reg_max_(Scalar(1e9)),
      th_grad_(ScaleNumerics<Scalar>(1e-12)),
      was_feasible_(false) {
  allocateData();
  const std::size_t n_alphas = 10;
  preg_ = Scalar(0.);
  dreg_ = Scalar(0.);
  alphas_.resize(n_alphas);
  for (std::size_t n = 0; n < n_alphas; ++n) {
    alphas_[n] = Scalar(1.) / pow(Scalar(2.), (Scalar)n);
  }
}

template <typename Scalar>
bool SolverKKTTpl<Scalar>::solve(const std::vector<VectorXs>& init_xs,
                                 const std::vector<VectorXs>& init_us,
                                 const std::size_t maxiter,
                                 const bool is_feasible, const Scalar) {
  setCandidate(init_xs, init_us, is_feasible);
  bool recalc = true;
  for (iter_ = 0; iter_ < maxiter; ++iter_) {
    while (true) {
      try {
        computeDirection(recalc);
      } catch (std::exception& e) {
        recalc = false;
        if (preg_ == reg_max_) {
          return false;
        } else {
          continue;
        }
      }
      break;
    }
    expectedImprovement();
    for (typename std::vector<Scalar>::const_iterator it = alphas_.begin();
         it != alphas_.end(); ++it) {
      steplength_ = *it;
      try {
        dV_ = tryStep(steplength_);
      } catch (std::exception& e) {
        continue;
      }
      dVexp_ = steplength_ * DV_[1] +
               Scalar(0.5) * steplength_ * steplength_ * DV_[2];
      if (DV_[1] < th_grad_ || !is_feasible_ || dV_ > th_acceptstep_ * dVexp_) {
        was_feasible_ = is_feasible_;
        setCandidate(xs_try_, us_try_, true);
        cost_ = cost_try_;
        break;
      }
    }
    stoppingCriteria();
    const std::size_t n_callbacks = callbacks_.size();
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

template <typename Scalar>
void SolverKKTTpl<Scalar>::computeDirection(const bool recalc) {
  const std::size_t T = problem_->get_T();
  if (recalc) {
    calcDiff();
  }
  computePrimalDual();
  const Eigen::VectorBlock<VectorXs, Eigen::Dynamic> p_x =
      primal_.segment(0, ndx_);
  const Eigen::VectorBlock<VectorXs, Eigen::Dynamic> p_u =
      primal_.segment(ndx_, nu_);

  std::size_t ix = 0;
  std::size_t iu = 0;
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::size_t ndxi = models[t]->get_state()->get_ndx();
    const std::size_t nui = models[t]->get_nu();
    dxs_[t] = p_x.segment(ix, ndxi);
    dus_[t] = p_u.segment(iu, nui);
    lambdas_[t] = dual_.segment(ix, ndxi);
    ix += ndxi;
    iu += nui;
  }
  const std::size_t ndxi =
      problem_->get_terminalModel()->get_state()->get_ndx();
  dxs_.back() = p_x.segment(ix, ndxi);
  lambdas_.back() = dual_.segment(ix, ndxi);
}

template <typename Scalar>
Scalar SolverKKTTpl<Scalar>::tryStep(const Scalar steplength) {
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& m = models[t];
    m->get_state()->integrate(xs_[t], steplength * dxs_[t], xs_try_[t]);
    if (m->get_nu() != 0) {
      us_try_[t] = us_[t];
      us_try_[t] += steplength * dus_[t];
    }
  }
  const std::shared_ptr<ActionModelAbstract> m = problem_->get_terminalModel();
  m->get_state()->integrate(xs_[T], steplength * dxs_[T], xs_try_[T]);
  cost_try_ = problem_->calc(xs_try_, us_try_);
  return cost_ - cost_try_;
}

template <typename Scalar>
Scalar SolverKKTTpl<Scalar>::stoppingCriteria() {
  const std::size_t T = problem_->get_T();
  std::size_t ix = 0;
  std::size_t iu = 0;
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
      problem_->get_runningDatas();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionDataAbstract>& d = datas[t];
    const std::size_t ndxi = models[t]->get_state()->get_ndx();
    const std::size_t nui = models[t]->get_nu();
    dF.segment(ix, ndxi) = lambdas_[t];
    dF.segment(ix, ndxi).noalias() -= d->Fx.transpose() * lambdas_[t + 1];
    dF.segment(ndx_ + iu, nui).noalias() = -lambdas_[t + 1].transpose() * d->Fu;
    ix += ndxi;
    iu += nui;
  }
  const std::size_t ndxi =
      problem_->get_terminalModel()->get_state()->get_ndx();
  dF.segment(ix, ndxi) = lambdas_.back();
  stop_ = (kktref_.segment(0, ndx_ + nu_) + dF).squaredNorm() +
          kktref_.segment(ndx_ + nu_, ndx_).squaredNorm();
  return stop_;
}

template <typename Scalar>
typename MathBaseTpl<Scalar>::Vector3s
SolverKKTTpl<Scalar>::expectedImprovement() {
  DV_.setZero();
  // -grad^T.primal
  DV_[1] = -kktref_.segment(0, ndx_ + nu_).dot(primal_);
  // -(hessian.primal)^T.primal
  kkt_primal_.noalias() = kkt_.block(0, 0, ndx_ + nu_, ndx_ + nu_) * primal_;
  DV_[2] = -kkt_primal_.dot(primal_);
  return DV_;
}

template <typename Scalar>
template <typename NewScalar>
SolverKKTTpl<NewScalar> SolverKKTTpl<Scalar>::cast() const {
  typedef SolverKKTTpl<NewScalar> ReturnType;
  typedef ShootingProblemTpl<NewScalar> ProblemType;
  ReturnType ret(
      std::make_shared<ProblemType>(problem_->template cast<NewScalar>()));
  // Setting the abstract parameters
  ret.setCallbacks(vector_cast<NewScalar>(callbacks_));
  ret.set_th_acceptstep(scalar_cast<NewScalar>(th_acceptstep_));
  ret.set_th_stop(scalar_cast<NewScalar>(th_stop_));
  // Setting the KKT parameters
  ret.set_alphas(vector_cast<NewScalar>(alphas_));
  ret.set_reg_incfactor(scalar_cast<NewScalar>(reg_incfactor_));
  ret.set_reg_decfactor(scalar_cast<NewScalar>(reg_decfactor_));
  ret.set_reg_min(scalar_cast<NewScalar>(reg_min_));
  ret.set_reg_max(scalar_cast<NewScalar>(reg_max_));
  ret.set_th_grad(scalar_cast<NewScalar>(th_grad_));
  return ret;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::MatrixXs& SolverKKTTpl<Scalar>::get_kkt()
    const {
  return kkt_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs& SolverKKTTpl<Scalar>::get_kktref()
    const {
  return kktref_;
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
SolverKKTTpl<Scalar>::get_primaldual() const {
  return primaldual_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::VectorXs>&
SolverKKTTpl<Scalar>::get_dxs() const {
  return dxs_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::VectorXs>&
SolverKKTTpl<Scalar>::get_dus() const {
  return dus_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::VectorXs>&
SolverKKTTpl<Scalar>::get_lambdas() const {
  return lambdas_;
}

template <typename Scalar>
std::size_t SolverKKTTpl<Scalar>::get_nx() const {
  return nx_;
}

template <typename Scalar>
std::size_t SolverKKTTpl<Scalar>::get_ndx() const {
  return ndx_;
}

template <typename Scalar>
std::size_t SolverKKTTpl<Scalar>::get_nu() const {
  return nu_;
}

template <typename Scalar>
Scalar SolverKKTTpl<Scalar>::calcDiff() {
  cost_ = problem_->calc(xs_, us_);
  cost_ = problem_->calcDiff(xs_, us_);
  // offset on constraint xnext = f(x,u) due to x0 = ref.
  const std::size_t cx0 =
      problem_->get_runningModels()[0]->get_state()->get_ndx();
  std::size_t ix = 0;
  std::size_t iu = 0;
  const std::size_t T = problem_->get_T();
  kkt_.block(ndx_ + nu_, 0, ndx_, ndx_).setIdentity();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& m =
        problem_->get_runningModels()[t];
    const std::shared_ptr<ActionDataAbstract>& d =
        problem_->get_runningDatas()[t];
    const std::size_t ndxi = m->get_state()->get_ndx();
    const std::size_t nui = m->get_nu();
    // Computing the gap at the initial state
    if (t == 0) {
      m->get_state()->diff(problem_->get_x0(), xs_[0],
                           kktref_.segment(ndx_ + nu_, ndxi));
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
    m->get_state()->diff(d->xnext, xs_[t + 1],
                         kktref_.segment(ndx_ + nu_ + cx0 + ix, ndxi));
    ix += ndxi;
    iu += nui;
  }
  const std::shared_ptr<ActionDataAbstract>& df = problem_->get_terminalData();
  const std::size_t ndxf =
      problem_->get_terminalModel()->get_state()->get_ndx();
  kkt_.block(ix, ix, ndxf, ndxf) = df->Lxx;
  kktref_.segment(ix, ndxf) = df->Lx;
  kkt_.block(0, ndx_ + nu_, ndx_ + nu_, ndx_) =
      kkt_.block(ndx_ + nu_, 0, ndx_, ndx_ + nu_).transpose();
  return cost_;
}

template <typename Scalar>
void SolverKKTTpl<Scalar>::computePrimalDual() {
  primaldual_ = kkt_.lu().solve(-kktref_);
  primal_ = primaldual_.segment(0, ndx_ + nu_);
  dual_ = primaldual_.segment(ndx_ + nu_, ndx_);
}

template <typename Scalar>
void SolverKKTTpl<Scalar>::allocateData() {
  const std::size_t T = problem_->get_T();
  dxs_.resize(T + 1);
  dus_.resize(T);
  lambdas_.resize(T + 1);
  xs_try_.resize(T + 1);
  us_try_.resize(T);
  nx_ = 0;
  ndx_ = 0;
  nu_ = 0;
  const std::size_t nx = problem_->get_nx();
  const std::size_t ndx = problem_->get_ndx();
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    if (t == 0) {
      xs_try_[t] = problem_->get_x0();
    } else {
      xs_try_[t] = VectorXs::Constant(nx, NAN);
    }
    const std::size_t nu = model->get_nu();
    us_try_[t] = VectorXs::Zero(nu);
    dxs_[t] = VectorXs::Zero(ndx);
    dus_[t] = VectorXs::Zero(nu);
    lambdas_[t] = VectorXs::Zero(ndx);
    nx_ += nx;
    ndx_ += ndx;
    nu_ += nu;
  }
  nx_ += nx;
  ndx_ += ndx;
  xs_try_.back() = problem_->get_terminalModel()->get_state()->zero();
  dxs_.back() = VectorXs::Zero(ndx);
  lambdas_.back() = VectorXs::Zero(ndx);
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

template <typename Scalar>
void SolverKKTTpl<Scalar>::set_alphas(const std::vector<Scalar>& alphas) {
  Scalar prev_alpha = alphas[0];
  if (prev_alpha != Scalar(1.)) {
    std::cerr << "Warning: alpha[0] should be 1" << std::endl;
  }
  for (std::size_t i = 1; i < alphas.size(); ++i) {
    Scalar alpha = alphas[i];
    if (Scalar(0.) >= alpha) {
      throw_pretty("Invalid argument: " << "alpha values has to be positive.");
    }
    if (alpha >= prev_alpha) {
      throw_pretty(
          "Invalid argument: " << "alpha values are monotonously decreasing.");
    }
    prev_alpha = alpha;
  }
  alphas_ = alphas;
}

template <typename Scalar>
void SolverKKTTpl<Scalar>::set_reg_incfactor(const Scalar regfactor) {
  if (regfactor <= Scalar(1.)) {
    throw_pretty(
        "Invalid argument: " << "reg_incfactor value is higher than 1.");
  }
  reg_incfactor_ = regfactor;
}

template <typename Scalar>
void SolverKKTTpl<Scalar>::set_reg_decfactor(const Scalar regfactor) {
  if (regfactor <= Scalar(1.)) {
    throw_pretty(
        "Invalid argument: " << "reg_decfactor value is higher than 1.");
  }
  reg_decfactor_ = regfactor;
}

template <typename Scalar>
void SolverKKTTpl<Scalar>::set_reg_min(const Scalar regmin) {
  if (Scalar(0.) > regmin) {
    throw_pretty("Invalid argument: " << "regmin value has to be positive.");
  }
  reg_min_ = regmin;
}

template <typename Scalar>
void SolverKKTTpl<Scalar>::set_reg_max(const Scalar regmax) {
  if (Scalar(0.) > regmax) {
    throw_pretty("Invalid argument: " << "regmax value has to be positive.");
  }
  reg_max_ = regmax;
}

template <typename Scalar>
void SolverKKTTpl<Scalar>::set_th_grad(const Scalar th_grad) {
  if (Scalar(0.) > th_grad) {
    throw_pretty("Invalid argument: " << "th_grad value has to be positive.");
  }
  th_grad_ = th_grad;
}

}  // namespace crocoddyl
