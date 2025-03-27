///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2023, Heriot-Watt University, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/solvers/intro.hpp"

#include "crocoddyl/core/utils/stop-watch.hpp"

namespace crocoddyl {

SolverIntro::SolverIntro(std::shared_ptr<ShootingProblem> problem)
    : SolverFDDP(problem),
      eq_solver_(LuNull),
      th_feas_(1e-4),
      rho_(0.3),
      upsilon_(0.),
      zero_upsilon_(false) {
  const std::size_t T = problem_->get_T();
  Hu_rank_.resize(T);
  KQuu_tmp_.resize(T);
  YZ_.resize(T);
  Hy_.resize(T);
  Qz_.resize(T);
  Qzz_.resize(T);
  Qxz_.resize(T);
  Quz_.resize(T);
  kz_.resize(T);
  Kz_.resize(T);
  ks_.resize(T);
  Ks_.resize(T);
  QuuinvHuT_.resize(T);
  Qzz_llt_.resize(T);
  Hu_lu_.resize(T);
  Hu_qr_.resize(T);
  Hy_lu_.resize(T);

  const std::size_t ndx = problem_->get_ndx();
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    const std::size_t nh = model->get_nh();
    Hu_rank_[t] = nh;
    KQuu_tmp_[t] = Eigen::MatrixXd::Zero(ndx, nu);
    YZ_[t] = Eigen::MatrixXd::Zero(nu, nu);
    Hy_[t] = Eigen::MatrixXd::Zero(nh, nh);
    Qz_[t] = Eigen::VectorXd::Zero(nh);
    Qzz_[t] = Eigen::MatrixXd::Zero(nh, nh);
    Qxz_[t] = Eigen::MatrixXd::Zero(ndx, nh);
    Quz_[t] = Eigen::MatrixXd::Zero(nu, nh);
    kz_[t] = Eigen::VectorXd::Zero(nu);
    Kz_[t] = Eigen::MatrixXd::Zero(nu, ndx);
    ks_[t] = Eigen::VectorXd::Zero(nh);
    Ks_[t] = Eigen::MatrixXd::Zero(nh, ndx);
    QuuinvHuT_[t] = Eigen::MatrixXd::Zero(nu, nh);
    Qzz_llt_[t] = Eigen::LLT<Eigen::MatrixXd>(nh);
    Hu_lu_[t] = Eigen::FullPivLU<Eigen::MatrixXd>(nh, nu);
    Hu_qr_[t] = Eigen::ColPivHouseholderQR<Eigen::MatrixXd>(nu, nh);
    Hy_lu_[t] = Eigen::PartialPivLU<Eigen::MatrixXd>(nh);
  }
}

SolverIntro::~SolverIntro() {}

bool SolverIntro::solve(const std::vector<Eigen::VectorXd>& init_xs,
                        const std::vector<Eigen::VectorXd>& init_us,
                        const std::size_t maxiter, const bool is_feasible,
                        const double init_reg) {
  START_PROFILER("SolverIntro::solve");
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
  if (zero_upsilon_) {
    upsilon_ = 0.;
  }

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
    expectedImprovement();

    // Update the penalty parameter for computing the merit function and its
    // directional derivative For more details see Section 3 of "An Interior
    // Point Algorithm for Large Scale Nonlinear Programming"
    if (hfeas_ != 0 && iter_ != 0) {
      upsilon_ =
          std::max(upsilon_, (d_[0] + .5 * d_[1]) / ((1 - rho_) * hfeas_));
    }

    // We need to recalculate the derivatives when the step length passes
    recalcDiff = false;
    for (std::vector<double>::const_iterator it = alphas_.begin();
         it != alphas_.end(); ++it) {
      steplength_ = *it;
      try {
        dV_ = tryStep(steplength_);
        dfeas_ = hfeas_ - hfeas_try_;
        dPhi_ = dV_ + upsilon_ * dfeas_;
      } catch (std::exception& e) {
        continue;
      }
      expectedImprovement();
      dVexp_ = steplength_ * (d_[0] + 0.5 * steplength_ * d_[1]);
      dPhiexp_ = dVexp_ + steplength_ * upsilon_ * dfeas_;
      if (dPhiexp_ >= 0) {  // descend direction
        if (std::abs(d_[0]) < th_grad_ || dPhi_ > th_acceptstep_ * dPhiexp_) {
          was_feasible_ = is_feasible_;
          setCandidate(xs_try_, us_try_, (was_feasible_) || (steplength_ == 1));
          cost_ = cost_try_;
          hfeas_ = hfeas_try_;
          merit_ = cost_ + upsilon_ * hfeas_;
          recalcDiff = true;
          break;
        }
      } else {  // reducing the gaps by allowing a small increment in the cost
                // value
        if (dV_ > th_acceptnegstep_ * dVexp_) {
          was_feasible_ = is_feasible_;
          setCandidate(xs_try_, us_try_, (was_feasible_) || (steplength_ == 1));
          cost_ = cost_try_;
          hfeas_ = hfeas_try_;
          merit_ = cost_ + upsilon_ * hfeas_;
          recalcDiff = true;
          break;
        }
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
    if (steplength_ <= th_stepinc_ || std::abs(d_[1]) <= th_feas_) {
      if (preg_ == reg_max_) {
        STOP_PROFILER("SolverIntro::solve");
        return false;
      }
      increaseRegularization();
    }

    if (is_feasible_ && stop_ < th_stop_) {
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
  stop_ = std::max(hfeas_, std::abs(d_[0] + 0.5 * d_[1]));
  return stop_;
}

void SolverIntro::resizeData() {
  START_PROFILER("SolverIntro::resizeData");
  SolverFDDP::resizeData();

  const std::size_t T = problem_->get_T();
  const std::size_t ndx = problem_->get_ndx();
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    const std::size_t nh = model->get_nh();
    KQuu_tmp_[t].conservativeResize(ndx, nu);
    YZ_[t].conservativeResize(nu, nu);
    Hy_[t].conservativeResize(nh, nh);
    Qz_[t].conservativeResize(nh);
    Qzz_[t].conservativeResize(nh, nh);
    Qxz_[t].conservativeResize(ndx, nh);
    Quz_[t].conservativeResize(nu, nh);
    kz_[t].conservativeResize(nu);
    Kz_[t].conservativeResize(nu, ndx);
    ks_[t].conservativeResize(nh);
    Ks_[t].conservativeResize(nh, ndx);
    QuuinvHuT_[t].conservativeResize(nu, nh);
  }
  STOP_PROFILER("SolverIntro::resizeData");
}

double SolverIntro::calcDiff() {
  START_PROFILER("SolverIntro::calcDiff");
  SolverFDDP::calcDiff();
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
      problem_->get_runningDatas();
  switch (eq_solver_) {
    case LuNull:
#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(problem_->get_nthreads())
#endif
      for (std::size_t t = 0; t < T; ++t) {
        const std::shared_ptr<crocoddyl::ActionModelAbstract>& model =
            models[t];
        const std::shared_ptr<crocoddyl::ActionDataAbstract>& data = datas[t];
        if (model->get_nu() > 0 && model->get_nh() > 0) {
          Hu_lu_[t].compute(data->Hu);
          Hu_rank_[t] = Hu_lu_[t].rank();
          YZ_[t].leftCols(Hu_rank_[t]).noalias() =
              (Hu_lu_[t].permutationP() * data->Hu).transpose();
          YZ_[t].rightCols(model->get_nu() - Hu_rank_[t]) = Hu_lu_[t].kernel();
          const Eigen::Block<Eigen::MatrixXd, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::RowMajor>
              Y = YZ_[t].leftCols(Hu_lu_[t].rank());
          Hy_[t].noalias() = data->Hu * Y;
          Hy_lu_[t].compute(Hy_[t]);
          const Eigen::Inverse<Eigen::PartialPivLU<Eigen::MatrixXd> > Hy_inv =
              Hy_lu_[t].inverse();
          ks_[t].noalias() = Hy_inv * data->h;
          Ks_[t].noalias() = Hy_inv * data->Hx;
          kz_[t].noalias() = Y * ks_[t];
          Kz_[t].noalias() = Y * Ks_[t];
        }
      }
      break;
    case QrNull:
#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(problem_->get_nthreads())
#endif
      for (std::size_t t = 0; t < T; ++t) {
        const std::shared_ptr<crocoddyl::ActionModelAbstract>& model =
            models[t];
        const std::shared_ptr<crocoddyl::ActionDataAbstract>& data = datas[t];
        if (model->get_nu() > 0 && model->get_nh() > 0) {
          Hu_qr_[t].compute(data->Hu.transpose());
          YZ_[t] = Hu_qr_[t].householderQ();
          Hu_rank_[t] = Hu_qr_[t].rank();
          const Eigen::Block<Eigen::MatrixXd, Eigen::Dynamic, Eigen::Dynamic,
                             Eigen::RowMajor>
              Y = YZ_[t].leftCols(Hu_qr_[t].rank());
          Hy_[t].noalias() = data->Hu * Y;
          Hy_lu_[t].compute(Hy_[t]);
          const Eigen::Inverse<Eigen::PartialPivLU<Eigen::MatrixXd> > Hy_inv =
              Hy_lu_[t].inverse();
          ks_[t].noalias() = Hy_inv * data->h;
          Ks_[t].noalias() = Hy_inv * data->Hx;
          kz_[t].noalias() = Y * ks_[t];
          Kz_[t].noalias() = Y * Ks_[t];
        }
      }
      break;
    case Schur:
      break;
  }

  STOP_PROFILER("SolverIntro::calcDiff");
  return cost_;
}

void SolverIntro::computeValueFunction(
    const std::size_t t, const std::shared_ptr<ActionModelAbstract>& model) {
  const std::size_t nu = model->get_nu();
  Vx_[t] = Qx_[t];
  Vxx_[t] = Qxx_[t];
  if (nu != 0) {
    START_PROFILER("SolverIntro::Vx");
    Quuk_[t].noalias() = Quu_[t] * k_[t];
    Vx_[t].noalias() -= Qxu_[t] * k_[t];
    Qu_[t] -= Quuk_[t];
    Vx_[t].noalias() -= K_[t].transpose() * Qu_[t];
    Qu_[t] += Quuk_[t];
    STOP_PROFILER("SolverIntro::Vx");
    START_PROFILER("SolverIntro::Vxx");
    KQuu_tmp_[t].noalias() = K_[t].transpose() * Quu_[t];
    KQuu_tmp_[t].noalias() -= 2 * Qxu_[t];
    Vxx_[t].noalias() += KQuu_tmp_[t] * K_[t];
    STOP_PROFILER("SolverIntro::Vxx");
  }
  Vxx_tmp_ = 0.5 * (Vxx_[t] + Vxx_[t].transpose());
  Vxx_[t] = Vxx_tmp_;

  if (!std::isnan(preg_)) {
    Vxx_[t].diagonal().array() += preg_;
  }

  // Compute and store the Vx gradient at end of the interval (rollout state)
  if (!is_feasible_) {
    Vx_[t].noalias() += Vxx_[t] * fs_[t];
  }
}

void SolverIntro::computeGains(const std::size_t t) {
  START_PROFILER("SolverIntro::computeGains");
  const std::shared_ptr<crocoddyl::ActionModelAbstract>& model =
      problem_->get_runningModels()[t];
  const std::shared_ptr<crocoddyl::ActionDataAbstract>& data =
      problem_->get_runningDatas()[t];

  const std::size_t nu = model->get_nu();
  const std::size_t nh = model->get_nh();
  switch (eq_solver_) {
    case LuNull:
    case QrNull:
      if (nu > 0 && nh > 0) {
        START_PROFILER("SolverIntro::Qzz_inv");
        const std::size_t rank = Hu_rank_[t];
        const std::size_t nullity = data->Hu.cols() - rank;
        const Eigen::Block<Eigen::MatrixXd, Eigen::Dynamic, Eigen::Dynamic,
                           Eigen::RowMajor>
            Z = YZ_[t].rightCols(nullity);
        Quz_[t].noalias() = Quu_[t] * Z;
        Qzz_[t].noalias() = Z.transpose() * Quz_[t];
        Qzz_llt_[t].compute(Qzz_[t]);
        STOP_PROFILER("SolverIntro::Qzz_inv");
        const Eigen::ComputationInfo& info = Qzz_llt_[t].info();
        if (info != Eigen::Success) {
          throw_pretty("backward error");
        }

        k_[t] = kz_[t];
        K_[t] = Kz_[t];
        Eigen::Transpose<Eigen::MatrixXd> QzzinvQzu = Quz_[t].transpose();
        Qzz_llt_[t].solveInPlace(QzzinvQzu);
        Qz_[t].noalias() = Z.transpose() * Qu_[t];
        Qzz_llt_[t].solveInPlace(Qz_[t]);
        Qxz_[t].noalias() = Qxu_[t] * Z;
        Eigen::Transpose<Eigen::MatrixXd> Qzx = Qxz_[t].transpose();
        Qzz_llt_[t].solveInPlace(Qzx);
        Qz_[t].noalias() -= QzzinvQzu * kz_[t];
        Qzx.noalias() -= QzzinvQzu * Kz_[t];
        k_[t].noalias() += Z * Qz_[t];
        K_[t].noalias() += Z * Qzx;
      } else {
        SolverFDDP::computeGains(t);
      }
      break;
    case Schur:
      SolverFDDP::computeGains(t);
      if (nu > 0 && nh > 0) {
        START_PROFILER("SolverIntro::Qzz_inv");
        QuuinvHuT_[t] = data->Hu.transpose();
        Quu_llt_[t].solveInPlace(QuuinvHuT_[t]);
        Qzz_[t].noalias() = data->Hu * QuuinvHuT_[t];
        Qzz_llt_[t].compute(Qzz_[t]);
        STOP_PROFILER("SolverIntro::Qzz_inv");
        const Eigen::ComputationInfo& info = Qzz_llt_[t].info();
        if (info != Eigen::Success) {
          throw_pretty("backward error");
        }
        Eigen::Transpose<Eigen::MatrixXd> HuQuuinv = QuuinvHuT_[t].transpose();
        Qzz_llt_[t].solveInPlace(HuQuuinv);
        ks_[t] = data->h;
        ks_[t].noalias() -= data->Hu * k_[t];
        Ks_[t] = data->Hx;
        Ks_[t].noalias() -= data->Hu * K_[t];
        k_[t].noalias() += QuuinvHuT_[t] * ks_[t];
        K_[t] += QuuinvHuT_[t] * Ks_[t];
      }
      break;
  }
  STOP_PROFILER("SolverIntro::computeGains");
}

EqualitySolverType SolverIntro::get_equality_solver() const {
  return eq_solver_;
}

double SolverIntro::get_th_feas() const { return th_feas_; }

double SolverIntro::get_rho() const { return rho_; }

double SolverIntro::get_upsilon() const { return upsilon_; }

bool SolverIntro::get_zero_upsilon() const { return zero_upsilon_; }

const std::vector<std::size_t>& SolverIntro::get_Hu_rank() const {
  return Hu_rank_;
}

const std::vector<Eigen::MatrixXd>& SolverIntro::get_YZ() const { return YZ_; }

const std::vector<Eigen::MatrixXd>& SolverIntro::get_Qzz() const {
  return Qzz_;
}

const std::vector<Eigen::MatrixXd>& SolverIntro::get_Qxz() const {
  return Qxz_;
}

const std::vector<Eigen::MatrixXd>& SolverIntro::get_Quz() const {
  return Quz_;
}

const std::vector<Eigen::VectorXd>& SolverIntro::get_Qz() const { return Qz_; }

const std::vector<Eigen::MatrixXd>& SolverIntro::get_Hy() const { return Hy_; }

const std::vector<Eigen::VectorXd>& SolverIntro::get_kz() const { return kz_; }

const std::vector<Eigen::MatrixXd>& SolverIntro::get_Kz() const { return Kz_; }

const std::vector<Eigen::VectorXd>& SolverIntro::get_ks() const { return ks_; }

const std::vector<Eigen::MatrixXd>& SolverIntro::get_Ks() const { return Ks_; }

void SolverIntro::set_equality_solver(const EqualitySolverType type) {
  eq_solver_ = type;
}

void SolverIntro::set_th_feas(const double th_feas) { th_feas_ = th_feas; }

void SolverIntro::set_rho(const double rho) {
  if (0. >= rho || rho > 1.) {
    throw_pretty("Invalid argument: " << "rho value should between 0 and 1.");
  }
  rho_ = rho;
}

void SolverIntro::set_zero_upsilon(const bool zero_upsilon) {
  zero_upsilon_ = zero_upsilon;
}

}  // namespace crocoddyl
