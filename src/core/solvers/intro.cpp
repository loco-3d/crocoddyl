///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "crocoddyl/core/solvers/intro.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/utils/stop-watch.hpp"

namespace crocoddyl {

SolverIntro::SolverIntro(boost::shared_ptr<ShootingProblem> problem, const bool reduced)
    : SolverDDP(problem), reduced_(reduced), rho_(0.3), dPhi_(0.), hfeas_try_(0.), upsilon_(0.) {
  reg_incfactor_ = 1e6;

  const std::size_t T = problem_->get_T();
  YZ_.resize(T);
  HuY_.resize(T);
  Qz_.resize(T);
  Qzz_.resize(T);
  Quz_.resize(T);
  Qxz_.resize(T);
  k_z_.resize(T);
  K_z_.resize(T);
  k_hat_.resize(T);
  K_hat_.resize(T);
  QuuinvHuT_.resize(T);
  Qzz_llt_.resize(T);
  Hu_lu_.resize(T);
  HuY_lu_.resize(T);

  const std::size_t ndx = problem_->get_ndx();
  const std::size_t nu = problem_->get_nu_max();
  QuuK_tmp_ = Eigen::MatrixXd::Zero(nu, ndx);
  ZQzzinvQzuI_ = Eigen::MatrixXd::Zero(nu, nu);
  const std::vector<boost::shared_ptr<ActionModelAbstract> >& models = problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const boost::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nh = model->get_nh();

    YZ_[t] = Eigen::MatrixXd::Zero(nu, nu);
    HuY_[t] = Eigen::MatrixXd::Zero(nh, nh);
    Qz_[t] = Eigen::VectorXd::Zero(nh);
    Qzz_[t] = Eigen::MatrixXd::Zero(nh, nh);
    Quz_[t] = Eigen::MatrixXd::Zero(nu, nh);
    Qxz_[t] = Eigen::MatrixXd::Zero(ndx, nh);
    k_z_[t] = Eigen::VectorXd::Zero(nu);
    K_z_[t] = Eigen::MatrixXd::Zero(nu, ndx);
    k_hat_[t] = Eigen::VectorXd::Zero(nh);
    K_hat_[t] = Eigen::MatrixXd::Zero(nh, ndx);
    QuuinvHuT_[t] = Eigen::MatrixXd::Zero(nu, nh);
    Qzz_llt_[t] = Eigen::LLT<Eigen::MatrixXd>(nh);
    // Hu_lu_[t] = Eigen::ColPivHouseholderQR<Eigen::MatrixXd>(nu, nh);
    Hu_lu_[t] = Eigen::FullPivLU<Eigen::MatrixXd>(nh, nu);
    HuY_lu_[t] = Eigen::PartialPivLU<Eigen::MatrixXd>(nh);
  }
}

SolverIntro::~SolverIntro() {}

bool SolverIntro::solve(const std::vector<Eigen::VectorXd>& init_xs, const std::vector<Eigen::VectorXd>& init_us,
                        const std::size_t maxiter, const bool is_feasible, const double reginit) {
  START_PROFILER("SolverIntro::solve");
  xs_try_[0] = problem_->get_x0();  // it is needed in case that init_xs[0] is infeasible
  setCandidate(init_xs, init_us, is_feasible);

  if (std::isnan(reginit)) {
    xreg_ = reg_min_;
    ureg_ = reg_min_;
  } else {
    xreg_ = reginit;
    ureg_ = reginit;
  }
  was_feasible_ = false;
  upsilon_ = 0.;

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
      upsilon_ = std::max(upsilon_, (d_[0] + .5 * d_[1]) / ((1 - rho_) * hfeas_));
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
      if (abs(d_[0]) < th_grad_ || !is_feasible_ || dPhi_ > th_acceptstep_ * dPhiexp_) {
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
      if (xreg_ == reg_max_) {
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

void SolverIntro::backwardPass() {
  START_PROFILER("SolverIntro::backwardPass");
  const boost::shared_ptr<ActionDataAbstract>& d_T = problem_->get_terminalData();
  Vxx_.back() = d_T->Lxx;
  Vx_.back() = d_T->Lx;

  if (!std::isnan(xreg_)) {
    Vxx_.back().diagonal().array() += xreg_;
  }

  if (!is_feasible_) {
    Vx_.back().noalias() += Vxx_.back() * fs_.back();
  }
  const std::vector<boost::shared_ptr<ActionModelAbstract> >& models = problem_->get_runningModels();
  const std::vector<boost::shared_ptr<ActionDataAbstract> >& datas = problem_->get_runningDatas();
  for (int t = static_cast<int>(problem_->get_T()) - 1; t >= 0; --t) {
    const boost::shared_ptr<ActionModelAbstract>& m = models[t];
    const boost::shared_ptr<ActionDataAbstract>& d = datas[t];
    const Eigen::MatrixXd& Vxx_p = Vxx_[t + 1];
    const Eigen::VectorXd& Vx_p = Vx_[t + 1];
    const std::size_t nu = m->get_nu();

    FxTVxx_p_.noalias() = d->Fx.transpose() * Vxx_p;
    START_PROFILER("SolverIntro::Qx");
    Qx_[t] = d->Lx;
    Qx_[t].noalias() += d->Fx.transpose() * Vx_p;
    STOP_PROFILER("SolverIntro::Qx");
    START_PROFILER("SolverIntro::Qxx");
    Qxx_[t] = d->Lxx;
    Qxx_[t].noalias() += FxTVxx_p_ * d->Fx;
    STOP_PROFILER("SolverIntro::Qxx");
    if (nu != 0) {
      FuTVxx_p_[t].topRows(nu).noalias() = d->Fu.transpose() * Vxx_p;
      START_PROFILER("SolverIntro::Qu");
      Qu_[t].head(nu) = d->Lu;
      Qu_[t].head(nu).noalias() += d->Fu.transpose() * Vx_p;
      STOP_PROFILER("SolverIntro::Qu");
      START_PROFILER("SolverIntro::Quu");
      Quu_[t].topLeftCorner(nu, nu) = d->Luu;
      Quu_[t].topLeftCorner(nu, nu).noalias() += FuTVxx_p_[t].topRows(nu) * d->Fu;
      STOP_PROFILER("SolverIntro::Quu");
      START_PROFILER("SolverIntro::Qxu");
      Qxu_[t].leftCols(nu) = d->Lxu;
      Qxu_[t].leftCols(nu).noalias() += FxTVxx_p_ * d->Fu;
      STOP_PROFILER("SolverIntro::Qxu");
      if (!std::isnan(ureg_)) {
        Quu_[t].diagonal().head(nu).array() += ureg_;
      }
    }

    computeGains(t);

    Vx_[t] = Qx_[t];
    Vxx_[t] = Qxx_[t];
    if (nu != 0) {
      START_PROFILER("SolverIntro::Vx");
      Quuk_[t].head(nu).noalias() = Quu_[t].topLeftCorner(nu, nu) * k_[t].head(nu);
      Vx_[t].noalias() -= K_[t].topRows(nu).transpose() * Qu_[t].head(nu);
      Vx_[t].noalias() -= Qxu_[t].leftCols(nu) * k_[t].head(nu);
      Vx_[t].noalias() += K_[t].topRows(nu).transpose() * Quuk_[t].head(nu);
      STOP_PROFILER("SolverIntro::Vx");
      START_PROFILER("SolverIntro::Vxx");
      QuuK_tmp_.topRows(nu).noalias() = Quu_[t].topLeftCorner(nu, nu) * K_[t].topRows(nu);
      Vxx_[t].noalias() -= 2 * Qxu_[t].leftCols(nu) * K_[t].topRows(nu);
      Vxx_[t].noalias() += K_[t].topRows(nu).transpose() * QuuK_tmp_.topRows(nu);
      STOP_PROFILER("SolverIntro::Vxx");
    }
    Vxx_tmp_ = 0.5 * (Vxx_[t] + Vxx_[t].transpose());
    Vxx_[t] = Vxx_tmp_;

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
  STOP_PROFILER("SolverIntro::backwardPass");
}

double SolverIntro::stoppingCriteria() {
  stop_ = std::max(hfeas_, abs(d_[0] + 0.5 * d_[1]));
  return stop_;
}

double SolverIntro::calcDiff() {
  START_PROFILER("SolverIntro::calcDiff");
  SolverDDP::calcDiff();
  if (reduced_) {
    const std::size_t T = problem_->get_T();
#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(problem_->get_nthreads())
#endif
    for (std::size_t t = 0; t < T; ++t) {
      const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = problem_->get_runningModels()[t];
      const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = problem_->get_runningDatas()[t];
      if (model->get_nu() > 0 && model->get_nh() > 0) {
        // Hu_lu_[t].compute(data->Hu.transpose());
        // YZ_[t] = Hu_lu_[t].householderQ();
        Hu_lu_[t].compute(data->Hu);
        YZ_[t] << Hu_lu_[t].image(data->Hu), Hu_lu_[t].kernel();
        const Eigen::Block<Eigen::MatrixXd, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Y =
            YZ_[t].leftCols(Hu_lu_[t].rank());
        HuY_[t].noalias() = -data->Hu * Y;
        HuY_lu_[t].compute(HuY_[t]);
        const Eigen::Inverse<Eigen::PartialPivLU<Eigen::MatrixXd> > HuYinv = HuY_lu_[t].inverse();
        k_hat_[t].noalias() = HuYinv * data->h;
        K_hat_[t].noalias() = HuYinv * data->Hx;
        k_z_[t].noalias() = Y * k_hat_[t];
        K_z_[t].noalias() = Y * K_hat_[t];
      }
    }
  }
  STOP_PROFILER("SolverIntro::calcDiff");
  return cost_;
}

void SolverIntro::computeGains(const std::size_t t) {
  START_PROFILER("SolverIntro::computeGains");
  const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = problem_->get_runningModels()[t];
  const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = problem_->get_runningDatas()[t];

  const std::size_t nu = model->get_nu();
  const std::size_t nh = model->get_nh();
  if (reduced_) {
    if (nu > 0 && nh > 0) {
      const std::size_t nullity = data->Hu.cols() - Hu_lu_[t].rank();
      const Eigen::Block<Eigen::MatrixXd, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> Z =
          YZ_[t].rightCols(nullity);
      Quz_[t].noalias() = Quu_[t] * Z;
      Qzz_[t].noalias() = Z.transpose() * Quz_[t];
      Qzz_llt_[t].compute(Qzz_[t]);
      const Eigen::ComputationInfo& info = Qzz_llt_[t].info();
      if (info != Eigen::Success) {
        throw_pretty("backward error");
      }
      Eigen::Transpose<Eigen::MatrixXd> Qzu = Quz_[t].transpose();
      Qzz_llt_[t].solveInPlace(Qzu);
      ZQzzinvQzuI_.noalias() = Z * Qzu;
      ZQzzinvQzuI_.diagonal().array() -= 1.;

      k_[t].noalias() = ZQzzinvQzuI_ * k_z_[t];
      K_[t].noalias() = ZQzzinvQzuI_ * K_z_[t];

      Eigen::VectorBlock<Eigen::VectorXd> k_z = k_z_[t].tail(nullity);
      Eigen::Block<Eigen::MatrixXd, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> K_z = K_z_[t].bottomRows(nullity);
      Qz_[t].noalias() = Z.transpose() * Qu_[t];
      Qxz_[t].noalias() = Qxu_[t] * Z;
      k_z = Qz_[t];
      Qzz_llt_[t].solveInPlace(k_z);
      K_z = Qxz_[t].transpose();
      Qzz_llt_[t].solveInPlace(K_z);
      k_[t].noalias() += Z * k_z;
      K_[t].noalias() += Z * K_z;
    } else {
      SolverDDP::computeGains(t);
    }
  } else {
    SolverDDP::computeGains(t);
    if (nu > 0 && nh > 0) {
      QuuinvHuT_[t].topRows(nu) = data->Hu.transpose();

      Eigen::Block<Eigen::MatrixXd, Eigen::Dynamic> QuuinvHuT = QuuinvHuT_[t].topRows(nu);
      Quu_llt_[t].solveInPlace(QuuinvHuT);
      Qzz_[t].noalias() = data->Hu * QuuinvHuT;
      Qzz_llt_[t].compute(Qzz_[t]);
      const Eigen::ComputationInfo& info = Qzz_llt_[t].info();
      if (info != Eigen::Success) {
        throw_pretty("backward error");
      }
      Eigen::Transpose<Eigen::Block<Eigen::MatrixXd, Eigen::Dynamic> > HuQuuinv =
          QuuinvHuT_[t].topRows(nu).transpose();
      Qzz_llt_[t].solveInPlace(HuQuuinv);
      k_hat_[t] = data->h;
      k_hat_[t].noalias() -= data->Hu * k_[t].head(nu);
      K_hat_[t] = data->Hx;
      K_hat_[t].noalias() -= data->Hu * K_[t].topRows(nu);
      k_[t].head(nu).noalias() += QuuinvHuT_[t].topRows(nu) * k_hat_[t];
      K_[t].topRows(nu) += QuuinvHuT_[t].topRows(nu) * K_hat_[t];
    }
  }
  STOP_PROFILER("SolverIntro::computeGains");
}

double SolverIntro::get_rho() const { return rho_; }

double SolverIntro::get_dPhi() const { return dPhi_; }

double SolverIntro::get_dPhiexp() const { return dPhiexp_; }

double SolverIntro::get_upsilon() const { return upsilon_; }

void SolverIntro::set_rho(const double rho) {
  if (0. >= rho || rho > 1.) {
    throw_pretty("Invalid argument: "
                 << "rho value should between 0 and 1.");
  }
  rho_ = rho;
}

}  // namespace crocoddyl
