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

SolverIntro::SolverIntro(boost::shared_ptr<ShootingProblem> problem)
    : SolverDDP(problem), eq_solver_(LuNull), rho_(0.3), dPhi_(0.), hfeas_try_(0.), upsilon_(0.) {
  reg_incfactor_ = 1e6;

  const std::size_t T = problem_->get_T();
  Hu_rank_.resize(T);
  QuuK_tmp_.resize(T);
  ZQzzinvQzuI_.resize(T);
  YZ_.resize(T);
  HuY_.resize(T);
  Qzz_.resize(T);
  Quz_.resize(T);
  k_z_.resize(T);
  K_z_.resize(T);
  k_hat_.resize(T);
  K_hat_.resize(T);
  QuuinvHuT_.resize(T);
  Qzz_llt_.resize(T);
  Hu_lu_.resize(T);
  Hu_qr_.resize(T);
  HuY_lu_.resize(T);

  const std::size_t ndx = problem_->get_ndx();
  const std::vector<boost::shared_ptr<ActionModelAbstract> >& models = problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const boost::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    const std::size_t nh = model->get_nh();
    Hu_rank_[t] = nh;
    QuuK_tmp_[t] = Eigen::MatrixXd::Zero(nu, ndx);
    ZQzzinvQzuI_[t] = Eigen::MatrixXd::Zero(nu, nu);
    YZ_[t] = Eigen::MatrixXd::Zero(nu, nu);
    HuY_[t] = Eigen::MatrixXd::Zero(nh, nh);
    Qzz_[t] = Eigen::MatrixXd::Zero(nh, nh);
    Quz_[t] = Eigen::MatrixXd::Zero(nu, nh);
    k_z_[t] = Eigen::VectorXd::Zero(nu);
    K_z_[t] = Eigen::MatrixXd::Zero(nu, ndx);
    k_hat_[t] = Eigen::VectorXd::Zero(nh);
    K_hat_[t] = Eigen::MatrixXd::Zero(nh, ndx);
    QuuinvHuT_[t] = Eigen::MatrixXd::Zero(nu, nh);
    Qzz_llt_[t] = Eigen::LLT<Eigen::MatrixXd>(nh);
    Hu_lu_[t] = Eigen::FullPivLU<Eigen::MatrixXd>(nh, nu);
    Hu_qr_[t] = Eigen::ColPivHouseholderQR<Eigen::MatrixXd>(nu, nh);
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

double SolverIntro::stoppingCriteria() {
  stop_ = std::max(hfeas_, abs(d_[0] + 0.5 * d_[1]));
  return stop_;
}

void SolverIntro::resizeData() {
  START_PROFILER("SolverIntro::resizeData");
  SolverDDP::resizeData();

  const std::size_t T = problem_->get_T();
  const std::size_t ndx = problem_->get_ndx();
  const std::vector<boost::shared_ptr<ActionModelAbstract> >& models = problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const boost::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    const std::size_t nh = model->get_nh();
    QuuK_tmp_[t].conservativeResize(nu, ndx);
    ZQzzinvQzuI_[t].conservativeResize(nu, nu);
    YZ_[t].conservativeResize(nu, nu);
    HuY_[t].conservativeResize(nh, nh);
    Qzz_[t].conservativeResize(nh, nh);
    Quz_[t].conservativeResize(nu, nh);
    k_z_[t].conservativeResize(nu);
    K_z_[t].conservativeResize(nu, ndx);
    k_hat_[t].conservativeResize(nh);
    K_hat_[t].conservativeResize(nh, ndx);
    QuuinvHuT_[t].conservativeResize(nu, nh);
  }
  STOP_PROFILER("SolverIntro::resizeData");
}

double SolverIntro::calcDiff() {
  START_PROFILER("SolverIntro::calcDiff");
  SolverDDP::calcDiff();
  const std::size_t T = problem_->get_T();
  const std::vector<boost::shared_ptr<ActionModelAbstract> >& models = problem_->get_runningModels();
  const std::vector<boost::shared_ptr<ActionDataAbstract> >& datas = problem_->get_runningDatas();
  switch (eq_solver_) {
    case LuNull:
#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(problem_->get_nthreads())
#endif
      for (std::size_t t = 0; t < T; ++t) {
        const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = models[t];
        const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = datas[t];
        if (model->get_nu() > 0 && model->get_nh() > 0) {
          Hu_lu_[t].compute(data->Hu);
          YZ_[t] << Hu_lu_[t].matrixLU().transpose(), Hu_lu_[t].kernel();
          Hu_rank_[t] = Hu_lu_[t].rank();
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
      break;
    case QrNull:
#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(problem_->get_nthreads())
#endif
      for (std::size_t t = 0; t < T; ++t) {
        const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = models[t];
        const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = datas[t];
        if (model->get_nu() > 0 && model->get_nh() > 0) {
          Hu_qr_[t].compute(data->Hu.transpose());
          YZ_[t] = Hu_qr_[t].householderQ();
          Hu_rank_[t] = Hu_qr_[t].rank();
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
      break;
    case Schur:
      break;
  }

  STOP_PROFILER("SolverIntro::calcDiff");
  return cost_;
}

void SolverIntro::computeValueFunction(const std::size_t t, const boost::shared_ptr<ActionModelAbstract>& model) {
  const std::size_t nu = model->get_nu();
  Vx_[t] = Qx_[t];
  Vxx_[t] = Qxx_[t];
  if (nu != 0) {
    START_PROFILER("SolverIntro::Vx");
    Quuk_[t].noalias() = Quu_[t] * k_[t];
    Vx_[t].noalias() -= K_[t].transpose() * Qu_[t];
    Vx_[t].noalias() -= Qxu_[t] * k_[t];
    Vx_[t].noalias() += K_[t].transpose() * Quuk_[t];
    STOP_PROFILER("SolverIntro::Vx");
    START_PROFILER("SolverIntro::Vxx");
    QuuK_tmp_[t].noalias() = Quu_[t] * K_[t];
    Vxx_[t].noalias() -= 2 * Qxu_[t] * K_[t];
    Vxx_[t].noalias() += K_[t].transpose() * QuuK_tmp_[t];
    STOP_PROFILER("SolverIntro::Vxx");
  }
  Vxx_tmp_ = 0.5 * (Vxx_[t] + Vxx_[t].transpose());
  Vxx_[t] = Vxx_tmp_;
  // Compute and store the Vx gradient at end of the interval (rollout state)
  if (!is_feasible_) {
    Vx_[t].noalias() += Vxx_[t] * fs_[t];
  }
}

void SolverIntro::computeGains(const std::size_t t) {
  START_PROFILER("SolverIntro::computeGains");
  const boost::shared_ptr<crocoddyl::ActionModelAbstract>& model = problem_->get_runningModels()[t];
  const boost::shared_ptr<crocoddyl::ActionDataAbstract>& data = problem_->get_runningDatas()[t];

  const std::size_t nu = model->get_nu();
  const std::size_t nh = model->get_nh();
  switch (eq_solver_) {
    case LuNull:
    case QrNull:
      if (nu > 0 && nh > 0) {
        const std::size_t rank = Hu_rank_[t];
        const std::size_t nullity = data->Hu.cols() - rank;
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
        ZQzzinvQzuI_[t].noalias() = Z * Qzu;
        ZQzzinvQzuI_[t].diagonal().array() -= 1.;
        k_[t].noalias() = ZQzzinvQzuI_[t] * k_z_[t];
        K_[t].noalias() = ZQzzinvQzuI_[t] * K_z_[t];

        Eigen::VectorBlock<Eigen::VectorXd> k_z = k_z_[t].tail(nullity);
        Eigen::Block<Eigen::MatrixXd, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> K_z =
            K_z_[t].bottomRows(nullity);
        k_z.noalias() = Z.transpose() * Qu_[t];
        Qzz_llt_[t].solveInPlace(k_z);
        K_z.transpose().noalias() = Qxu_[t] * Z;
        Qzz_llt_[t].solveInPlace(K_z);
        k_[t].noalias() += Z * k_z;
        K_[t].noalias() += Z * K_z;
      } else {
        SolverDDP::computeGains(t);
      }
      break;
    case Schur:
      SolverDDP::computeGains(t);
      if (nu > 0 && nh > 0) {
        QuuinvHuT_[t] = data->Hu.transpose();
        Quu_llt_[t].solveInPlace(QuuinvHuT_[t]);
        Qzz_[t].noalias() = data->Hu * QuuinvHuT_[t];
        Qzz_llt_[t].compute(Qzz_[t]);
        const Eigen::ComputationInfo& info = Qzz_llt_[t].info();
        if (info != Eigen::Success) {
          throw_pretty("backward error");
        }
        Eigen::Transpose<Eigen::MatrixXd> HuQuuinv = QuuinvHuT_[t].transpose();
        Qzz_llt_[t].solveInPlace(HuQuuinv);
        k_hat_[t] = data->h;
        k_hat_[t].noalias() -= data->Hu * k_[t];
        K_hat_[t] = data->Hx;
        K_hat_[t].noalias() -= data->Hu * K_[t];
        k_[t].noalias() += QuuinvHuT_[t] * k_hat_[t];
        K_[t] += QuuinvHuT_[t] * K_hat_[t];
      }
      break;
  }
  STOP_PROFILER("SolverIntro::computeGains");
}

EqualitySolverType SolverIntro::get_equality_solver() const { return eq_solver_; }

double SolverIntro::get_rho() const { return rho_; }

double SolverIntro::get_dPhi() const { return dPhi_; }

double SolverIntro::get_dPhiexp() const { return dPhiexp_; }

double SolverIntro::get_upsilon() const { return upsilon_; }

void SolverIntro::set_equality_solver(const EqualitySolverType type) { eq_solver_ = type; }

void SolverIntro::set_rho(const double rho) {
  if (0. >= rho || rho > 1.) {
    throw_pretty("Invalid argument: "
                 << "rho value should between 0 and 1.");
  }
  rho_ = rho;
}

}  // namespace crocoddyl
