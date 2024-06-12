///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2024, Heriot-Watt University, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/solvers/intro.hpp"

#include "crocoddyl/core/utils/stop-watch.hpp"

namespace crocoddyl {

SolverIntro::SolverIntro(std::shared_ptr<ShootingProblem> problem,
                         const DynamicsSolverType dyn_solver,
                         const EqualitySolverType eq_solver)
    : SolverFDDP(problem, dyn_solver), eq_solver_(eq_solver) {
  allocateData();
}

SolverIntro::~SolverIntro() {}

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
    KQuu_2Qxu_[t].conservativeResize(ndx, nu);
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

void SolverIntro::calcDir() {
  START_PROFILER("SolverIntro::calcDir");
  SolverFDDP::calcDir();
  switch (eq_solver_) {
    case LuNull:
      calcLuNullDir();
      break;
    case QrNull:
      calcQrNullDir();
      break;
    default:
      break;
  }
  STOP_PROFILER("SolverIntro::calcDir");
}

void SolverIntro::computePolicy(const std::size_t t) {
  START_PROFILER("SolverIntro::computePolicy");
  switch (eq_solver_) {
    case LuNull:
    case QrNull:
      computeNullPolicy(t);
      break;
    case Schur:
      computeSchurPolicy(t);
      break;
  }
  STOP_PROFILER("SolverIntro::computePolicy");
}

void SolverIntro::computeValueFunction(
    const std::size_t t, const std::shared_ptr<ActionModelAbstract>& model) {
  START_PROFILER("SolverIntro::computeValueFunction");
  assert_pretty(t < problem_->get_T(),
                "Invalid argument: t should be between 0 and " +
                    std::to_string(problem_->get_T()););
  const std::size_t nu = model->get_nu();
  Vx_[t] = Qx_[t];
  Vxx_[t] = Qxx_[t];
  if (nu != 0) {
    START_PROFILER("SolverIntro::Vx");
    Quuk_[t].noalias() = Quu_[t] * k_[t];
    Quuk_[t] -= Qu_[t];
    Vx_[t].noalias() -= Qxu_[t] * k_[t];
    Vx_[t].noalias() += K_[t].transpose() * Quuk_[t];
    Quuk_[t] += Qu_[t];
    STOP_PROFILER("SolverIntro::Vx");
    START_PROFILER("SolverIntro::Vxx");
    KQuu_2Qxu_[t].noalias() = K_[t].transpose() * Quu_[t];
    KQuu_2Qxu_[t].noalias() -= 2 * Qxu_[t];
    Vxx_[t].noalias() += KQuu_2Qxu_[t] * K_[t];
    STOP_PROFILER("SolverIntro::Vxx");
  }
  Vxx_tmp_ = 0.5 * (Vxx_[t] + Vxx_[t].transpose());
  Vxx_[t] = Vxx_tmp_;
  Vxx_f_[t].noalias() = Vxx_[t] * fs_[t];
  STOP_PROFILER("SolverIntro::computeValueFunction");
}

void SolverIntro::allocateData() {
  const std::size_t ndx = problem_->get_ndx();
  const std::size_t T = problem_->get_T();
  Hu_rank_.resize(T);
  KQuu_2Qxu_.resize(T);
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
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    const std::size_t nh = model->get_nh();
    Hu_rank_[t] = nh;
    KQuu_2Qxu_[t] = MatrixXdRowMajor::Zero(ndx, nu);
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

void SolverIntro::calcLuNullDir() {
  START_PROFILER("SolverIntro::calcLuNullDir");
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
      problem_->get_runningDatas();
#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(problem_->get_nthreads())
#endif
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& model = models[t];
    const std::shared_ptr<crocoddyl::ActionDataAbstract>& data = datas[t];
    if (model->get_nu() > 0 && model->get_nh() > 0) {
      Hu_lu_[t].compute(data->Hu);
      YZ_[t] << Hu_lu_[t].matrixLU().transpose(), Hu_lu_[t].kernel();
      Hu_rank_[t] = Hu_lu_[t].rank();
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
  STOP_PROFILER("SolverIntro::calcLuNullDir");
}

void SolverIntro::calcQrNullDir() {
  START_PROFILER("SolverIntro::calcQrNullDir");
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract> >& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract> >& datas =
      problem_->get_runningDatas();
#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(problem_->get_nthreads())
#endif
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<crocoddyl::ActionModelAbstract>& model = models[t];
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
  STOP_PROFILER("SolverIntro::calcQrNullDir");
}

void SolverIntro::computeNullPolicy(const std::size_t t) {
  START_PROFILER("SolverIntro::computeNullPolicy");
  const std::shared_ptr<ActionModelAbstract>& model =
      problem_->get_runningModels()[t];
  const std::shared_ptr<ActionDataAbstract>& data =
      problem_->get_runningDatas()[t];
  const std::size_t nu = model->get_nu();
  const std::size_t nh = model->get_nh();
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
    Qz_[t].noalias() = Z.transpose() * Qu_[t];
    Qxz_[t].noalias() = Qxu_[t] * Z;
    Eigen::Transpose<Eigen::MatrixXd> Qzx = Qxz_[t].transpose();
    Eigen::Transpose<Eigen::MatrixXd> QzzinvQzu = Quz_[t].transpose();
    Qzz_llt_[t].solveInPlace(Qz_[t]);
    Qzz_llt_[t].solveInPlace(Qzx);
    Qzz_llt_[t].solveInPlace(QzzinvQzu);
    Qz_[t].noalias() -= QzzinvQzu * kz_[t];
    Qzx.noalias() -= QzzinvQzu * Kz_[t];
    k_[t] = kz_[t];
    K_[t] = Kz_[t];
    k_[t].noalias() += Z * Qz_[t];
    K_[t].noalias() += Z * Qzx;
  } else if (nu > 0) {
    SolverFDDP::computePolicy(t);
  }
  STOP_PROFILER("SolverIntro::computeNullPolicy");
}

void SolverIntro::computeSchurPolicy(const std::size_t t) {
  START_PROFILER("SolverIntro::computeSchurPolicy");
  const std::shared_ptr<ActionModelAbstract>& model =
      problem_->get_runningModels()[t];
  const std::shared_ptr<ActionDataAbstract>& data =
      problem_->get_runningDatas()[t];
  const std::size_t nu = model->get_nu();
  const std::size_t nh = model->get_nh();
  if (nu > 0) {
    SolverFDDP::computePolicy(t);
  }
  if (nu > 0 && nh > 0) {
    START_PROFILER("SolverIntro::Qzz_cholesky");
    QuuinvHuT_[t] = data->Hu.transpose();
    Quu_llt_[t].solveInPlace(QuuinvHuT_[t]);
    Qzz_[t].noalias() = data->Hu * QuuinvHuT_[t];
    Qzz_llt_[t].compute(Qzz_[t]);
    STOP_PROFILER("SolverIntro::Qzz_cholesky");
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
  STOP_PROFILER("SolverIntro::computeSchurPolicy");
}

EqualitySolverType SolverIntro::get_equality_solver() const {
  return eq_solver_;
}

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

}  // namespace crocoddyl
