///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2026, Heriot-Watt University, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

namespace crocoddyl {

template <typename Scalar>
SolverIntroTpl<Scalar>::SolverIntroTpl(std::shared_ptr<ShootingProblem> problem,
                                       const DynamicsSolverType dyn_solver,
                                       const EqualitySolverType eq_solver,
                                       const EqualitySolverType term_solver)
    : SolverIntroTpl(std::static_pointer_cast<ProblemAbstract>(problem),
                     dyn_solver, eq_solver, term_solver, AStateNone) {}

template <typename Scalar>
SolverIntroTpl<Scalar>::SolverIntroTpl(
    std::shared_ptr<ProblemAbstract> problem,
    const DynamicsSolverType dyn_solver, const EqualitySolverType eq_solver,
    const EqualitySolverType term_solver,
    const ArrivalStateSolverType astate_solver)
    : SolverFDDP(problem, dyn_solver, term_solver, astate_solver),
      eq_solver_(eq_solver) {
  allocateData();
}

template <typename Scalar>
void SolverIntroTpl<Scalar>::resizeRunningData() {
  START_PROFILER("SolverIntro::resizeRunningData");
  SolverFDDP::resizeRunningData();
  const std::size_t T = problem_->get_T();
  const std::size_t ndx = problem_->get_ndx();
  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    const std::size_t nh = model->get_nh();
    assert_pretty(nh <= nu,
                  "Invalid argument: the number of equality constraints must "
                  "not exceed the control dimension.");
    const std::size_t nz = eq_solver_ == Schur ? nh : nu - nh;
    KQuu_2Qxu_[t].conservativeResize(ndx, nu);
    YZ_[t].conservativeResize(nu, nu);
    Hy_[t].conservativeResize(nh, nh);
    Qz_[t].conservativeResize(nz);
    Qzz_[t].conservativeResize(nz, nz);
    Qxz_[t].conservativeResize(ndx, nz);
    Quz_[t].conservativeResize(nu, nz);
    QzzinvQzu_[t].conservativeResize(nz, nu);
    Qpz_[t].conservativeResize(model->get_np(), nz);
    Qzx_[t].conservativeResize(nz, ndx);
    Pn_[t].conservativeResize(nz, model->get_np());
    QuuP_2Qpu_[t].conservativeResize(nu, model->get_np());
    kz_[t].conservativeResize(nu);
    Kz_[t].conservativeResize(nu, ndx);
    ks_[t].conservativeResize(nh);
    Ks_[t].conservativeResize(nh, ndx);
    Ps_[t].conservativeResize(nh, model->get_np());
    Pz_[t].conservativeResize(nu, model->get_np());
    QuuinvHuT_[t].conservativeResize(nu, nh);
    Qzz_llt_[t] = Eigen::LLT<MatrixXs>(nz);
    Hu_lu_[t] = Eigen::FullPivLU<MatrixXs>(nh, nu);
    Hu_qr_[t] = Eigen::ColPivHouseholderQR<MatrixXs>(nu, nh);
    Hy_lu_[t] = Eigen::PartialPivLU<MatrixXs>(nh);
  }
  STOP_PROFILER("SolverIntro::resizeRunningData");
}

template <typename Scalar>
void SolverIntroTpl<Scalar>::resizeTerminalData() {
  START_PROFILER("SolverIntro::resizeTerminalData");
  SolverFDDP::resizeTerminalData();
  const std::size_t T = problem_->get_T();
  const std::size_t nh_T = problem_->get_terminalModel()->get_nh_T();
  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    const std::size_t nh = model->get_nh();
    Kcs_[t].conservativeResize(nh, nh_T);
    QuuKc_Quc_[t].conservativeResize(nu, nh_T);
    const std::size_t nz = eq_solver_ == Schur ? nh : nu - nh;
    Qzc_[t].conservativeResize(nz, nh_T);
  }
  STOP_PROFILER("SolverIntro::resizeTerminalData");
}

template <typename Scalar>
void SolverIntroTpl<Scalar>::calcDir() {
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

template <typename Scalar>
void SolverIntroTpl<Scalar>::computeDirection(const bool recalc) {
  START_PROFILER("SolverIntro::computeDirection");
  if (recalc) {
    calcDir();
  }
  SolverFDDP::computeDirection(false);
  STOP_PROFILER("SolverIntro::computeDirection");
}

template <typename Scalar>
void SolverIntroTpl<Scalar>::computePolicy(const std::size_t t) {
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

template <typename Scalar>
void SolverIntroTpl<Scalar>::computeBatchPolicy(const std::size_t t) {
  START_PROFILER("SolverIntro::computeBatchPolicy");
  switch (eq_solver_) {
    case LuNull:
    case QrNull:
      computeNullBatchPolicy(t);
      break;
    case Schur:
      computeSchurBatchPolicy(t);
      break;
  }
  STOP_PROFILER("SolverIntro::computeBatchPolicy");
}

template <typename Scalar>
void SolverIntroTpl<Scalar>::computeValueFunction(
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
    Vx_[t].noalias() += K_[t].transpose() * Quuk_[t];
    Vx_[t].noalias() -= Qxu_[t] * k_[t];
    Quuk_[t] += Qu_[t];
    STOP_PROFILER("SolverIntro::Vx");
    START_PROFILER("SolverIntro::Vxx");
    KQuu_2Qxu_[t].noalias() = K_[t].transpose() * Quu_[t];
    KQuu_2Qxu_[t].noalias() -= Scalar(2.) * Qxu_[t];
    Vxx_[t].noalias() += KQuu_2Qxu_[t] * K_[t];
    STOP_PROFILER("SolverIntro::Vxx");
  }
  Vxx_tmp_ = Scalar(0.5) * (Vxx_[t] + Vxx_[t].transpose());
  Vxx_[t] = Vxx_tmp_;
  Vxx_f_[t].noalias() = Vxx_[t] * fs_[t];
  STOP_PROFILER("SolverIntro::computeValueFunction");
}

template <typename Scalar>
void SolverIntroTpl<Scalar>::computeBatchValueFunction(const std::size_t t) {
  START_PROFILER("SolverIntro::computeBatchValueFunction");
  SolverFDDP::computeBatchValueFunction(t);
  QuuKc_Quc_[t].noalias() = Quu_[t] * Kc_[t];
  QuuKc_Quc_[t] -= Quc_[t];
  Vxc_[t].noalias() += K_[t].transpose() * QuuKc_Quc_[t];
  if (this->n_phases_ > 0) {
    this->Vpc_[t].noalias() += P_[t].transpose() * QuuKc_Quc_[t];
  }
  STOP_PROFILER("SolverIntro::computeBatchValueFunction");
}

template <typename Scalar>
void SolverIntroTpl<Scalar>::allocateData() {
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
  QzzinvQzu_.resize(T);
  Qpz_.resize(T);
  Qzx_.resize(T);
  Pn_.resize(T);
  QuuP_2Qpu_.resize(T);
  kz_.resize(T);
  Kz_.resize(T);
  ks_.resize(T);
  Ks_.resize(T);
  Ps_.resize(T);
  Pz_.resize(T);
  QuuinvHuT_.resize(T);
  Qzz_llt_.resize(T);
  Hu_lu_.resize(T);
  Hu_qr_.resize(T);
  Hy_lu_.resize(T);
  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
      problem_->get_runningModels();
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    const std::size_t nh = model->get_nh();
    const std::size_t np = model->get_np();
    assert_pretty(nh <= nu,
                  "Invalid argument: the number of equality constraints must "
                  "not exceed the control dimension.");
    const std::size_t nz = eq_solver_ == Schur ? nh : nu - nh;
    Hu_rank_[t] = nh;
    KQuu_2Qxu_[t] = MatrixXsRowMajor::Zero(ndx, nu);
    YZ_[t] = MatrixXs::Zero(nu, nu);
    Hy_[t] = MatrixXs::Zero(nh, nh);
    Qz_[t] = VectorXs::Zero(nz);
    Qzz_[t] = MatrixXs::Zero(nz, nz);
    Qxz_[t] = MatrixXs::Zero(ndx, nz);
    Quz_[t] = MatrixXs::Zero(nu, nz);
    QzzinvQzu_[t] = MatrixXs::Zero(nz, nu);
    Qpz_[t] = MatrixXs::Zero(np, nz);
    Qzx_[t] = MatrixXs::Zero(nz, ndx);
    Pn_[t] = MatrixXs::Zero(nz, np);
    QuuP_2Qpu_[t] = MatrixXs::Zero(nu, np);
    kz_[t] = VectorXs::Zero(nu);
    Kz_[t] = MatrixXs::Zero(nu, ndx);
    ks_[t] = VectorXs::Zero(nh);
    Ks_[t] = MatrixXs::Zero(nh, ndx);
    Ps_[t] = MatrixXs::Zero(nh, np);
    Pz_[t] = MatrixXs::Zero(nu, np);
    QuuinvHuT_[t] = MatrixXs::Zero(nu, nh);
    Qzz_llt_[t] = Eigen::LLT<MatrixXs>(nz);
    Hu_lu_[t] = Eigen::FullPivLU<MatrixXs>(nh, nu);
    Hu_qr_[t] = Eigen::ColPivHouseholderQR<MatrixXs>(nu, nh);
    Hy_lu_[t] = Eigen::PartialPivLU<MatrixXs>(nh);
  }
  // Terminal constraint data
  const std::size_t nh_T = problem_->get_terminalModel()->get_nh_T();
  Kcs_.resize(T);
  QuuKc_Quc_.resize(T);
  Qzc_.resize(T);
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::size_t nu = model->get_nu();
    const std::size_t nh = model->get_nh();
    Kcs_[t] = MatrixXs::Zero(nh, nh_T);
    QuuKc_Quc_[t] = MatrixXs::Zero(nu, nh_T);
    const std::size_t nz = eq_solver_ == Schur ? nh : nu - nh;
    Qzc_[t] = MatrixXs::Zero(nz, nh_T);
  }
}

template <typename Scalar>
void SolverIntroTpl<Scalar>::calcLuNullDir() {
  START_PROFILER("SolverIntro::calcLuNullDir");
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract>>& datas =
      problem_->get_runningDatas();
#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(problem_->get_nthreads())
#endif
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::shared_ptr<ActionDataAbstract>& data = datas[t];
    if (model->get_nu() > 0 && model->get_nh() > 0) {
      const std::size_t nh = model->get_nh();
      Hu_lu_[t].compute(data->Hu);
      Hu_rank_[t] = Hu_lu_[t].rank();
      if (Hu_rank_[t] != nh) {
        throw_pretty("constrained backward error: Hu is rank deficient");
      }
      YZ_[t].leftCols(Hu_rank_[t]).noalias() =
          (Hu_lu_[t].permutationP() * data->Hu).transpose();
      YZ_[t].rightCols(model->get_nu() - Hu_rank_[t]) = Hu_lu_[t].kernel();
      const Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic,
                         Eigen::RowMajor>
          Y = YZ_[t].leftCols(Hu_rank_[t]);
      Hy_[t].noalias() = data->Hu * Y;
      Hy_lu_[t].compute(Hy_[t]);
      ks_[t].noalias() = Hy_lu_[t].solve(data->h);
      Ks_[t].noalias() = Hy_lu_[t].solve(data->Hx);
      kz_[t].noalias() = Y * ks_[t];
      Kz_[t].noalias() = Y * Ks_[t];
      if (this->n_phases_ > 0 && model->get_np() > 0) {
        Ps_[t].noalias() = Hy_lu_[t].solve(data->Hp);
        Pz_[t].noalias() = Y * Ps_[t];
      }
    }
  }
  STOP_PROFILER("SolverIntro::calcLuNullDir");
}

template <typename Scalar>
void SolverIntroTpl<Scalar>::calcQrNullDir() {
  START_PROFILER("SolverIntro::calcQrNullDir");
  const std::size_t T = problem_->get_T();
  const std::vector<std::shared_ptr<ActionModelAbstract>>& models =
      problem_->get_runningModels();
  const std::vector<std::shared_ptr<ActionDataAbstract>>& datas =
      problem_->get_runningDatas();
#ifdef CROCODDYL_WITH_MULTITHREADING
#pragma omp parallel for num_threads(problem_->get_nthreads())
#endif
  for (std::size_t t = 0; t < T; ++t) {
    const std::shared_ptr<ActionModelAbstract>& model = models[t];
    const std::shared_ptr<ActionDataAbstract>& data = datas[t];
    if (model->get_nu() > 0 && model->get_nh() > 0) {
      const std::size_t nh = model->get_nh();
      Hu_qr_[t].compute(data->Hu.transpose());
      YZ_[t] = Hu_qr_[t].householderQ();
      Hu_rank_[t] = Hu_qr_[t].rank();
      if (Hu_rank_[t] != nh) {
        throw_pretty("constrained backward error: Hu is rank deficient");
      }
      const Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic,
                         Eigen::RowMajor>
          Y = YZ_[t].leftCols(Hu_rank_[t]);
      Hy_[t].noalias() = data->Hu * Y;
      Hy_lu_[t].compute(Hy_[t]);
      ks_[t].noalias() = Hy_lu_[t].solve(data->h);
      Ks_[t].noalias() = Hy_lu_[t].solve(data->Hx);
      kz_[t].noalias() = Y * ks_[t];
      Kz_[t].noalias() = Y * Ks_[t];
      if (this->n_phases_ > 0 && model->get_np() > 0) {
        Ps_[t].noalias() = Hy_lu_[t].solve(data->Hp);
        Pz_[t].noalias() = Y * Ps_[t];
      }
    }
  }
  STOP_PROFILER("SolverIntro::calcQrNullDir");
}

template <typename Scalar>
void SolverIntroTpl<Scalar>::computeNullPolicy(const std::size_t t) {
  START_PROFILER("SolverIntro::computeNullPolicy");
  const std::shared_ptr<ActionModelAbstract>& model =
      problem_->get_runningModels()[t];
  const std::size_t nu = model->get_nu();
  const std::size_t nh = model->get_nh();
  if (nu > 0 && nh > 0) {
    START_PROFILER("SolverIntro::Qzz_inv");
    const std::size_t rank = Hu_rank_[t];
    const std::size_t nullity = nu - rank;
    const Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic,
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
    Qzx_[t] = Qxz_[t].transpose();
    QzzinvQzu_[t] = Quz_[t].transpose();
    Qzz_llt_[t].solveInPlace(Qz_[t]);
    Qzz_llt_[t].solveInPlace(Qzx_[t]);
    Qzz_llt_[t].solveInPlace(QzzinvQzu_[t]);
    Qz_[t].noalias() -= QzzinvQzu_[t] * kz_[t];
    Qzx_[t].noalias() -= QzzinvQzu_[t] * Kz_[t];
    k_[t] = kz_[t];
    K_[t] = Kz_[t];
    k_[t].noalias() += Z * Qz_[t];
    K_[t].noalias() += Z * Qzx_[t];
  } else if (nu > 0) {
    SolverFDDP::computePolicy(t);
  }
  STOP_PROFILER("SolverIntro::computeNullPolicy");
}

template <typename Scalar>
void SolverIntroTpl<Scalar>::parametrizedPolicy(const std::size_t t) {
  START_PROFILER("SolverIntro::parametrizedPolicy");
  const std::shared_ptr<ActionModelAbstract>& model =
      problem_->get_runningModels()[t];
  const std::shared_ptr<ActionDataAbstract>& data =
      problem_->get_runningDatas()[t];
  const std::size_t nu = model->get_nu();
  const std::size_t nh = model->get_nh();
  if (nu == 0) {
    STOP_PROFILER("SolverIntro::parametrizedPolicy");
    return;
  }
  if (nh > 0 && (eq_solver_ == LuNull || eq_solver_ == QrNull)) {
    // Null-space parametrized policy:
    //   P = Pz + Z * (Qzz^{-1} * (Qpu * Z)^T - Qzz^{-1} * Quz^T * Pz)
    // with Pz being the particular control response induced by Hp.
    // Qzz_llt_ was already computed in computeNullPolicy.
    const std::size_t rank = Hu_rank_[t];
    const std::size_t nullity = nu - rank;
    const Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>
        Z = YZ_[t].rightCols(nullity);
    Qpz_[t].noalias() = Qpu_[t] * Z;
    Pn_[t] = Qpz_[t].transpose();
    Qzz_llt_[t].solveInPlace(Pn_[t]);
    Pn_[t].noalias() -= QzzinvQzu_[t] * Pz_[t];
    P_[t] = Pz_[t];
    P_[t].noalias() += Z * Pn_[t];
  } else if (nh > 0 && eq_solver_ == Schur) {
    SolverFDDP::parametrizedPolicy(t);
    Ps_[t] = data->Hp;
    Ps_[t].noalias() -= data->Hu * P_[t];
    P_[t].noalias() += QuuinvHuT_[t] * Ps_[t];
  } else {
    SolverFDDP::parametrizedPolicy(t);
  }
  STOP_PROFILER("SolverIntro::parametrizedPolicy");
}

template <typename Scalar>
void SolverIntroTpl<Scalar>::parametrizedValueFunction(
    const std::size_t t, const std::shared_ptr<ActionModelAbstract>& model) {
  START_PROFILER("SolverIntro::parametrizedValueFunction");
  const std::size_t nu = model->get_nu();
  Vp_[t] = Qp_[t];
  Vpx_[t] = Qpx_[t];
  Vpp_[t] = Qpp_[t];
  if (nu != 0) {
    Quuk_[t].noalias() = Quu_[t] * k_[t];
    Quuk_[t] -= Qu_[t];

    QuuP_2Qpu_[t].noalias() = Quu_[t] * P_[t];
    QuuP_2Qpu_[t].noalias() -= Scalar(2.) * Qpu_[t].transpose();
    KQuu_2Qxu_[t] += Qxu_[t];

    Vp_[t].noalias() -= Qpu_[t] * k_[t];
    Vp_[t].noalias() += P_[t].transpose() * Quuk_[t];
    Vpp_[t].noalias() += P_[t].transpose() * QuuP_2Qpu_[t];
    Vpx_[t].noalias() -= Qpu_[t] * K_[t];
    Vpx_[t].noalias() += P_[t].transpose() * KQuu_2Qxu_[t].transpose();

    Quuk_[t] += Qu_[t];
    KQuu_2Qxu_[t] -= Qxu_[t];
  }
  const std::size_t np = model->get_np();
  Vpp_sym_.topLeftCorner(np, np).noalias() =
      Scalar(0.5) * (Vpp_[t] + Vpp_[t].transpose());
  Vpp_[t] = Vpp_sym_.topLeftCorner(np, np);
  Vpx_f_[t].noalias() = Vpx_[t] * fs_[t];
  STOP_PROFILER("SolverIntro::parametrizedValueFunction");
}

template <typename Scalar>
void SolverIntroTpl<Scalar>::computeNullBatchPolicy(const std::size_t t) {
  START_PROFILER("SolverIntro::computeNullBatchPolicy");
  const std::shared_ptr<ActionModelAbstract>& model =
      problem_->get_runningModels()[t];
  const std::size_t nu = model->get_nu();
  const std::size_t nh = model->get_nh();
  if (nu > 0 && nh > 0) {
    const std::size_t rank = Hu_rank_[t];
    const std::size_t nullity = nu - rank;
    const Eigen::Block<MatrixXs, Eigen::Dynamic, Eigen::Dynamic,
                       Eigen::RowMajor>
        Z = YZ_[t].rightCols(nullity);
    Qzc_[t].noalias() = Z.transpose() * Quc_[t];
    Qzz_llt_[t].solveInPlace(Qzc_[t]);
    Kc_[t].noalias() = Z * Qzc_[t];
  } else if (nu > 0) {
    // Unconstrained policy
    SolverFDDP::computeBatchPolicy(t);
  }
  STOP_PROFILER("SolverIntro::computeNullBatchPolicy");
}

template <typename Scalar>
void SolverIntroTpl<Scalar>::computeSchurPolicy(const std::size_t t) {
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
    Eigen::Transpose<MatrixXs> HuQuuinv = QuuinvHuT_[t].transpose();
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

template <typename Scalar>
void SolverIntroTpl<Scalar>::computeSchurBatchPolicy(const std::size_t t) {
  START_PROFILER("SolverIntro::computeSchurBatchPolicy");
  const std::shared_ptr<ActionModelAbstract>& model =
      problem_->get_runningModels()[t];
  const std::shared_ptr<ActionDataAbstract>& data =
      problem_->get_runningDatas()[t];
  const std::size_t nu = model->get_nu();
  const std::size_t nh = model->get_nh();
  if (nu > 0) {
    SolverFDDP::computeBatchPolicy(t);
  }
  if (nu > 0 && nh > 0) {
    Kcs_[t].noalias() = -data->Hu * Kc_[t];
    Kc_[t].noalias() += QuuinvHuT_[t] * Kcs_[t];
  }
  STOP_PROFILER("SolverIntro::computeSchurBatchPolicy");
}

template <typename Scalar>
template <typename NewScalar>
SolverIntroTpl<NewScalar> SolverIntroTpl<Scalar>::cast() const {
  typedef SolverIntroTpl<NewScalar> ReturnType;
  typedef ShootingProblemTpl<NewScalar> ProblemType;
  if (problem_->get_n_phases() != 0) {
    throw_pretty(
        "Invalid operation: parameterized problems cannot be cast by "
        "SolverIntro.");
  }
  auto sp = std::dynamic_pointer_cast<ShootingProblemTpl<Scalar>>(problem_);
  if (sp == nullptr) {
    throw_pretty(
        "Invalid operation: parameterized problems cannot be cast by "
        "SolverIntro.");
  }
  ReturnType ret(
      std::static_pointer_cast<ProblemAbstractTpl<NewScalar>>(
          std::make_shared<ProblemType>(sp->template cast<NewScalar>())),
      dyn_solver_, eq_solver_, term_solver_, this->get_astate_solver());
  if (dyn_solver_ == HybridShoot && Ts_.size() > 1) {
    ret.set_dynamics_solver(dyn_solver_, Ts_[1] - Ts_[0]);
  }
  // Setting the abstract parameters
  ret.setCallbacks(vector_cast<NewScalar>(callbacks_));
  ret.set_th_acceptstep(scalar_cast<NewScalar>(th_acceptstep_));
  ret.set_th_gaptol(scalar_cast<NewScalar>(th_gaptol_));
  ret.set_feasnorm(feasnorm_);
  ret.set_th_stop(
      std::sqrt(std::numeric_limits<NewScalar>::epsilon()) < NewScalar(th_stop_)
          ? scalar_cast<NewScalar>(th_stop_)
          : std::sqrt(
                std::numeric_limits<NewScalar>::
                    epsilon()));  // Stopping threshold shouldn't be lower than
                                  // square root of the machine precision
  // Setting the FDDP parameters
  ret.set_alphas(vector_cast<NewScalar>(alphas_));
  ret.set_reg_incfactor(scalar_cast<NewScalar>(reg_incfactor_));
  ret.set_reg_decfactor(scalar_cast<NewScalar>(reg_decfactor_));
  ret.set_reg_min(
      ScaleNumerics<Scalar>(1e-9) < NewScalar(reg_min_)
          ? scalar_cast<NewScalar>(reg_min_)
          : ScaleNumerics<NewScalar>(
                1e-9));  // Minimum regularization value shouldn't be lower than
                         // 1e-9 or 1e-5 for doubles or floats
  ret.set_reg_max(
      ScaleNumerics<Scalar>(1e9, 1e-4) > NewScalar(reg_max_)
          ? scalar_cast<NewScalar>(reg_max_)
          : ScaleNumerics<NewScalar>(
                1e9, 1e-4));  // Maximum regularization value shouldn't be
                              // higher than 1e9 or 1e5 for doubles or floats
  ret.set_th_grad(scalar_cast<NewScalar>(th_grad_));
  ret.set_th_noimprovement(scalar_cast<NewScalar>(th_noimprovement_));
  ret.set_th_stepdec(scalar_cast<NewScalar>(th_stepdec_));
  ret.set_th_stepinc(scalar_cast<NewScalar>(th_stepinc_));
  ret.set_th_minimprove(scalar_cast<NewScalar>(th_minimprove_));
  ret.set_th_acceptnegstep(scalar_cast<NewScalar>(th_acceptnegstep_));
  ret.set_th_acceptminstep(scalar_cast<NewScalar>(th_acceptminstep_));
  ret.set_rho(scalar_cast<NewScalar>(rho_));
  ret.set_th_minfeas(scalar_cast<NewScalar>(th_minfeas_));
  ret.set_upsilon_decfactor(scalar_cast<NewScalar>(upsilon_decfactor_));
  ret.set_zero_upsilon(zero_upsilon_);
  ret.setCandidate(vector_cast<NewScalar>(xs_), vector_cast<NewScalar>(us_),
                   is_feasible_);
  return ret;
}

template <typename Scalar>
EqualitySolverType SolverIntroTpl<Scalar>::get_equality_solver() const {
  return eq_solver_;
}

template <typename Scalar>
const std::vector<std::size_t>& SolverIntroTpl<Scalar>::get_Hu_rank() const {
  return Hu_rank_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::MatrixXs>&
SolverIntroTpl<Scalar>::get_YZ() const {
  return YZ_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::MatrixXs>&
SolverIntroTpl<Scalar>::get_Qzz() const {
  return Qzz_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::MatrixXs>&
SolverIntroTpl<Scalar>::get_Qxz() const {
  return Qxz_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::MatrixXs>&
SolverIntroTpl<Scalar>::get_Quz() const {
  return Quz_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::VectorXs>&
SolverIntroTpl<Scalar>::get_Qz() const {
  return Qz_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::MatrixXs>&
SolverIntroTpl<Scalar>::get_Hy() const {
  return Hy_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::VectorXs>&
SolverIntroTpl<Scalar>::get_kz() const {
  return kz_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::MatrixXs>&
SolverIntroTpl<Scalar>::get_Kz() const {
  return Kz_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::VectorXs>&
SolverIntroTpl<Scalar>::get_ks() const {
  return ks_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::MatrixXs>&
SolverIntroTpl<Scalar>::get_Ks() const {
  return Ks_;
}

template <typename Scalar>
const std::vector<typename MathBaseTpl<Scalar>::MatrixXs>&
SolverIntroTpl<Scalar>::get_Qzc() const {
  return Qzc_;
}

template <typename Scalar>
void SolverIntroTpl<Scalar>::set_equality_solver(
    const EqualitySolverType type) {
  if (eq_solver_ != type) {
    eq_solver_ = type;
    resizeRunningData();
    resizeTerminalData();
  }
}

}  // namespace crocoddyl
