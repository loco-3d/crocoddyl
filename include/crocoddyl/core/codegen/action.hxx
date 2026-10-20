
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, LAAS-CNRS, INRIA, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_CODEGEN_ACTION_HXX_
#define CROCODDYL_CORE_CODEGEN_ACTION_HXX_

namespace crocoddyl {

template <typename Scalar>
ActionModelCodeGenTpl<Scalar>::ActionModelCodeGenTpl(
    std::shared_ptr<Base> model, const std::string& lib_fname, bool autodiff,
    const std::size_t np, ParamsEnvironment updateParams, CompilerType compiler,
    const std::string& compile_options)
    : Base(model->get_state(), model->get_nu(), model->get_nr(),
           model->get_ng(), model->get_nh(), model->get_ng_T(),
           model->get_nh_T(), np == 0u ? model->get_np() : np),
      model_(model),
      ad_model_(model->template cast<ADScalar>()),
      ad_data_(ad_model_->createData()),
      ad_data_pert_(ad_model_->createData()),
      autodiff_(autodiff),
      np_(np == 0u ? model->get_np() : np),
      nX_(state_->get_nx() + nu_ + np_),
      nX_T_(state_->get_nx() + np_),
      nX2_(nX_ + (autodiff ? state_->get_ndx() + nu_ + np_ : 0u)),
      nX2_T_(nX_T_ + (autodiff ? state_->get_ndx() + np_ : 0u)),
      nX3_(state_->get_nx()),
      nY1_(1 + state_->get_nx() + ng_ + nh_),
      nY1_T_(1 + ng_T_ + nh_T_),
      nY3_(nu_),
      ad_X_(nX_),
      ad_X_T_(nX_T_),
      ad_X2_(nX2_),
      ad_X2_T_(nX2_T_),
      ad_X3_(nX3_),
      ad_Y1_(nY1_),
      ad_Y1_T_(nY1_T_),
      ad_Y3_(nY3_),
      Y1fun_name_("calc"),
      Y1Tfun_name_("calc_T"),
      Y2fun_name_("calcDiff"),
      Y2Tfun_name_("calcDiff_T"),
      Y2Costfun_name_("calcDiff_cost"),
      Y2CostTfun_name_("calcDiff_cost_T"),
      Y3fun_name_("quasiStatic"),
      lib_fname_(lib_fname),
      compiler_type_(compiler),
      compile_options_(compile_options),
      updateParams_(updateParams),
      ad_calc_(std::make_unique<ADFun>()),
      ad_calc_T_(std::make_unique<ADFun>()),
      ad_calcDiff_(std::make_unique<ADFun>()),
      ad_calcDiff_T_(std::make_unique<ADFun>()),
      ad_calcDiffCost_(std::make_unique<ADFun>()),
      ad_calcDiffCost_T_(std::make_unique<ADFun>()),
      ad_quasiStatic_(std::make_unique<ADFun>()) {
  const std::size_t ndx = state_->get_ndx();
  nYh_ = ndx * ndx + ndx * nu_ + nu_ * nu_;
  nYh_ += np_ * np_ + np_ * ndx + np_ * nu_;
  nYh_T_ = ndx * ndx + np_ * np_ + np_ * ndx;
  if (autodiff_) {
    nY2_ = 1 + ndx + ng_ + nh_;
    nY2_T_ = 1 + ndx + ng_T_ + nh_T_;
  } else {
    nY2_ = ndx * ndx + ndx * nu_ + ndx * np_;     // dynamics
    nY2_ += ndx + nu_ + np_;                      // cost gradients
    nY2_ += ndx * ndx + ndx * nu_ + nu_ * nu_;    // classic cost Hessian
    nY2_ += np_ * np_ + np_ * ndx + np_ * nu_;    // parameter cost Hessian
    nY2_ += ng_ * (ndx + nu_ + np_);              // inequalities
    nY2_ += nh_ * (ndx + nu_ + np_);              // equalities
    nY2_T_ = ndx + np_;                           // terminal cost gradients
    nY2_T_ += ndx * ndx + np_ * np_ + np_ * ndx;  // terminal Hessian
    nY2_T_ += (ng_T_ + nh_T_) * (ndx + np_);      // terminal constraints
  }
  ad_Y2_.resize(nY2_);
  ad_Y2_T_.resize(nY2_T_);
  initLib();
  compileLib();
  loadLib(lib_fname_);
  wCostHess_ = VectorXs::Zero(nY2_);
  wCostHess_(0) = Scalar(1.);
  wCostHess_T_ = VectorXs::Zero(nY2_T_);
  wCostHess_T_(0) = Scalar(1.);
  wCostHessScalar_ = VectorXs::Ones(1);
  wCostHessScalar_T_ = VectorXs::Ones(1);
}

template <typename Scalar>
ActionModelCodeGenTpl<Scalar>::ActionModelCodeGenTpl(
    std::shared_ptr<ADBase> ad_model, const std::string& lib_fname,
    bool autodiff, const std::size_t np, ParamsEnvironment updateParams,
    CompilerType compiler, const std::string& compile_options)
    : Base(ad_model->get_state()->template cast<Scalar>(), ad_model->get_nu(),
           ad_model->get_nr(), ad_model->get_ng(), ad_model->get_nh(),
           ad_model->get_ng_T(), ad_model->get_nh_T(),
           np == 0u ? ad_model->get_np() : np),
      model_(ad_model->template cast<Scalar>()),
      ad_model_(ad_model),
      ad_data_(ad_model_->createData()),
      ad_data_pert_(ad_model_->createData()),
      autodiff_(autodiff),
      np_(np == 0u ? ad_model->get_np() : np),
      nX_(state_->get_nx() + nu_ + np_),
      nX_T_(state_->get_nx() + np_),
      nX2_(nX_ + (autodiff ? state_->get_ndx() + nu_ + np_ : 0u)),
      nX2_T_(nX_T_ + (autodiff ? state_->get_ndx() + np_ : 0u)),
      nX3_(state_->get_nx()),
      nY1_(1 + state_->get_nx() + ng_ + nh_),
      nY1_T_(1 + ng_T_ + nh_T_),
      nY3_(nu_),
      ad_X_(nX_),
      ad_X_T_(nX_T_),
      ad_X2_(nX2_),
      ad_X2_T_(nX2_T_),
      ad_X3_(nX3_),
      ad_Y1_(nY1_),
      ad_Y1_T_(nY1_T_),
      ad_Y3_(nY3_),
      Y1fun_name_("calc"),
      Y1Tfun_name_("calc_T"),
      Y2fun_name_("calcDiff"),
      Y2Tfun_name_("calcDiff_T"),
      Y2Costfun_name_("calcDiff_cost"),
      Y2CostTfun_name_("calcDiff_cost_T"),
      Y3fun_name_("quasiStatic"),
      lib_fname_(lib_fname),
      compiler_type_(compiler),
      compile_options_(compile_options),
      updateParams_(updateParams),
      ad_calc_(std::make_unique<ADFun>()),
      ad_calc_T_(std::make_unique<ADFun>()),
      ad_calcDiff_(std::make_unique<ADFun>()),
      ad_calcDiff_T_(std::make_unique<ADFun>()),
      ad_calcDiffCost_(std::make_unique<ADFun>()),
      ad_calcDiffCost_T_(std::make_unique<ADFun>()),
      ad_quasiStatic_(std::make_unique<ADFun>()) {
  const std::size_t ndx = state_->get_ndx();
  nYh_ = ndx * ndx + ndx * nu_ + nu_ * nu_;
  nYh_ += np_ * np_ + np_ * ndx + np_ * nu_;
  nYh_T_ = ndx * ndx + np_ * np_ + np_ * ndx;
  if (autodiff_) {
    nY2_ = 1 + ndx + ng_ + nh_;
    nY2_T_ = 1 + ndx + ng_T_ + nh_T_;
  } else {
    nY2_ = ndx * ndx + ndx * nu_ + ndx * np_;     // dynamics
    nY2_ += ndx + nu_ + np_;                      // cost gradients
    nY2_ += ndx * ndx + ndx * nu_ + nu_ * nu_;    // classic cost Hessian
    nY2_ += np_ * np_ + np_ * ndx + np_ * nu_;    // parameter cost Hessian
    nY2_ += ng_ * (ndx + nu_ + np_);              // inequalities
    nY2_ += nh_ * (ndx + nu_ + np_);              // equalities
    nY2_T_ = ndx + np_;                           // terminal cost gradients
    nY2_T_ += ndx * ndx + np_ * np_ + np_ * ndx;  // terminal Hessian
    nY2_T_ += (ng_T_ + nh_T_) * (ndx + np_);      // terminal constraints
  }
  ad_Y2_.resize(nY2_);
  ad_Y2_T_.resize(nY2_T_);
  initLib();
  compileLib();
  loadLib(lib_fname_);
  wCostHess_ = VectorXs::Zero(nY2_);
  wCostHess_(0) = Scalar(1.);
  wCostHess_T_ = VectorXs::Zero(nY2_T_);
  wCostHess_T_(0) = Scalar(1.);
  wCostHessScalar_ = VectorXs::Ones(1);
  wCostHessScalar_T_ = VectorXs::Ones(1);
}

template <typename Scalar>
ActionModelCodeGenTpl<Scalar>::ActionModelCodeGenTpl(
    const ActionModelCodeGenTpl<Scalar>& other)
    : Base(other),
      model_(other.model_),
      ad_model_(other.ad_model_),
      ad_data_(ad_model_ ? ad_model_->createData() : nullptr),
      ad_data_pert_(ad_model_ ? ad_model_->createData() : nullptr),
      autodiff_(other.autodiff_),
      np_(other.np_),
      nX_(other.nX_),
      nX_T_(other.nX_T_),
      nX2_(other.nX2_),
      nX2_T_(other.nX2_T_),
      nX3_(other.nX3_),
      nY1_(other.nY1_),
      nY1_T_(other.nY1_T_),
      nY2_(other.nY2_),
      nY2_T_(other.nY2_T_),
      nYh_(other.nYh_),
      nYh_T_(other.nYh_T_),
      nY3_(other.nY3_),
      ad_X_(other.nX_),
      ad_X_T_(other.nX_T_),
      ad_X2_(other.nX2_),
      ad_X2_T_(other.nX2_T_),
      ad_X3_(other.nX3_),
      ad_Y1_(other.nY1_),
      ad_Y1_T_(other.nY1_T_),
      ad_Y2_(other.nY2_),
      ad_Y2_T_(other.nY2_T_),
      ad_Y3_(other.nY3_),
      Y1fun_name_(other.Y1fun_name_),
      Y1Tfun_name_(other.Y1Tfun_name_),
      Y2fun_name_(other.Y2fun_name_),
      Y2Tfun_name_(other.Y2Tfun_name_),
      Y2Costfun_name_(other.Y2Costfun_name_),
      Y2CostTfun_name_(other.Y2CostTfun_name_),
      Y3fun_name_(other.Y3fun_name_),
      lib_fname_(other.lib_fname_),
      compiler_type_(other.compiler_type_),
      compile_options_(other.compile_options_),
      updateParams_(other.updateParams_),
      params_(other.params_),
      ad_calc_(clone_adfun(*other.ad_calc_)),
      ad_calc_T_(clone_adfun(*other.ad_calc_T_)),
      ad_calcDiff_(clone_adfun(*other.ad_calcDiff_)),
      ad_calcDiff_T_(clone_adfun(*other.ad_calcDiff_T_)),
      ad_calcDiffCost_(clone_adfun(*other.ad_calcDiffCost_)),
      ad_calcDiffCost_T_(clone_adfun(*other.ad_calcDiffCost_T_)),
      ad_quasiStatic_(clone_adfun(*other.ad_quasiStatic_)),
      calcCG_(std::make_unique<CSourceGen>(*ad_calc_, Y1fun_name_)),
      calcCG_T_(std::make_unique<CSourceGen>(*ad_calc_T_, Y1Tfun_name_)),
      calcDiffCG_(std::make_unique<CSourceGen>(*ad_calcDiff_, Y2fun_name_)),
      calcDiffCG_T_(
          std::make_unique<CSourceGen>(*ad_calcDiff_T_, Y2Tfun_name_)),
      calcDiffCostCG_(
          std::make_unique<CSourceGen>(*ad_calcDiffCost_, Y2Costfun_name_)),
      calcDiffCostCG_T_(
          std::make_unique<CSourceGen>(*ad_calcDiffCost_T_, Y2CostTfun_name_)),
      quasiStaticCG_(
          std::make_unique<CSourceGen>(*ad_quasiStatic_, Y3fun_name_)),
      libCG_(std::make_unique<LibraryCSourceGen>(
          *calcCG_, *calcCG_T_, *calcDiffCG_, *calcDiffCG_T_, *calcDiffCostCG_,
          *calcDiffCostCG_T_, *quasiStaticCG_)),
      dynLibManager_(
          std::make_unique<LibraryProcessor>(*other.libCG_, lib_fname_)) {
  loadLib(lib_fname_);
  wCostHess_ = other.wCostHess_;
  wCostHess_T_ = other.wCostHess_T_;
  wCostHessScalar_ = other.wCostHessScalar_;
  wCostHessScalar_T_ = other.wCostHessScalar_T_;
}

template <typename Scalar>
ActionModelCodeGenTpl<Scalar>::ActionModelCodeGenTpl(
    const std::string& lib_fname, std::shared_ptr<Base> model)
    : Base(model->get_state()->template cast<Scalar>(), model->get_nu(),
           model->get_nr(), model->get_ng(), model->get_nh(), model->get_ng_T(),
           model->get_nh_T(), model->get_np()),
      model_(model),
      Y1fun_name_("calc"),
      Y1Tfun_name_("calc_T"),
      Y2fun_name_("calcDiff"),
      Y2Tfun_name_("calcDiff_T"),
      Y2Costfun_name_("calcDiff_cost"),
      Y2CostTfun_name_("calcDiff_cost_T"),
      Y3fun_name_("quasiStatic"),
      lib_fname_(lib_fname),
      ad_calc_(std::make_unique<ADFun>()),
      ad_calc_T_(std::make_unique<ADFun>()),
      ad_calcDiff_(std::make_unique<ADFun>()),
      ad_calcDiff_T_(std::make_unique<ADFun>()),
      ad_calcDiffCost_(std::make_unique<ADFun>()),
      ad_calcDiffCost_T_(std::make_unique<ADFun>()),
      ad_quasiStatic_(std::make_unique<ADFun>()) {
  loadLib(lib_fname_);
  nX_ = calcFun_->Domain();
  nX_T_ = calcFun_T_->Domain();
  nY1_ = calcFun_->Range();
  nY1_T_ = calcFun_T_->Range();
  nX2_ = calcDiffFun_->Domain();
  nX2_T_ = calcDiffFun_T_->Domain();
  nY2_ = calcDiffFun_->Range();
  nY2_T_ = calcDiffFun_T_->Range();
  nYh_ = calcDiffCostFun_->Range();
  nYh_T_ = calcDiffCostFun_T_->Range();
  nY3_ = quasiStaticFun_->Range();
  nX3_ = quasiStaticFun_->Domain();
  np_ = nX_T_ - state_->get_nx();
  const std::size_t nx = model_->get_state()->get_nx();
  if (nX_ != nx + nu_ + np_) {
    throw_pretty(
        "The number of independent variables nX in the code generated library "
        "is not equal to the number of independent variables in the model");
  }
  if (nX_T_ != nx + np_) {
    throw_pretty(
        "The number of independent variables nX_T in the code generated "
        "library is not equal to the number of independent variables in the "
        "model");
  }
  if (nY1_ != 1 + nx + ng_ + nh_) {
    throw_pretty(
        "The number of dependent variables nY1 in the code generated library "
        "is not equal to the number of dependent variables in the model");
  }
  if (nY1_T_ != 1 + ng_T_ + nh_T_) {
    throw_pretty(
        "The number of dependent variables nY1_T in the code generated library "
        "is not equal to the number of dependent variables in the model");
  }
  if (nY3_ != nu_) {
    throw_pretty(
        "The number of dependent variables nY3 in the code generated library "
        "is not equal to the number of dependent variables in the model");
  }
  ad_X_.resize(nX_);
  ad_X_T_.resize(nX_T_);
  ad_X2_.resize(nX2_);
  ad_X2_T_.resize(nX2_T_);
  ad_X3_.resize(nX3_);
  ad_Y1_.resize(nY1_);
  ad_Y1_T_.resize(nY1_T_);
  ad_Y2_.resize(nY2_);
  ad_Y2_T_.resize(nY2_T_);
  ad_Y3_.resize(nY3_);
  autodiff_ = calcDiffFun_->isJacobianAvailable();
  if (autodiff_ && (!calcDiffCostFun_ || !calcDiffCostFun_T_)) {
    throw_pretty(
        "The code generated library contains a stale autodiff calcDiff tape; "
        "please regenerate it");
  }
  wCostHess_ = VectorXs::Zero(nY2_);
  wCostHess_(0) = Scalar(1.);
  wCostHess_T_ = VectorXs::Zero(nY2_T_);
  wCostHess_T_(0) = Scalar(1.);
  wCostHessScalar_ = VectorXs::Ones(1);
  wCostHessScalar_T_ = VectorXs::Ones(1);
}

template <typename Scalar>
ActionModelCodeGenTpl<Scalar>::ActionModelCodeGenTpl(
    const std::string& lib_fname, std::shared_ptr<ADBase> ad_model)
    : Base(ad_model->get_state()->template cast<Scalar>(), ad_model->get_nu(),
           ad_model->get_nr(), ad_model->get_ng(), ad_model->get_nh(),
           ad_model->get_ng_T(), ad_model->get_nh_T(), ad_model->get_np()),
      model_(ad_model->template cast<Scalar>()),
      Y1fun_name_("calc"),
      Y1Tfun_name_("calc_T"),
      Y2fun_name_("calcDiff"),
      Y2Tfun_name_("calcDiff_T"),
      Y2Costfun_name_("calcDiff_cost"),
      Y2CostTfun_name_("calcDiff_cost_T"),
      Y3fun_name_("quasiStatic"),
      lib_fname_(lib_fname),
      ad_calc_(std::make_unique<ADFun>()),
      ad_calc_T_(std::make_unique<ADFun>()),
      ad_calcDiff_(std::make_unique<ADFun>()),
      ad_calcDiff_T_(std::make_unique<ADFun>()),
      ad_calcDiffCost_(std::make_unique<ADFun>()),
      ad_calcDiffCost_T_(std::make_unique<ADFun>()),
      ad_quasiStatic_(std::make_unique<ADFun>()) {
  loadLib(lib_fname_);
  nX_ = calcFun_->Domain();
  nX_T_ = calcFun_T_->Domain();
  nY1_ = calcFun_->Range();
  nY1_T_ = calcFun_T_->Range();
  nX2_ = calcDiffFun_->Domain();
  nX2_T_ = calcDiffFun_T_->Domain();
  nY2_ = calcDiffFun_->Range();
  nY2_T_ = calcDiffFun_T_->Range();
  nYh_ = calcDiffCostFun_->Range();
  nYh_T_ = calcDiffCostFun_T_->Range();
  nY3_ = quasiStaticFun_->Range();
  nX3_ = quasiStaticFun_->Domain();
  np_ = nX_T_ - state_->get_nx();
  const std::size_t nx = model_->get_state()->get_nx();
  if (nX_ != nx + nu_ + np_) {
    throw_pretty(
        "The number of independent variables nX in the code generated library "
        "is not equal to the number of independent variables in the model");
  }
  if (nX_T_ != nx + np_) {
    throw_pretty(
        "The number of independent variables nX_T in the code generated "
        "library is not equal to the number of independent variables in the "
        "model");
  }
  if (nY1_ != 1 + nx + ng_ + nh_) {
    throw_pretty(
        "The number of dependent variables nY1 in the code generated library "
        "is not equal to the number of dependent variables in the model");
  }
  if (nY1_T_ != 1 + ng_T_ + nh_T_) {
    throw_pretty(
        "The number of dependent variables nY1_T in the code generated library "
        "is not equal to the number of dependent variables in the model");
  }
  if (nY3_ != nu_) {
    throw_pretty(
        "The number of dependent variables nY3 in the code generated library "
        "is not equal to the number of dependent variables in the model");
  }
  ad_X_.resize(nX_);
  ad_X_T_.resize(nX_T_);
  ad_X2_.resize(nX2_);
  ad_X2_T_.resize(nX2_T_);
  ad_X3_.resize(nX3_);
  ad_Y1_.resize(nY1_);
  ad_Y1_T_.resize(nY1_T_);
  ad_Y2_.resize(nY2_);
  ad_Y2_T_.resize(nY2_T_);
  ad_Y3_.resize(nY3_);
  autodiff_ = calcDiffFun_->isJacobianAvailable();
  if (autodiff_ && (!calcDiffCostFun_ || !calcDiffCostFun_T_)) {
    throw_pretty(
        "The code generated library contains a stale autodiff calcDiff tape; "
        "please regenerate it");
  }
  wCostHess_ = VectorXs::Zero(nY2_);
  wCostHess_(0) = Scalar(1.);
  wCostHess_T_ = VectorXs::Zero(nY2_T_);
  wCostHess_T_(0) = Scalar(1.);
  wCostHessScalar_ = VectorXs::Ones(1);
  wCostHessScalar_T_ = VectorXs::Ones(1);
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::initLib() {
  START_PROFILER("ActionModelCodeGen::initLib");
  const CodegenEigenThreadGuard eigen_thread_guard(1);
  // Generate source code for calc
  recordCalc();
  calcCG_ =
      std::unique_ptr<CSourceGen>(new CSourceGen(*ad_calc_.get(), Y1fun_name_));
  calcCG_->setCreateForwardZero(true);
  calcCG_->setCreateJacobian(false);
  calcCG_->setCreateHessian(false);
  // Generate source code for calc in terminal nodes
  recordCalc_T();
  calcCG_T_ = std::unique_ptr<CSourceGen>(
      new CSourceGen(*ad_calc_T_.get(), Y1Tfun_name_));
  calcCG_T_->setCreateForwardZero(true);
  calcCG_T_->setCreateJacobian(false);
  calcCG_T_->setCreateHessian(false);
  // Generate source code for calcDiff
  recordCalcDiff();
  calcDiffCG_ = std::unique_ptr<CSourceGen>(
      new CSourceGen(*ad_calcDiff_.get(), Y2fun_name_));
  calcDiffCG_->setCreateForwardZero(!autodiff_);
  calcDiffCG_->setCreateJacobian(autodiff_);
  calcDiffCG_->setCreateHessian(false);
  // Generate source code for calcDiff in terminal nodes
  recordCalcDiff_T();
  calcDiffCG_T_ = std::unique_ptr<CSourceGen>(
      new CSourceGen(*ad_calcDiff_T_.get(), Y2Tfun_name_));
  calcDiffCG_T_->setCreateForwardZero(!autodiff_);
  calcDiffCG_T_->setCreateJacobian(autodiff_);
  calcDiffCG_T_->setCreateHessian(false);
  // Generate local cost-Hessian block functions separately. This avoids
  // generating dense Hessians for every dynamics and constraint output while
  // preserving the wrapped model's calcDiff Hessian convention.
  recordCalcDiffCost();
  calcDiffCostCG_ = std::unique_ptr<CSourceGen>(
      new CSourceGen(*ad_calcDiffCost_.get(), Y2Costfun_name_));
  calcDiffCostCG_->setCreateForwardZero(autodiff_);
  calcDiffCostCG_->setCreateJacobian(false);
  calcDiffCostCG_->setCreateHessian(false);
  recordCalcDiffCost_T();
  calcDiffCostCG_T_ = std::unique_ptr<CSourceGen>(
      new CSourceGen(*ad_calcDiffCost_T_.get(), Y2CostTfun_name_));
  calcDiffCostCG_T_->setCreateForwardZero(autodiff_);
  calcDiffCostCG_T_->setCreateJacobian(false);
  calcDiffCostCG_T_->setCreateHessian(false);
  // Generate source code for quasiStatic
  recordQuasiStatic();
  quasiStaticCG_ = std::unique_ptr<CSourceGen>(
      new CSourceGen(*ad_quasiStatic_.get(), Y3fun_name_));
  quasiStaticCG_->setCreateForwardZero(true);
  quasiStaticCG_->setCreateJacobian(false);
  // Generate library for calc and calcDiff
  libCG_ = std::unique_ptr<LibraryCSourceGen>(new LibraryCSourceGen(
      *calcCG_, *calcCG_T_, *calcDiffCG_, *calcDiffCG_T_, *calcDiffCostCG_,
      *calcDiffCostCG_T_, *quasiStaticCG_));
  // Create dynamic library manager
  dynLibManager_ = std::unique_ptr<LibraryProcessor>(
      new LibraryProcessor(*libCG_, lib_fname_));
  STOP_PROFILER("ActionModelCodeGen::initLib");
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::compileLib() {
  START_PROFILER("ActionModelCodeGen::compileLib");
  if (!dynLibManager_) {
    throw_pretty("The library "
                 << lib_fname_ + SystemInfo::DYNAMIC_LIB_EXTENSION
                 << " should not be compiled again");
  }
  auto splitFlags = [](const std::string& flags) {
    std::istringstream iss(flags);
    std::vector<std::string> out;
    std::string flag;
    while (iss >> flag) {
      out.push_back(flag);
    }
    return out;
  };
  switch (compiler_type_) {
    case GCC: {
      CppAD::cg::GccCompiler<Scalar> compiler(compilerExecutable(GCC));
      std::vector<std::string> compile_flags = compiler.getCompileFlags();
      compile_flags.clear();
      auto extra_flags = splitFlags(compile_options_);
      compile_flags.insert(compile_flags.end(), extra_flags.begin(),
                           extra_flags.end());
      compiler.setCompileFlags(compile_flags);
      dynLibManager_->createDynamicLibrary(compiler, false);
      break;
    }
    case CLANG: {
      CppAD::cg::ClangCompiler<Scalar> compiler(compilerExecutable(CLANG));
      std::vector<std::string> compile_flags = compiler.getCompileFlags();
      compile_flags.clear();
      auto extra_flags = splitFlags(compile_options_);
      compile_flags.insert(compile_flags.end(), extra_flags.begin(),
                           extra_flags.end());
      compiler.setCompileFlags(compile_flags);
      dynLibManager_->createDynamicLibrary(compiler, false);
      break;
    }
  }
  STOP_PROFILER("ActionModelCodeGen::compileLib");
}

template <typename Scalar>
bool ActionModelCodeGenTpl<Scalar>::existLib(
    const std::string& lib_fname) const {
  const std::string filename = lib_fname + SystemInfo::DYNAMIC_LIB_EXTENSION;
  std::ifstream file(filename.c_str());
  return file.good();
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::loadLib(const std::string& lib_fname) {
  if (!existLib(lib_fname)) {
    throw_pretty("The library " << lib_fname + SystemInfo::DYNAMIC_LIB_EXTENSION
                                << " doesn't exist");
  }
  const std::string filename = lib_fname + SystemInfo::DYNAMIC_LIB_EXTENSION;
  if (dynLibManager_) {
    const auto it = dynLibManager_->getOptions().find("dlOpenMode");
    if (it == dynLibManager_->getOptions().end()) {
      dynLib_.reset(new LinuxDynamicLib(filename));
    } else {
      int dlOpenMode = std::stoi(it->second);
      dynLib_.reset(new LinuxDynamicLib(filename, dlOpenMode));
    }
  } else {
    dynLib_.reset(new LinuxDynamicLib(filename));
  }
  calcFun_ = dynLib_->model(Y1fun_name_.c_str());
  calcFun_T_ = dynLib_->model(Y1Tfun_name_.c_str());
  calcDiffFun_ = dynLib_->model(Y2fun_name_.c_str());
  calcDiffFun_T_ = dynLib_->model(Y2Tfun_name_.c_str());
  calcDiffCostFun_ = dynLib_->model(Y2Costfun_name_.c_str());
  calcDiffCostFun_T_ = dynLib_->model(Y2CostTfun_name_.c_str());
  quasiStaticFun_ = dynLib_->model(Y3fun_name_.c_str());
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::update_p(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& p) {
  if (static_cast<std::size_t>(p.size()) != np_) {
    throw_pretty(
        "Invalid argument: " << "p has wrong dimension (it should be " +
                                    std::to_string(np_) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  d->X.tail(np_) = p;
  d->X_T.tail(np_) = p;
  d->X2.segment(state_->get_nx() + nu_, np_) = p;
  d->X2_T.segment(state_->get_nx(), np_) = p;
  if (params_ && d->params_data) {
    params_->update(d->params_data, p);
  }
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::set_params(
    const std::shared_ptr<ActionDataAbstract>& data,
    std::shared_ptr<ParameterManager> params) {
  if (!params) {
    throw_pretty("Invalid argument: " << "params cannot be null");
  }
  if (params->get_np() != np_) {
    throw_pretty(
        "Invalid argument: " << "params has wrong dimension (it should be " +
                                    std::to_string(np_) + ")");
  }
  params_ = params;
  Data* d = static_cast<Data*>(data.get());
  if (d->params_data) {
    update_p(data, d->params_data->params->p);
  } else {
    update_p(data, params_->zero());
  }
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::calc(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  START_PROFILER("ActionModelCodeGen::calc");
  Data* d = static_cast<Data*>(data.get());
  d->resize(this);
  const std::size_t nx = state_->get_nx();
  d->X.head(nx) = x;
  d->X.segment(nx, nu_) = u;
  syncDataParameters(d);
  START_PROFILER("ActionModelCodeGen::calc::ForwardZero");
  calcFun_->ForwardZero(d->X, d->Y1);
  STOP_PROFILER("ActionModelCodeGen::calc::ForwardZero");
  d->set_Y1(this);
  STOP_PROFILER("ActionModelCodeGen::calc");
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::calc(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  START_PROFILER("ActionModelCodeGen::calc_T");
  Data* d = static_cast<Data*>(data.get());
  d->g.conservativeResize(ng_T_);
  d->h.conservativeResize(nh_T_);
  const std::size_t nx = state_->get_nx();
  d->X_T.head(nx) = x;
  syncDataParameters(d);
  START_PROFILER("ActionModelCodeGen::calc_T::ForwardZero");
  calcFun_T_->ForwardZero(d->X_T, d->Y1_T);
  STOP_PROFILER("ActionModelCodeGen::calc_T::ForwardZero");
  d->xnext = x;
  d->set_Y1_T(this);
  STOP_PROFILER("ActionModelCodeGen::calc_T");
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::calcDiff(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  START_PROFILER("ActionModelCodeGen::calcDiff");
  Data* d = static_cast<Data*>(data.get());
  d->resize(this);
  const std::size_t nx = state_->get_nx();
  d->X.head(nx) = x;
  d->X.segment(nx, nu_) = u;
  if (autodiff_) {
    d->X2.head(nx) = x;
    d->X2.segment(nx, nu_) = u;
    d->X2.segment(nX_, state_->get_ndx() + nu_ + np_).setZero();
  }
  syncDataParameters(d);
  if (autodiff_) {
    START_PROFILER("ActionModelCodeGen::calcDiff::Jacobian");
    d->J2 = calcDiffFun_->Jacobian(d->X2);
    STOP_PROFILER("ActionModelCodeGen::calcDiff::Jacobian");
    d->set_D2(this);
    START_PROFILER("ActionModelCodeGen::calcDiff::ForwardZeroCost");
    calcDiffCostFun_->ForwardZero(d->X2, d->Yh);
    STOP_PROFILER("ActionModelCodeGen::calcDiff::ForwardZeroCost");
    d->set_Yh_hessian(this);
  } else {
    START_PROFILER("ActionModelCodeGen::calcDiff::ForwardZero");
    calcDiffFun_->ForwardZero(d->X, d->Y2);
    STOP_PROFILER("ActionModelCodeGen::calcDiff::ForwardZero");
    d->set_Y2(this);
  }
  STOP_PROFILER("ActionModelCodeGen::calcDiff");
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::calcDiff(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x) {
  START_PROFILER("ActionModelCodeGen::calcDiff_T");
  Data* d = static_cast<Data*>(data.get());
  d->Gx.conservativeResize(ng_T_, state_->get_ndx());
  d->Gp.conservativeResize(ng_T_, np_);
  d->Hx.conservativeResize(nh_T_, state_->get_ndx());
  d->Hp.conservativeResize(nh_T_, np_);
  const std::size_t nx = state_->get_nx();
  d->X_T.head(nx) = x;
  if (autodiff_) {
    d->X2_T.head(nx) = x;
    d->X2_T.segment(nX_T_, state_->get_ndx() + np_).setZero();
  }
  syncDataParameters(d);
  if (autodiff_) {
    START_PROFILER("ActionModelCodeGen::calcDiff_T::Jacobian");
    d->J2_T = calcDiffFun_T_->Jacobian(d->X2_T);
    STOP_PROFILER("ActionModelCodeGen::calcDiff_T::Jacobian");
    d->set_D2_T(this);
    START_PROFILER("ActionModelCodeGen::calcDiff_T::ForwardZeroCost");
    calcDiffCostFun_T_->ForwardZero(d->X2_T, d->Yh_T);
    STOP_PROFILER("ActionModelCodeGen::calcDiff_T::ForwardZeroCost");
    d->set_Yh_hessian_T(this);
  } else {
    START_PROFILER("ActionModelCodeGen::calcDiff_T::ForwardZero");
    calcDiffFun_T_->ForwardZero(d->X_T, d->Y2_T);
    STOP_PROFILER("ActionModelCodeGen::calcDiff_T::ForwardZero");
    d->set_Y2_T(this);
  }
  STOP_PROFILER("ActionModelCodeGen::calcDiff_T");
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar>>
ActionModelCodeGenTpl<Scalar>::createData() {
  return createData(std::shared_ptr<ParameterDataManagerTpl<Scalar>>());
}

template <typename Scalar>
std::shared_ptr<ActionDataAbstractTpl<Scalar>>
ActionModelCodeGenTpl<Scalar>::createData(
    const std::shared_ptr<ParameterDataManagerTpl<Scalar>>& params_data) {
  const std::shared_ptr<ActionDataAbstract>& data =
      std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
  std::static_pointer_cast<Data>(data)->params_data = params_data;
  enableMultithreading() = true;  // This enables multithreading in Python
  return data;
}

template <typename Scalar>
bool ActionModelCodeGenTpl<Scalar>::checkData(
    const std::shared_ptr<ActionDataAbstract>& data) {
  return std::dynamic_pointer_cast<Data>(data) != nullptr;
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::syncDataParameters(Data* const data) const {
  if (data->params_data == nullptr) {
    return;
  }
  if (data->params_data->params == nullptr ||
      data->params_data->params->np != np_) {
    throw_pretty(
        "Invalid argument: shared parameter data has wrong dimension "
        "(it should be " +
        std::to_string(np_) + ")");
  }
  data->X.tail(np_) = data->params_data->params->p;
  data->X_T.tail(np_) = data->params_data->params->p;
  data->X2.segment(state_->get_nx() + nu_, np_) = data->params_data->params->p;
  data->X2_T.segment(state_->get_nx(), np_) = data->params_data->params->p;
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::quasiStatic(
    const std::shared_ptr<ActionDataAbstract>& data, Eigen::Ref<VectorXs> u,
    const Eigen::Ref<const VectorXs>& x, const std::size_t /*maxiter = 100*/,
    const Scalar /*tol*/) {
  START_PROFILER("ActionModelCodeGen::quasiStatic");
  Data* d = static_cast<Data*>(data.get());
  d->X3 = x;
  START_PROFILER("ActionModelCodeGen::quasiStatic::ForwardZero");
  quasiStaticFun_->ForwardZero(d->X3, d->Y3);
  STOP_PROFILER("ActionModelCodeGen::quasiStatic::ForwardZero");
  u = Eigen::Map<VectorXs>(d->Y3.data(), nu_);
  STOP_PROFILER("ActionModelCodeGen::quasiStatic");
}

template <typename Scalar>
template <typename NewScalar>
ActionModelCodeGenTpl<NewScalar> ActionModelCodeGenTpl<Scalar>::cast() const {
  typedef ActionModelCodeGenTpl<NewScalar> ReturnType;
  typedef typename ReturnType::ADScalar ADNewScalar;
  ReturnType ret(model_->template cast<NewScalar>(), lib_fname_, autodiff_, np_,
                 cast_function<ADScalar, ADNewScalar>(updateParams_),
                 compiler_type_, compile_options_);
  return ret;
}

template <typename Scalar>
const std::shared_ptr<ActionModelAbstractTpl<Scalar>>&
ActionModelCodeGenTpl<Scalar>::get_model() const {
  return model_;
}

template <typename Scalar>
std::size_t ActionModelCodeGenTpl<Scalar>::get_np() const {
  return np_;
}

template <typename Scalar>
std::size_t ActionModelCodeGenTpl<Scalar>::get_ng() const {
  return model_->get_ng();
}

template <typename Scalar>
std::size_t ActionModelCodeGenTpl<Scalar>::get_nh() const {
  return model_->get_nh();
}

template <typename Scalar>
std::size_t ActionModelCodeGenTpl<Scalar>::get_ng_T() const {
  return model_->get_ng_T();
}

template <typename Scalar>
std::size_t ActionModelCodeGenTpl<Scalar>::get_nh_T() const {
  return model_->get_nh_T();
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
ActionModelCodeGenTpl<Scalar>::get_g_lb() const {
  return model_->get_g_lb();
}

template <typename Scalar>
const typename MathBaseTpl<Scalar>::VectorXs&
ActionModelCodeGenTpl<Scalar>::get_g_ub() const {
  return model_->get_g_ub();
}

template <typename Scalar>
std::size_t ActionModelCodeGenTpl<Scalar>::get_nX() const {
  return nX_;
}

template <typename Scalar>
std::size_t ActionModelCodeGenTpl<Scalar>::get_nX_T() const {
  return nX_T_;
}

template <typename Scalar>
std::size_t ActionModelCodeGenTpl<Scalar>::get_nX2() const {
  return nX2_;
}

template <typename Scalar>
std::size_t ActionModelCodeGenTpl<Scalar>::get_nX2_T() const {
  return nX2_T_;
}

template <typename Scalar>
std::size_t ActionModelCodeGenTpl<Scalar>::get_nX3() const {
  return nX3_;
}

template <typename Scalar>
std::size_t ActionModelCodeGenTpl<Scalar>::get_nY1() const {
  return nY1_;
}

template <typename Scalar>
std::size_t ActionModelCodeGenTpl<Scalar>::get_nY1_T() const {
  return nY1_T_;
}

template <typename Scalar>
std::size_t ActionModelCodeGenTpl<Scalar>::get_nY2() const {
  return nY2_;
}

template <typename Scalar>
std::size_t ActionModelCodeGenTpl<Scalar>::get_nY2_T() const {
  return nY2_T_;
}

template <typename Scalar>
std::size_t ActionModelCodeGenTpl<Scalar>::get_nYh() const {
  return nYh_;
}

template <typename Scalar>
std::size_t ActionModelCodeGenTpl<Scalar>::get_nYh_T() const {
  return nYh_T_;
}

template <typename Scalar>
std::size_t ActionModelCodeGenTpl<Scalar>::get_nY3() const {
  return nY3_;
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::print(std::ostream& os) const {
  model_->print(os);
}

template <typename Scalar>
ActionModelCodeGenTpl<Scalar>::ActionModelCodeGenTpl()
    : model_(nullptr),
      np_(0),
      lib_fname_(""),
      compiler_type_(defaultCompilerType()),
      compile_options_("-O -ffast-math -march=native"),
      updateParams_(EmptyParamsEnv) {
  // Add initialization logic if necessary
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::recordCalc() {
  const std::size_t nx = state_->get_nx();
  // Define the calc's input as the independent variables
  CppAD::Independent(ad_X_);
  // Record the calc's environment variables
  recordParams(ad_X_.tail(np_));
  // Collect computation in calc
  ad_model_->calc(ad_data_, ad_X_.head(nx), ad_X_.segment(nx, nu_));
  tapeCalcOutput();
  // Define calc's output as the dependent variable
  ad_calc_->Dependent(ad_X_, ad_Y1_);
  ad_calc_->optimize("no_compare_op");
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::recordCalc_T() {
  const std::size_t nx = state_->get_nx();
  // Define the calc's input as the independent variables
  CppAD::Independent(ad_X_T_);
  // Record the calc's environment variables
  recordParams(ad_X_T_.tail(np_));
  // Collect computation in calc
  ad_model_->calc(ad_data_, ad_X_T_.head(nx));
  tapeCalcOutput_T();
  // Define calc's output as the dependent variable
  ad_calc_T_->Dependent(ad_X_T_, ad_Y1_T_);
  ad_calc_T_->optimize("no_compare_op");
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::recordCalcDiff() {
  const std::size_t nx = state_->get_nx();
  const std::size_t ndx = state_->get_ndx();
  if (autodiff_) {
    CppAD::Independent(ad_X2_);
    const Eigen::VectorBlock<ADVectorXs> dx = ad_X2_.segment(nX_, ndx);
    const Eigen::VectorBlock<ADVectorXs> du = ad_X2_.segment(nX_ + ndx, nu_);
    const Eigen::VectorBlock<ADVectorXs> dp =
        ad_X2_.segment(nX_ + ndx + nu_, np_);

    ADVectorXs x_pert(nx);
    ADVectorXs u_pert(nu_);
    ADVectorXs p_pert(np_);
    ADVectorXs dxnext(ndx);

    recordParams(ad_data_, ad_X2_.segment(nx + nu_, np_));
    ad_model_->calc(ad_data_, ad_X2_.head(nx), ad_X2_.segment(nx, nu_));
    ad_model_->get_state()->integrate(ad_X2_.head(nx), dx, x_pert);
    u_pert = ad_X2_.segment(nx, nu_) + du;
    p_pert = ad_X2_.segment(nx + nu_, np_) + dp;
    recordParams(ad_data_pert_, p_pert);
    ad_model_->calc(ad_data_pert_, x_pert, u_pert);
    ad_model_->get_state()->diff(ad_data_->xnext, ad_data_pert_->xnext, dxnext);

    Eigen::DenseIndex it_Y2 = 0;
    ad_Y2_[it_Y2] = ad_data_pert_->cost;
    it_Y2 += 1;
    ad_Y2_.segment(it_Y2, ndx) = dxnext;
    it_Y2 += ndx;
    ad_Y2_.segment(it_Y2, ng_) = ad_data_pert_->g.head(ng_);
    it_Y2 += ng_;
    ad_Y2_.segment(it_Y2, nh_) = ad_data_pert_->h.head(nh_);
    ad_calcDiff_->Dependent(ad_X2_, ad_Y2_);
    ad_calcDiff_->optimize("no_compare_op");
    return;
  }
  // Define the calcDiff's input as the independent variables
  CppAD::Independent(ad_X_);
  // Record the calcDiff's environment variables
  recordParams(ad_X_.tail(np_));
  // Collect computation in calcDiff
  ad_model_->calc(ad_data_, ad_X_.head(nx), ad_X_.segment(nx, nu_));
  ad_model_->calcDiff(ad_data_, ad_X_.head(nx), ad_X_.segment(nx, nu_));
  tapeCalcDiffOutput();
  // Define calcDiff's output as the dependent variable
  ad_calcDiff_->Dependent(ad_X_, ad_Y2_);
  ad_calcDiff_->optimize("no_compare_op");
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::recordCalcDiff_T() {
  const std::size_t nx = state_->get_nx();
  const std::size_t ndx = state_->get_ndx();
  if (autodiff_) {
    CppAD::Independent(ad_X2_T_);
    const Eigen::VectorBlock<ADVectorXs> dx = ad_X2_T_.segment(nX_T_, ndx);
    const Eigen::VectorBlock<ADVectorXs> dp =
        ad_X2_T_.segment(nX_T_ + ndx, np_);

    ADVectorXs x_pert(nx);
    ADVectorXs p_pert(np_);
    ADVectorXs dxnext(ndx);

    recordParams(ad_data_, ad_X2_T_.segment(nx, np_));
    ad_model_->calc(ad_data_, ad_X2_T_.head(nx));
    ad_model_->get_state()->integrate(ad_X2_T_.head(nx), dx, x_pert);
    p_pert = ad_X2_T_.segment(nx, np_) + dp;
    recordParams(ad_data_pert_, p_pert);
    ad_model_->calc(ad_data_pert_, x_pert);
    dxnext = dx;

    Eigen::DenseIndex it_Y2 = 0;
    ad_Y2_T_[it_Y2] = ad_data_pert_->cost;
    it_Y2 += 1;
    ad_Y2_T_.segment(it_Y2, ndx) = dxnext;
    it_Y2 += ndx;
    ad_Y2_T_.segment(it_Y2, ng_T_) = ad_data_pert_->g.head(ng_T_);
    it_Y2 += ng_T_;
    ad_Y2_T_.segment(it_Y2, nh_T_) = ad_data_pert_->h.head(nh_T_);
    ad_calcDiff_T_->Dependent(ad_X2_T_, ad_Y2_T_);
    ad_calcDiff_T_->optimize("no_compare_op");
    return;
  }
  // Define the calcDiff's input as the independent variables
  CppAD::Independent(ad_X_T_);
  // Record the calcDiff's environment variables
  recordParams(ad_X_T_.tail(np_));
  // Collect computation in calcDiff
  ad_model_->calc(ad_data_, ad_X_T_.head(nx));
  ad_model_->calcDiff(ad_data_, ad_X_T_.head(nx));
  tapeCalcDiffOutput_T();
  // Define calcDiff's output as the dependent variable
  ad_calcDiff_T_->Dependent(ad_X_T_, ad_Y2_T_);
  ad_calcDiff_T_->optimize("no_compare_op");
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::tapeCalcDiffHessianOutput(ADVectorXs& y) {
  const std::size_t ndx = state_->get_ndx();
  Eigen::DenseIndex it_y = 0;
  Eigen::Map<ADMatrixXs>(y.data() + it_y, ndx, ndx) = ad_data_->Lxx;
  it_y += ndx * ndx;
  Eigen::Map<ADMatrixXs>(y.data() + it_y, ndx, nu_) = ad_data_->Lxu;
  it_y += ndx * nu_;
  Eigen::Map<ADMatrixXs>(y.data() + it_y, nu_, nu_) = ad_data_->Luu;
  it_y += nu_ * nu_;
  Eigen::Map<ADMatrixXs> Lpp_map(y.data() + it_y, np_, np_);
  if (static_cast<std::size_t>(ad_data_->Lpp.rows()) == np_ &&
      static_cast<std::size_t>(ad_data_->Lpp.cols()) == np_) {
    Lpp_map = ad_data_->Lpp;
  } else {
    Lpp_map.setZero();
  }
  it_y += np_ * np_;
  Eigen::Map<ADMatrixXs> Lpx_map(y.data() + it_y, np_, ndx);
  if (static_cast<std::size_t>(ad_data_->Lpx.rows()) == np_) {
    Lpx_map = ad_data_->Lpx;
  } else {
    Lpx_map.setZero();
  }
  it_y += np_ * ndx;
  Eigen::Map<ADMatrixXs> Lpu_map(y.data() + it_y, np_, nu_);
  if (static_cast<std::size_t>(ad_data_->Lpu.rows()) == np_) {
    Lpu_map = ad_data_->Lpu;
  } else {
    Lpu_map.setZero();
  }
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::tapeCalcDiffHessianOutput_T(ADVectorXs& y) {
  const std::size_t ndx = state_->get_ndx();
  Eigen::DenseIndex it_y = 0;
  Eigen::Map<ADMatrixXs>(y.data() + it_y, ndx, ndx) = ad_data_->Lxx;
  it_y += ndx * ndx;
  Eigen::Map<ADMatrixXs> Lpp_map(y.data() + it_y, np_, np_);
  if (static_cast<std::size_t>(ad_data_->Lpp.rows()) == np_ &&
      static_cast<std::size_t>(ad_data_->Lpp.cols()) == np_) {
    Lpp_map = ad_data_->Lpp;
  } else {
    Lpp_map.setZero();
  }
  it_y += np_ * np_;
  Eigen::Map<ADMatrixXs> Lpx_map(y.data() + it_y, np_, ndx);
  if (static_cast<std::size_t>(ad_data_->Lpx.rows()) == np_) {
    Lpx_map = ad_data_->Lpx;
  } else {
    Lpx_map.setZero();
  }
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::recordCalcDiffCost() {
  const std::size_t nx = state_->get_nx();
  CppAD::Independent(ad_X2_);
  ADVectorXs y(nYh_);
  y.setZero();
  recordParams(ad_data_, ad_X2_.segment(nx + nu_, np_));
  ad_model_->calc(ad_data_, ad_X2_.head(nx), ad_X2_.segment(nx, nu_));
  ad_model_->calcDiff(ad_data_, ad_X2_.head(nx), ad_X2_.segment(nx, nu_));
  tapeCalcDiffHessianOutput(y);
  ad_calcDiffCost_->Dependent(ad_X2_, y);
  ad_calcDiffCost_->optimize("no_compare_op");
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::recordCalcDiffCost_T() {
  const std::size_t nx = state_->get_nx();
  CppAD::Independent(ad_X2_T_);
  ADVectorXs y(nYh_T_);
  y.setZero();
  recordParams(ad_data_, ad_X2_T_.segment(nx, np_));
  ad_model_->calc(ad_data_, ad_X2_T_.head(nx));
  ad_model_->calcDiff(ad_data_, ad_X2_T_.head(nx));
  tapeCalcDiffHessianOutput_T(y);
  ad_calcDiffCost_T_->Dependent(ad_X2_T_, y);
  ad_calcDiffCost_T_->optimize("no_compare_op");
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::recordQuasiStatic() {
  // Define the quasiStatic's input as the independent variables
  CppAD::Independent(ad_X3_);
  // The generated model is used for calc/calcDiff. Taping the underlying
  // quasiStatic path can introduce NaNs for contact models through pseudo
  // inverse branches, so keep this symbol structurally valid and deterministic.
  ad_Y3_.setZero();
  // Define quasiStatic's output as the dependent variable
  ad_quasiStatic_->Dependent(ad_X3_, ad_Y3_);
  ad_quasiStatic_->optimize("no_compare_op");
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::recordParams(
    const Eigen::Ref<const ADVectorXs>& p) {
  recordParams(ad_data_, p);
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::recordParams(
    const std::shared_ptr<ADActionDataAbstract>& data,
    const Eigen::Ref<const ADVectorXs>& p) {
  if (np_ > 0 && ad_model_->get_np() == np_) {
    ad_model_->update_p(data, p);
  }
  updateParams_(ad_model_, p);
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::tapeCalcOutput() {
  Eigen::DenseIndex it_Y1 = 0;
  ad_Y1_[it_Y1] = ad_data_->cost;
  it_Y1 += 1;
  ad_Y1_.segment(it_Y1, state_->get_nx()) = ad_data_->xnext;
  it_Y1 += state_->get_nx();
  ad_Y1_.segment(it_Y1, ng_) = ad_data_->g;
  it_Y1 += ng_;
  ad_Y1_.segment(it_Y1, nh_) = ad_data_->h;
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::tapeCalcOutput_T() {
  Eigen::DenseIndex it_Y1 = 0;
  ad_Y1_T_[it_Y1] = ad_data_->cost;
  it_Y1 += 1;
  ad_Y1_T_.segment(it_Y1, ng_T_) = ad_data_->g;
  it_Y1 += ng_T_;
  ad_Y1_T_.segment(it_Y1, nh_T_) = ad_data_->h;
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::tapeCalcDiffOutput() {
  const std::size_t ndx = state_->get_ndx();
  Eigen::DenseIndex it_Y2 = 0;
  Eigen::Map<ADMatrixXs>(ad_Y2_.data() + it_Y2, ndx, ndx) = ad_data_->Fx;
  it_Y2 += ndx * ndx;
  Eigen::Map<ADMatrixXs>(ad_Y2_.data() + it_Y2, ndx, nu_) = ad_data_->Fu;
  it_Y2 += ndx * nu_;
  Eigen::Map<ADMatrixXs> Fp_map(ad_Y2_.data() + it_Y2, ndx, np_);
  if (static_cast<std::size_t>(ad_data_->Fp.cols()) == np_) {
    Fp_map = ad_data_->Fp;
  } else {
    Fp_map.setZero();
  }
  it_Y2 += ndx * np_;
  Eigen::Map<ADVectorXs>(ad_Y2_.data() + it_Y2, ndx) = ad_data_->Lx;
  it_Y2 += ndx;
  Eigen::Map<ADVectorXs>(ad_Y2_.data() + it_Y2, nu_) = ad_data_->Lu;
  it_Y2 += nu_;
  Eigen::Map<ADVectorXs> Lp_map(ad_Y2_.data() + it_Y2, np_);
  if (static_cast<std::size_t>(ad_data_->Lp.size()) == np_) {
    Lp_map = ad_data_->Lp;
  } else {
    Lp_map.setZero();
  }
  it_Y2 += np_;
  Eigen::Map<ADMatrixXs>(ad_Y2_.data() + it_Y2, ndx, ndx) = ad_data_->Lxx;
  it_Y2 += ndx * ndx;
  Eigen::Map<ADMatrixXs>(ad_Y2_.data() + it_Y2, ndx, nu_) = ad_data_->Lxu;
  it_Y2 += ndx * nu_;
  Eigen::Map<ADMatrixXs>(ad_Y2_.data() + it_Y2, nu_, nu_) = ad_data_->Luu;
  it_Y2 += nu_ * nu_;
  Eigen::Map<ADMatrixXs> Lpp_map(ad_Y2_.data() + it_Y2, np_, np_);
  if (static_cast<std::size_t>(ad_data_->Lpp.rows()) == np_ &&
      static_cast<std::size_t>(ad_data_->Lpp.cols()) == np_) {
    Lpp_map = ad_data_->Lpp;
  } else {
    Lpp_map.setZero();
  }
  it_Y2 += np_ * np_;
  Eigen::Map<ADMatrixXs> Lpx_map(ad_Y2_.data() + it_Y2, np_, ndx);
  if (static_cast<std::size_t>(ad_data_->Lpx.rows()) == np_) {
    Lpx_map = ad_data_->Lpx;
  } else {
    Lpx_map.setZero();
  }
  it_Y2 += np_ * ndx;
  Eigen::Map<ADMatrixXs> Lpu_map(ad_Y2_.data() + it_Y2, np_, nu_);
  if (static_cast<std::size_t>(ad_data_->Lpu.rows()) == np_) {
    Lpu_map = ad_data_->Lpu;
  } else {
    Lpu_map.setZero();
  }
  it_Y2 += np_ * nu_;
  Eigen::Map<ADMatrixXs>(ad_Y2_.data() + it_Y2, ng_, ndx) = ad_data_->Gx;
  it_Y2 += ng_ * ndx;
  Eigen::Map<ADMatrixXs>(ad_Y2_.data() + it_Y2, ng_, nu_) = ad_data_->Gu;
  it_Y2 += ng_ * nu_;
  Eigen::Map<ADMatrixXs> Gp_map(ad_Y2_.data() + it_Y2, ng_, np_);
  if (static_cast<std::size_t>(ad_data_->Gp.cols()) == np_) {
    Gp_map = ad_data_->Gp;
  } else {
    Gp_map.setZero();
  }
  it_Y2 += ng_ * np_;
  Eigen::Map<ADMatrixXs>(ad_Y2_.data() + it_Y2, nh_, ndx) = ad_data_->Hx;
  it_Y2 += nh_ * ndx;
  Eigen::Map<ADMatrixXs>(ad_Y2_.data() + it_Y2, nh_, nu_) = ad_data_->Hu;
  it_Y2 += nh_ * nu_;
  Eigen::Map<ADMatrixXs> Hp_map(ad_Y2_.data() + it_Y2, nh_, np_);
  if (static_cast<std::size_t>(ad_data_->Hp.cols()) == np_) {
    Hp_map = ad_data_->Hp;
  } else {
    Hp_map.setZero();
  }
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::tapeCalcDiffOutput_T() {
  const std::size_t ndx = state_->get_ndx();
  Eigen::DenseIndex it_Y2 = 0;
  Eigen::Map<ADVectorXs>(ad_Y2_T_.data() + it_Y2, ndx) = ad_data_->Lx;
  it_Y2 += ndx;
  Eigen::Map<ADVectorXs> Lp_map(ad_Y2_T_.data() + it_Y2, np_);
  if (static_cast<std::size_t>(ad_data_->Lp.size()) == np_) {
    Lp_map = ad_data_->Lp;
  } else {
    Lp_map.setZero();
  }
  it_Y2 += np_;
  Eigen::Map<ADMatrixXs>(ad_Y2_T_.data() + it_Y2, ndx, ndx) = ad_data_->Lxx;
  it_Y2 += ndx * ndx;
  Eigen::Map<ADMatrixXs> Lpp_map(ad_Y2_T_.data() + it_Y2, np_, np_);
  if (static_cast<std::size_t>(ad_data_->Lpp.rows()) == np_ &&
      static_cast<std::size_t>(ad_data_->Lpp.cols()) == np_) {
    Lpp_map = ad_data_->Lpp;
  } else {
    Lpp_map.setZero();
  }
  it_Y2 += np_ * np_;
  Eigen::Map<ADMatrixXs> Lpx_map(ad_Y2_T_.data() + it_Y2, np_, ndx);
  if (static_cast<std::size_t>(ad_data_->Lpx.rows()) == np_) {
    Lpx_map = ad_data_->Lpx;
  } else {
    Lpx_map.setZero();
  }
  it_Y2 += np_ * ndx;
  Eigen::Map<ADMatrixXs>(ad_Y2_T_.data() + it_Y2, ng_T_, ndx) = ad_data_->Gx;
  it_Y2 += ng_T_ * ndx;
  Eigen::Map<ADMatrixXs> Gp_map(ad_Y2_T_.data() + it_Y2, ng_T_, np_);
  if (static_cast<std::size_t>(ad_data_->Gp.cols()) == np_) {
    Gp_map = ad_data_->Gp;
  } else {
    Gp_map.setZero();
  }
  it_Y2 += ng_T_ * np_;
  Eigen::Map<ADMatrixXs>(ad_Y2_T_.data() + it_Y2, nh_T_, ndx) = ad_data_->Hx;
  it_Y2 += nh_T_ * ndx;
  Eigen::Map<ADMatrixXs> Hp_map(ad_Y2_T_.data() + it_Y2, nh_T_, np_);
  if (static_cast<std::size_t>(ad_data_->Hp.cols()) == np_) {
    Hp_map = ad_data_->Hp;
  } else {
    Hp_map.setZero();
  }
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::EmptyParamsEnv(
    std::shared_ptr<ADBase>, const Eigen::Ref<const ADVectorXs>&) {}

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_CODEGEN_ACTION_HXX_
