
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, INRIA, University of Edinburgh,
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
           model->get_ng(), model->get_nh()),
      model_(model),
      ad_model_(model->template cast<ADScalar>()),
      ad_data_(ad_model_->createData()),
      autodiff_(autodiff),
      np_(np),
      nX_(state_->get_nx() + nu_ + np_),
      nX_T_(state_->get_nx() + np_),
      nX3_(state_->get_nx()),
      nY1_(1 + state_->get_nx() + ng_ + nh_),
      nY1_T_(1 + ng_T_ + nh_T_),
      nY3_(nu_),
      ad_X_(nX_),
      ad_X_T_(nX_T_),
      ad_X3_(nX3_),
      ad_Y1_(nY1_),
      ad_Y1_T_(nY1_T_),
      ad_Y3_(nY3_),
      Y1fun_name_("calc"),
      Y1Tfun_name_("calc_T"),
      Y2fun_name_("calcDiff"),
      Y2Tfun_name_("calcDiff_T"),
      Y3fun_name_("quasiStatic"),
      lib_fname_(lib_fname),
      compiler_type_(compiler),
      compile_options_(compile_options),
      updateParams_(updateParams),
      ad_calc_(std::make_unique<ADFun>()),
      ad_calc_T_(std::make_unique<ADFun>()),
      ad_calcDiff_(std::make_unique<ADFun>()),
      ad_calcDiff_T_(std::make_unique<ADFun>()),
      ad_quasiStatic_(std::make_unique<ADFun>()) {
  const std::size_t ndx = state_->get_ndx();
  nY2_ = 2 * ndx * ndx + 2 * ndx * nu_ + nu_ * nu_ + ndx +
         nu_;                                     // cost and dynamics
  nY2_ += ng_ * (ndx + nu_) + nh_ * (ndx + nu_);  // constraints
  nY2_T_ = ndx * ndx + ndx;                       // cost and dynamics
  nY2_T_ += (ng_T_ + nh_T_) * ndx;                // constraints
  ad_Y2_.resize(nY2_);
  ad_Y2_T_.resize(nY2_T_);
  initLib();
  compileLib();
  loadLib(lib_fname_);
  wCostHess_ = VectorXs::Zero(nY1_);
  wCostHess_(0) = Scalar(1.);
  wCostHess_T_ = VectorXs::Zero(nY1_T_);
  wCostHess_T_(0) = Scalar(1.);
}

template <typename Scalar>
ActionModelCodeGenTpl<Scalar>::ActionModelCodeGenTpl(
    std::shared_ptr<ADBase> ad_model, const std::string& lib_fname,
    bool autodiff, const std::size_t np, ParamsEnvironment updateParams,
    CompilerType compiler, const std::string& compile_options)
    : Base(ad_model->get_state()->template cast<Scalar>(), ad_model->get_nu(),
           ad_model->get_nr(), ad_model->get_ng(), ad_model->get_nh()),
      model_(ad_model->template cast<Scalar>()),
      ad_model_(ad_model),
      ad_data_(ad_model_->createData()),
      autodiff_(autodiff),
      np_(np),
      nX_(state_->get_nx() + nu_ + np_),
      nX_T_(state_->get_nx() + np_),
      nX3_(state_->get_nx()),
      nY1_(1 + state_->get_nx() + ng_ + nh_),
      nY1_T_(1 + ng_T_ + nh_T_),
      nY3_(nu_),
      ad_X_(nX_),
      ad_X_T_(nX_T_),
      ad_X3_(nX3_),
      ad_Y1_(nY1_),
      ad_Y1_T_(nY1_T_),
      ad_Y3_(nY3_),
      Y1fun_name_("calc"),
      Y1Tfun_name_("calc_T"),
      Y2fun_name_("calcDiff"),
      Y2Tfun_name_("calcDiff_T"),
      Y3fun_name_("quasiStatic"),
      lib_fname_(lib_fname),
      compiler_type_(compiler),
      compile_options_(compile_options),
      updateParams_(updateParams),
      ad_calc_(std::make_unique<ADFun>()),
      ad_calc_T_(std::make_unique<ADFun>()),
      ad_calcDiff_(std::make_unique<ADFun>()),
      ad_calcDiff_T_(std::make_unique<ADFun>()),
      ad_quasiStatic_(std::make_unique<ADFun>()) {
  const std::size_t ndx = state_->get_ndx();
  nY2_ = 2 * ndx * ndx + 2 * ndx * nu_ + nu_ * nu_ + ndx +
         nu_;                                     // cost and dynamics
  nY2_ += ng_ * (ndx + nu_) + nh_ * (ndx + nu_);  // constraints
  nY2_T_ = ndx * ndx + ndx;                       // cost and dynamics
  nY2_T_ += (ng_T_ + nh_T_) * ndx;                // constraints
  ad_Y2_.resize(nY2_);
  ad_Y2_T_.resize(nY2_T_);
  initLib();
  compileLib();
  loadLib(lib_fname_);
  wCostHess_ = VectorXs::Zero(nY1_);
  wCostHess_(0) = Scalar(1.);
  wCostHess_T_ = VectorXs::Zero(nY1_T_);
  wCostHess_T_(0) = Scalar(1.);
}

template <typename Scalar>
ActionModelCodeGenTpl<Scalar>::ActionModelCodeGenTpl(
    const ActionModelCodeGenTpl<Scalar>& other)
    : Base(other),
      model_(other.model_),
      ad_model_(other.ad_model_),
      autodiff_(other.autodiff_),
      np_(other.np_),
      nX_(other.nX_),
      nX_T_(other.nX_T_),
      nX3_(other.nX3_),
      nY1_(other.nY1_),
      nY1_T_(other.nY1_T_),
      nY2_(other.nY2_),
      nY2_T_(other.nY2_T_),
      nY3_(other.nY3_),
      ad_X_(other.nX_),
      ad_X_T_(other.nX_T_),
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
      Y3fun_name_(other.Y3fun_name_),
      lib_fname_(other.lib_fname_),
      compiler_type_(other.compiler_type_),
      compile_options_(other.compile_options_),
      updateParams_(other.updateParams_),
      ad_calc_(clone_adfun(*other.ad_calc_)),
      ad_calc_T_(clone_adfun(*other.ad_calc_T_)),
      ad_calcDiff_(clone_adfun(*other.ad_calcDiff_)),
      ad_calcDiff_T_(clone_adfun(*other.ad_calcDiff_T_)),
      ad_quasiStatic_(clone_adfun(*other.ad_quasiStatic_)),
      calcCG_(std::make_unique<CSourceGen>(*ad_calc_, Y1fun_name_)),
      calcCG_T_(std::make_unique<CSourceGen>(*ad_calc_T_, Y1Tfun_name_)),
      calcDiffCG_(std::make_unique<CSourceGen>(*ad_calcDiff_, Y2fun_name_)),
      calcDiffCG_T_(
          std::make_unique<CSourceGen>(*ad_calcDiff_T_, Y2Tfun_name_)),
      quasiStaticCG_(
          std::make_unique<CSourceGen>(*ad_quasiStatic_, Y3fun_name_)),
      libCG_(std::make_unique<LibraryCSourceGen>(
          *calcCG_, *calcCG_T_, *calcDiffCG_, *calcDiffCG_T_, *quasiStaticCG_)),
      dynLibManager_(
          std::make_unique<LibraryProcessor>(*other.libCG_, lib_fname_)) {
  loadLib(lib_fname_);
}

template <typename Scalar>
ActionModelCodeGenTpl<Scalar>::ActionModelCodeGenTpl(
    const std::string& lib_fname, std::shared_ptr<Base> model)
    : Base(model->get_state()->template cast<Scalar>(), model->get_nu(),
           model->get_nr(), model->get_ng(), model->get_nh()),
      model_(model),
      Y1fun_name_("calc"),
      Y1Tfun_name_("calc_T"),
      Y2fun_name_("calcDiff"),
      Y2Tfun_name_("calcDiff_T"),
      Y3fun_name_("quasiStatic"),
      lib_fname_(lib_fname),
      ad_calc_(std::make_unique<ADFun>()),
      ad_calc_T_(std::make_unique<ADFun>()),
      ad_calcDiff_(std::make_unique<ADFun>()),
      ad_calcDiff_T_(std::make_unique<ADFun>()),
      ad_quasiStatic_(std::make_unique<ADFun>()) {
  loadLib(lib_fname_);
  nX_ = calcFun_->Domain();
  nX_T_ = calcFun_T_->Domain();
  nY1_ = calcFun_->Range();
  nY1_T_ = calcFun_T_->Range();
  nY2_ = calcDiffFun_->Range();
  nY2_T_ = calcDiffFun_T_->Range();
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
  ad_X3_.resize(nX3_);
  ad_Y1_.resize(nY1_);
  ad_Y1_T_.resize(nY1_T_);
  ad_Y2_.resize(nY2_);
  ad_Y2_T_.resize(nY2_T_);
  ad_Y3_.resize(nY3_);
  autodiff_ = calcFun_->isJacobianAvailable() && calcFun_->isHessianAvailable();
  wCostHess_ = VectorXs::Zero(nY1_);
  wCostHess_(0) = Scalar(1.);
  wCostHess_T_ = VectorXs::Zero(nY1_T_);
  wCostHess_T_(0) = Scalar(1.);
}

template <typename Scalar>
ActionModelCodeGenTpl<Scalar>::ActionModelCodeGenTpl(
    const std::string& lib_fname, std::shared_ptr<ADBase> ad_model)
    : Base(ad_model->get_state()->template cast<Scalar>(), ad_model->get_nu(),
           ad_model->get_nr(), ad_model->get_ng(), ad_model->get_nh()),
      model_(ad_model->template cast<Scalar>()),
      Y1fun_name_("calc"),
      Y1Tfun_name_("calc_T"),
      Y2fun_name_("calcDiff"),
      Y2Tfun_name_("calcDiff_T"),
      Y3fun_name_("quasiStatic"),
      lib_fname_(lib_fname),
      ad_calc_(std::make_unique<ADFun>()),
      ad_calc_T_(std::make_unique<ADFun>()),
      ad_calcDiff_(std::make_unique<ADFun>()),
      ad_calcDiff_T_(std::make_unique<ADFun>()),
      ad_quasiStatic_(std::make_unique<ADFun>()) {
  loadLib(lib_fname_);
  nX_ = calcFun_->Domain();
  nX_T_ = calcFun_T_->Domain();
  nY1_ = calcFun_->Range();
  nY1_T_ = calcFun_T_->Range();
  nY2_ = calcDiffFun_->Range();
  nY2_T_ = calcDiffFun_T_->Range();
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
  ad_X3_.resize(nX3_);
  ad_Y1_.resize(nY1_);
  ad_Y1_T_.resize(nY1_T_);
  ad_Y2_.resize(nY2_);
  ad_Y2_T_.resize(nY2_T_);
  ad_Y3_.resize(nY3_);
  autodiff_ = calcFun_->isJacobianAvailable() && calcFun_->isHessianAvailable();
  wCostHess_ = VectorXs::Zero(nY1_);
  wCostHess_(0) = Scalar(1.);
  wCostHess_T_ = VectorXs::Zero(nY1_T_);
  wCostHess_T_(0) = Scalar(1.);
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::initLib() {
  START_PROFILER("ActionModelCodeGen::initLib");
  // Generate source code for calc
  recordCalc();
  calcCG_ =
      std::unique_ptr<CSourceGen>(new CSourceGen(*ad_calc_.get(), Y1fun_name_));
  calcCG_->setCreateForwardZero(true);
  calcCG_->setCreateJacobian(autodiff_);
  calcCG_->setCreateHessian(autodiff_);
  // Generate source code for calc in terminal nodes
  recordCalc_T();
  calcCG_T_ = std::unique_ptr<CSourceGen>(
      new CSourceGen(*ad_calc_T_.get(), Y1Tfun_name_));
  calcCG_T_->setCreateForwardZero(true);
  calcCG_T_->setCreateJacobian(autodiff_);
  calcCG_T_->setCreateHessian(autodiff_);
  // Generate source code for calcDiff
  recordCalcDiff();
  calcDiffCG_ = std::unique_ptr<CSourceGen>(
      new CSourceGen(*ad_calcDiff_.get(), Y2fun_name_));
  calcDiffCG_->setCreateForwardZero(!autodiff_);
  calcDiffCG_->setCreateJacobian(false);
  // Generate source code for calcDiff in terminal nodes
  recordCalcDiff_T();
  calcDiffCG_T_ = std::unique_ptr<CSourceGen>(
      new CSourceGen(*ad_calcDiff_T_.get(), Y2Tfun_name_));
  calcDiffCG_T_->setCreateForwardZero(!autodiff_);
  calcDiffCG_T_->setCreateJacobian(false);
  // Generate source code for quasiStatic
  recordQuasiStatic();
  quasiStaticCG_ = std::unique_ptr<CSourceGen>(
      new CSourceGen(*ad_quasiStatic_.get(), Y3fun_name_));
  quasiStaticCG_->setCreateForwardZero(true);
  quasiStaticCG_->setCreateJacobian(false);
  // Generate library for calc and calcDiff
  libCG_ = std::unique_ptr<LibraryCSourceGen>(new LibraryCSourceGen(
      *calcCG_, *calcCG_T_, *calcDiffCG_, *calcDiffCG_T_, *quasiStaticCG_));
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
  switch (compiler_type_) {
    case GCC: {
      CppAD::cg::GccCompiler<Scalar> compiler("/usr/bin/gcc");
      std::vector<std::string> compile_flags = compiler.getCompileFlags();
      compile_flags[0] = compile_options_;
      compiler.setCompileFlags(compile_flags);
      dynLibManager_->createDynamicLibrary(compiler, false);
      break;
    }
    case CLANG: {
      CppAD::cg::ClangCompiler<Scalar> compiler("/usr/bin/clang");
      std::vector<std::string> compile_flags = compiler.getCompileFlags();
      compile_flags[0] = compile_options_;
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
  quasiStaticFun_ = dynLib_->model(Y3fun_name_.c_str());
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::update_p(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& p) const {
  if (static_cast<std::size_t>(p.size()) != np_) {
    throw_pretty(
        "Invalid argument: " << "p has wrong dimension (it should be " +
                                    std::to_string(np_) + ")");
  }
  Data* d = static_cast<Data*>(data.get());
  d->X.tail(np_) = p;
  d->X_T.tail(np_) = p;
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::calc(
    const std::shared_ptr<ActionDataAbstract>& data,
    const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u) {
  START_PROFILER("ActionModelCodeGen::calc");
  Data* d = static_cast<Data*>(data.get());
  const std::size_t nx = state_->get_nx();
  d->X.head(nx) = x;
  d->X.segment(nx, nu_) = u;
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
  const std::size_t nx = state_->get_nx();
  d->X_T.head(nx) = x;
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
  const std::size_t nx = state_->get_nx();
  d->X.head(nx) = x;
  d->X.segment(nx, nu_) = u;
  if (autodiff_) {
    START_PROFILER("ActionModelCodeGen::calcDiff::Jacobian");
    d->J1 = calcFun_->Jacobian(d->X);
    STOP_PROFILER("ActionModelCodeGen::calcDiff::Jacobian");
    START_PROFILER("ActionModelCodeGen::calcDiff::Hessian");
    d->H1 = calcFun_->Hessian(d->X, wCostHess_);
    STOP_PROFILER("ActionModelCodeGen::calcDiff::Hessian");
    d->set_D1(this);
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
  const std::size_t nx = state_->get_nx();
  d->X_T.head(nx) = x;
  if (autodiff_) {
    START_PROFILER("ActionModelCodeGen::calcDiff_T::Jacobian");
    d->J1_T = calcFun_T_->Jacobian(d->X_T);
    STOP_PROFILER("ActionModelCodeGen::calcDiff_T::Jacobian");
    START_PROFILER("ActionModelCodeGen::calcDiff_T::Hessian");
    d->H1_T = calcFun_T_->Hessian(d->X_T, wCostHess_T_);
    STOP_PROFILER("ActionModelCodeGen::calcDiff_T::Hessian");
    d->set_D1_T(this);
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
  const std::shared_ptr<ActionDataAbstract>& data =
      std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
  enableMultithreading() = true;  // This enables multithreading in Python
  return data;
}

template <typename Scalar>
bool ActionModelCodeGenTpl<Scalar>::checkData(
    const std::shared_ptr<ActionDataAbstract>& data) {
  return model_->checkData(data);
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
      compiler_type_(CLANG),
      compile_options_("-Ofast -march=native"),
      updateParams_(EmptyParamsEnv) {
  // Add initialization logic if necessary
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::recordCalc() {
  const std::size_t nx = state_->get_nx();
  // Define the calc's input as the independent variables
  CppAD::Independent(ad_X_);
  // Record the calc's environment variables
  updateParams_(ad_model_, ad_X_.tail(np_));
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
  updateParams_(ad_model_, ad_X_T_.tail(np_));
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
  // Define the calcDiff's input as the independent variables
  CppAD::Independent(ad_X_);
  // Record the calcDiff's environment variables
  updateParams_(ad_model_, ad_X_.tail(np_));
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
  // Define the calcDiff's input as the independent variables
  CppAD::Independent(ad_X_T_);
  // Record the calcDiff's environment variables
  updateParams_(ad_model_, ad_X_T_.tail(np_));
  // Collect computation in calcDiff
  ad_model_->calc(ad_data_, ad_X_T_.head(nx));
  ad_model_->calcDiff(ad_data_, ad_X_T_.head(nx));
  tapeCalcDiffOutput_T();
  // Define calcDiff's output as the dependent variable
  ad_calcDiff_T_->Dependent(ad_X_T_, ad_Y2_T_);
  ad_calcDiff_T_->optimize("no_compare_op");
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::recordQuasiStatic() {
  const std::size_t nx = state_->get_nx();
  // Define the quasiStatic's input as the independent variables
  CppAD::Independent(ad_X3_);
  // Collect computation in quasiStatic
  ad_model_->quasiStatic(ad_data_, ad_Y3_, ad_X3_.head(nx), 100);
  // Define quasiStatic's output as the dependent variable
  ad_quasiStatic_->Dependent(ad_X3_, ad_Y3_);
  ad_quasiStatic_->optimize("no_compare_op");
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
  Eigen::Map<ADVectorXs>(ad_Y2_.data() + it_Y2, ndx) = ad_data_->Lx;
  it_Y2 += ndx;
  Eigen::Map<ADVectorXs>(ad_Y2_.data() + it_Y2, nu_) = ad_data_->Lu;
  it_Y2 += nu_;
  Eigen::Map<ADMatrixXs>(ad_Y2_.data() + it_Y2, ndx, ndx) = ad_data_->Lxx;
  it_Y2 += ndx * ndx;
  Eigen::Map<ADMatrixXs>(ad_Y2_.data() + it_Y2, ndx, nu_) = ad_data_->Lxu;
  it_Y2 += ndx * nu_;
  Eigen::Map<ADMatrixXs>(ad_Y2_.data() + it_Y2, nu_, nu_) = ad_data_->Luu;
  it_Y2 += nu_ * nu_;
  Eigen::Map<ADMatrixXs>(ad_Y2_.data() + it_Y2, ng_, ndx) = ad_data_->Gx;
  it_Y2 += ng_ * ndx;
  Eigen::Map<ADMatrixXs>(ad_Y2_.data() + it_Y2, ng_, nu_) = ad_data_->Gu;
  it_Y2 += ng_ * nu_;
  Eigen::Map<ADMatrixXs>(ad_Y2_.data() + it_Y2, nh_, ndx) = ad_data_->Hx;
  it_Y2 += nh_ * ndx;
  Eigen::Map<ADMatrixXs>(ad_Y2_.data() + it_Y2, nh_, nu_) = ad_data_->Hu;
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::tapeCalcDiffOutput_T() {
  const std::size_t ndx = state_->get_ndx();
  Eigen::DenseIndex it_Y2 = 0;
  Eigen::Map<ADVectorXs>(ad_Y2_T_.data() + it_Y2, ndx) = ad_data_->Lx;
  it_Y2 += ndx;
  Eigen::Map<ADMatrixXs>(ad_Y2_T_.data() + it_Y2, ndx, ndx) = ad_data_->Lxx;
  it_Y2 += ndx * ndx;
  Eigen::Map<ADMatrixXs>(ad_Y2_T_.data() + it_Y2, ng_, ndx) = ad_data_->Gx;
  it_Y2 += ng_T_ * ndx;
  Eigen::Map<ADMatrixXs>(ad_Y2_T_.data() + it_Y2, nh_, ndx) = ad_data_->Hx;
}

template <typename Scalar>
void ActionModelCodeGenTpl<Scalar>::EmptyParamsEnv(
    std::shared_ptr<ADBase>, const Eigen::Ref<const ADVectorXs>&) {}
}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_CODEGEN_ACTION_HXX_
