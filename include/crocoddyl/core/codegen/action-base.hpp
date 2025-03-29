
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, INRIA, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_CODEGEN_ACTION_BASE_HPP_
#define CROCODDYL_CORE_CODEGEN_ACTION_BASE_HPP_

#ifdef CROCODDYL_WITH_CODEGEN

#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/utils/stop-watch.hpp"

namespace crocoddyl {

template <typename Scalar>
std::unique_ptr<CppAD::ADFun<CppAD::cg::CG<Scalar>>> clone_adfun(
    const CppAD::ADFun<CppAD::cg::CG<Scalar>>& original) {
  auto cloned = std::make_unique<CppAD::ADFun<CppAD::cg::CG<Scalar>>>();
  *cloned = original;  // Use assignment operator to copy the function
  return cloned;
}

template <typename FromScalar, typename ToScalar>
std::function<
    void(std::shared_ptr<ActionModelAbstractTpl<ToScalar>>,
         const Eigen::Ref<const typename MathBaseTpl<ToScalar>::VectorXs>&)>
cast_function(
    const std::function<void(
        std::shared_ptr<ActionModelAbstractTpl<FromScalar>>,
        const Eigen::Ref<const typename MathBaseTpl<FromScalar>::VectorXs>&)>&
        fn) {
  return [fn](std::shared_ptr<ActionModelAbstractTpl<ToScalar>> to_base,
              const Eigen::Ref<const typename MathBaseTpl<ToScalar>::VectorXs>&
                  to_vector) {
    // Convert arguments
    const std::shared_ptr<ActionModelAbstractTpl<FromScalar>>& from_base =
        to_base->template cast<FromScalar>();
    const typename MathBaseTpl<FromScalar>::VectorXs from_vector =
        to_vector.template cast<FromScalar>();
    // Call the original function with converted arguments
    fn(from_base, from_vector);
  };
}

enum CompilerType { GCC = 0, CLANG };

template <typename Scalar>
struct ActionDataCodeGenTpl;

template <typename _Scalar>
class ActionModelCodeGenTpl : public ActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_FLOATINGPOINT_CAST(ActionModelBase, ActionModelCodeGenTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActionModelAbstractTpl<Scalar> Base;
  typedef ActionDataCodeGenTpl<Scalar> Data;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  typedef CppAD::cg::CG<Scalar> CGScalar;
  typedef CppAD::AD<CGScalar> ADScalar;
  typedef MathBaseTpl<ADScalar> ADMathBase;
  typedef crocoddyl::ActionModelAbstractTpl<ADScalar> ADBase;
  typedef ActionDataAbstractTpl<ADScalar> ADActionDataAbstract;
  typedef typename ADMathBase::VectorXs ADVectorXs;
  typedef typename ADMathBase::MatrixXs ADMatrixXs;
  typedef CppAD::ADFun<CGScalar> ADFun;
  typedef CppAD::cg::ModelCSourceGen<Scalar> CSourceGen;
  typedef CppAD::cg::ModelLibraryCSourceGen<Scalar> LibraryCSourceGen;
  typedef CppAD::cg::DynamicModelLibraryProcessor<Scalar> LibraryProcessor;
  typedef CppAD::cg::DynamicLib<Scalar> DynamicLib;
  typedef CppAD::cg::GenericModel<Scalar> GenericModel;
  typedef CppAD::cg::LinuxDynamicLib<Scalar> LinuxDynamicLib;
  typedef CppAD::cg::system::SystemInfo<> SystemInfo;
  typedef std::function<void(std::shared_ptr<ADBase>,
                             const Eigen::Ref<const ADVectorXs>&)>
      ParamsEnvironment;

  /**
   * @brief Initialize the code generated action model from a model
   *
   * @param[in] model         Action model which we want to code generate
   * @param[in] lib_fname     Name of the code generated library
   * @param[in] updateParams  Function used to update the calc and calcDiff's
   * parameters (default empty function)
   * @param[in] compiler      Type of compiler GCC or CLANG (default: CLANG)
   * @param[in] compile_options  Compilation flags (default: "-Ofast
   * -march=native")
   */
  ActionModelCodeGenTpl(
      std::shared_ptr<Base> model, const std::string& lib_fname,
      const std::size_t nP = 0, ParamsEnvironment updateParams = EmptyParamsEnv,
      CompilerType compiler = CLANG,
      const std::string& compile_options = "-Ofast -march=native")
      : Base(model->get_state(), model->get_nu(), model->get_nr(),
             model->get_ng(), model->get_nh()),
        model_(model),
        ad_model_(model->template cast<ADScalar>()),
        ad_data_(ad_model_->createData()),
        nP_(nP),
        nX_(state_->get_nx() + nu_ + nP_),
        nX_T_(state_->get_nx() + nP_),
        nY1_(1 + state_->get_nx() + ng_ + nh_),
        nY1_T_(1 + ng_T_ + nh_T_),
        ad_X_(nX_),
        ad_X_T_(nX_T_),
        ad_Y1_(nY1_),
        ad_Y1_T_(nY1_T_),
        Y1fun_name_("calc"),
        Y1Tfun_name_("calc_T"),
        Y2fun_name_("calcDiff"),
        Y2Tfun_name_("calcDiff_T"),
        lib_fname_(lib_fname),
        compiler_type_(compiler),
        compile_options_(compile_options),
        updateParams_(updateParams),
        ad_calc_(std::make_unique<ADFun>()),
        ad_calc_T_(std::make_unique<ADFun>()),
        ad_calcDiff_(std::make_unique<ADFun>()),
        ad_calcDiff_T_(std::make_unique<ADFun>()) {
    const std::size_t ndx = state_->get_ndx();
    nY2_ = 2 * ndx * ndx + 2 * ndx * nu_ + nu_ * nu_ + ndx +
           nu_;                                     // cost and dynamics
    nY2_ += ng_ * (ndx + nu_) + nh_ * (ndx + nu_);  // constraints
    nY2_T_ = ndx * ndx + ndx;                       // cost and dynamics
    nY2_T_ += (ng_T_ + nh_T_) * ndx;                // constraints
    ad_Y2_.resize(nY2_);
    ad_Y2_T_.resize(nY2_T_);
    initLib();
    loadLib();
  }

  /**
   * @brief Initialize the code generated action model from an AD model
   *
   * @param[in] ad_model      Action model used to code generate
   * @param[in] lib_fname     Name of the code generated library
   * @param[in] updateParams  Function used to update the calc and calcDiff's
   * parameters (default empty function)
   * @param[in] compiler      Type of compiler GCC or CLANG (default: CLANG)
   * @param[in] compile_options  Compilation flags (default: "-Ofast
   * -march=native")
   */
  ActionModelCodeGenTpl(
      std::shared_ptr<ADBase> ad_model, const std::string& lib_fname,
      const std::size_t nP = 0, ParamsEnvironment updateParams = EmptyParamsEnv,
      CompilerType compiler = CLANG,
      const std::string& compile_options = "-Ofast -march=native")
      : Base(ad_model->get_state()->template cast<Scalar>(), ad_model->get_nu(),
             ad_model->get_nr(), ad_model->get_ng(), ad_model->get_nh()),
        model_(ad_model->template cast<Scalar>()),
        ad_model_(ad_model),
        ad_data_(ad_model_->createData()),
        nP_(nP),
        nX_(state_->get_nx() + nu_ + nP_),
        nX_T_(state_->get_nx() + nP_),
        nY1_(1 + state_->get_nx() + ng_ + nh_),
        nY1_T_(1 + ng_T_ + nh_T_),
        ad_X_(nX_),
        ad_X_T_(nX_T_),
        ad_Y1_(nY1_),
        ad_Y1_T_(nY1_T_),
        Y1fun_name_("calc"),
        Y1Tfun_name_("calc_T"),
        Y2fun_name_("calcDiff"),
        Y2Tfun_name_("calcDiff_T"),
        lib_fname_(lib_fname),
        compiler_type_(compiler),
        compile_options_(compile_options),
        updateParams_(updateParams),
        ad_calc_(std::make_unique<ADFun>()),
        ad_calc_T_(std::make_unique<ADFun>()),
        ad_calcDiff_(std::make_unique<ADFun>()),
        ad_calcDiff_T_(std::make_unique<ADFun>()) {
    const std::size_t ndx = state_->get_ndx();
    nY2_ = 2 * ndx * ndx + 2 * ndx * nu_ + nu_ * nu_ + ndx +
           nu_;                                     // cost and dynamics
    nY2_ += ng_ * (ndx + nu_) + nh_ * (ndx + nu_);  // constraints
    nY2_T_ = ndx * ndx + ndx;                       // cost and dynamics
    nY2_T_ += (ng_T_ + nh_T_) * ndx;                // constraints
    ad_Y2_.resize(nY2_);
    ad_Y2_T_.resize(nY2_T_);
    initLib();
    loadLib();
  }

  /**
   * @brief Copy constructor
   * @param other  Action model to be copied
   */
  ActionModelCodeGenTpl(const ActionModelCodeGenTpl<Scalar>& other)
      : Base(other),
        model_(other.model_),
        ad_model_(other.ad_model_),
        nP_(other.nP_),
        nX_(other.nX_),
        nX_T_(other.nX_T_),
        nY1_(other.nY1_),
        nY1_T_(other.nY1_T_),
        nY2_(other.nY2_),
        nY2_T_(other.nY2_T_),
        ad_X_(other.nX_),
        ad_X_T_(other.nX_T_),
        ad_Y1_(other.nY1_),
        ad_Y1_T_(other.nY1_T_),
        ad_Y2_(other.nY2_),
        ad_Y2_T_(other.nY2_T_),
        Y1fun_name_(other.Y1fun_name_),
        Y1Tfun_name_(other.Y1Tfun_name_),
        Y2fun_name_(other.Y2fun_name_),
        Y2Tfun_name_(other.Y2Tfun_name_),
        lib_fname_(other.lib_fname_),
        compiler_type_(other.compiler_type_),
        compile_options_(other.compile_options_),
        updateParams_(other.updateParams_),
        ad_calc_(clone_adfun(*other.ad_calc_)),
        ad_calc_T_(clone_adfun(*other.ad_calc_T_)),
        ad_calcDiff_(clone_adfun(*other.ad_calcDiff_)),
        ad_calcDiff_T_(clone_adfun(*other.ad_calcDiff_T_)),
        calcCG_(std::make_unique<CSourceGen>(*ad_calc_, Y1fun_name_)),
        calcCG_T_(std::make_unique<CSourceGen>(*ad_calc_T_, Y1Tfun_name_)),
        calcDiffCG_(std::make_unique<CSourceGen>(*ad_calcDiff_, Y2fun_name_)),
        calcDiffCG_T_(
            std::make_unique<CSourceGen>(*ad_calcDiff_T_, Y2Tfun_name_)),
        libCG_(std::make_unique<LibraryCSourceGen>(
            *calcCG_, *calcCG_T_, *calcDiffCG_, *calcDiffCG_T_)),
        dynLibManager_(
            std::make_unique<LibraryProcessor>(*other.libCG_, lib_fname_)) {
    loadLib(false);
  }

  virtual ~ActionModelCodeGenTpl() = default;

  /**
   * @brief Initialize the code-generated library
   */
  void initLib() {
    START_PROFILER("ActionModelCodeGen::initLib");
    // Generate source code for calc
    recordCalc();
    calcCG_ = std::unique_ptr<CSourceGen>(
        new CSourceGen(*ad_calc_.get(), Y1fun_name_));
    calcCG_->setCreateForwardZero(true);
    calcCG_->setCreateJacobian(false);
    // Generate source code for calc in terminal nodes
    recordCalc_T();
    calcCG_T_ = std::unique_ptr<CSourceGen>(
        new CSourceGen(*ad_calc_T_.get(), Y1Tfun_name_));
    calcCG_T_->setCreateForwardZero(true);
    calcCG_T_->setCreateJacobian(false);
    // Generate source code for calcDiff
    recordCalcDiff();
    calcDiffCG_ = std::unique_ptr<CSourceGen>(
        new CSourceGen(*ad_calcDiff_.get(), Y2fun_name_));
    calcDiffCG_->setCreateForwardZero(true);
    calcDiffCG_->setCreateJacobian(false);
    // Generate source code for calcDiff in terminal nodes
    recordCalcDiff_T();
    calcDiffCG_T_ = std::unique_ptr<CSourceGen>(
        new CSourceGen(*ad_calcDiff_T_.get(), Y2Tfun_name_));
    calcDiffCG_T_->setCreateForwardZero(true);
    calcDiffCG_T_->setCreateJacobian(false);
    // Generate library for calc and calcDiff
    libCG_ = std::unique_ptr<LibraryCSourceGen>(new LibraryCSourceGen(
        *calcCG_, *calcCG_T_, *calcDiffCG_, *calcDiffCG_T_));
    // Create dynamic library manager
    dynLibManager_ = std::unique_ptr<LibraryProcessor>(
        new LibraryProcessor(*libCG_, lib_fname_));
    STOP_PROFILER("ActionModelCodeGen::initLib");
  }

  /**
   * @brief Compile the code-generated library
   */
  void compileLib() {
    START_PROFILER("ActionModelCodeGen::compileLib");
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

  /**
   * @brief Check if the code-generated library exists
   *
   * @return Return true if the code-generated library exists, otherwise false.
   */
  bool existLib() const {
    const std::string filename =
        dynLibManager_->getLibraryName() + SystemInfo::DYNAMIC_LIB_EXTENSION;
    std::ifstream file(filename.c_str());
    return file.good();
  }

  /**
   * @brief Load the code-generated library
   *
   * @param generate_if_exist  True for compiling the library when it exists
   */
  void loadLib(const bool generate_if_exist = true) {
    if (!existLib() || generate_if_exist) {
      compileLib();
    }
    const auto it = dynLibManager_->getOptions().find("dlOpenMode");
    const std::string filename =
        dynLibManager_->getLibraryName() + SystemInfo::DYNAMIC_LIB_EXTENSION;
    if (it == dynLibManager_->getOptions().end()) {
      dynLib_.reset(new LinuxDynamicLib(filename));
    } else {
      int dlOpenMode = std::stoi(it->second);
      dynLib_.reset(new LinuxDynamicLib(filename, dlOpenMode));
    }
    calcFun_ = dynLib_->model(Y1fun_name_.c_str());
    calcFun_T_ = dynLib_->model(Y1Tfun_name_.c_str());
    calcDiffFun_ = dynLib_->model(Y2fun_name_.c_str());
    calcDiffFun_T_ = dynLib_->model(Y2Tfun_name_.c_str());
  }

  void set_parameters(const std::shared_ptr<ActionDataAbstract>& data,
                      const Eigen::Ref<const VectorXs>& p) const {
    Data* d = static_cast<Data*>(data.get());
    d->X.tail(nP_) = p;
  }

  /**
   * @brief Compute the next state and cost value using a code-generated library
   *
   * @param[in] data  Action data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override {
    START_PROFILER("ActionModelCodeGen::calc");
    Data* d = static_cast<Data*>(data.get());
    const std::size_t nx = state_->get_nx();
    d->X.head(nx) = x;
    d->X.segment(nx, nu_) = u;
    START_PROFILER("ActionModelCodeGen::calc::ForwardZero");
    calcFun_->ForwardZero(d->X, d->Y1);
    STOP_PROFILER("ActionModelCodeGen::calc::ForwardZero");
    d->set_Y1();
    STOP_PROFILER("ActionModelCodeGen::calc");
  }

  /**
   * @brief Compute the cost value and constraint infeasibilities for terminal
   * nodes using a code-generated library
   */
  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override {
    START_PROFILER("ActionModelCodeGen::calc_T");
    Data* d = static_cast<Data*>(data.get());
    const std::size_t nx = state_->get_nx();
    d->X_T.head(nx) = x;
    START_PROFILER("ActionModelCodeGen::calc_T::ForwardZero");
    calcFun_T_->ForwardZero(d->X_T, d->Y1_T);
    STOP_PROFILER("ActionModelCodeGen::calc_T::ForwardZero");
    d->xnext = x;
    d->set_Y1_T();
    STOP_PROFILER("ActionModelCodeGen::calc_T");
  }

  /**
   * @brief Compute the derivatives of the dynamics and cost functions using a
   * code-generated library
   *
   * In contrast to action models, this code-generated calcDiff doesn't assumes
   * that `calc()` has been run first. This function builds a linear-quadratic
   * approximation of the action model (i.e. dynamical system and cost
   * function).
   *
   * @param[in] data  Action data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override {
    START_PROFILER("ActionModelCodeGen::calcDiff");
    Data* d = static_cast<Data*>(data.get());
    const std::size_t nx = state_->get_nx();
    d->X.head(nx) = x;
    d->X.segment(nx, nu_) = u;
    START_PROFILER("ActionModelCodeGen::calcDiff::ForwardZero");
    calcDiffFun_->ForwardZero(d->X, d->Y2);
    STOP_PROFILER("ActionModelCodeGen::calcDiff::ForwardZero");
    d->set_Y2();
    STOP_PROFILER("ActionModelCodeGen::calcDiff");
  }

  /**
   * @brief Compute the derivatives of cost functions for terminal nodes using a
   * code-generated library
   */
  virtual void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x) override {
    START_PROFILER("ActionModelCodeGen::calcDiff_T");
    Data* d = static_cast<Data*>(data.get());
    const std::size_t nx = state_->get_nx();
    d->X_T.head(nx) = x;
    START_PROFILER("ActionModelCodeGen::calcDiff_T::ForwardZero");
    calcDiffFun_T_->ForwardZero(d->X_T, d->Y2_T);
    STOP_PROFILER("ActionModelCodeGen::calcDiff_T::ForwardZero");
    d->set_Y2_T();
    STOP_PROFILER("ActionModelCodeGen::calcDiff_T");
  }

  /**
   * @brief Create the data for the code-generated action
   *
   * @return the action data
   */
  virtual std::shared_ptr<ActionDataAbstract> createData() override {
    return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
  }

  /**
   * @brief Checks that a specific data belongs to this model
   */
  virtual bool checkData(
      const std::shared_ptr<ActionDataAbstract>& data) override {
    return model_->checkData(data);
  }

  /**
   * @brief Computes the quasic static commands
   *
   * The quasic static commands are the ones produced for a the reference
   * posture as an equilibrium point, i.e. for
   * \f$\mathbf{f^q_x}\delta\mathbf{q}+\mathbf{f_u}\delta\mathbf{u}=\mathbf{0}\f$
   *
   * @param[in] data    Action data
   * @param[out] u      Quasic static commands
   * @param[in] x       State point (velocity has to be zero)
   * @param[in] maxiter Maximum allowed number of iterations
   * @param[in] tol     Tolerance
   */
  virtual void quasiStatic(const std::shared_ptr<ActionDataAbstract>& data,
                           Eigen::Ref<VectorXs> u,
                           const Eigen::Ref<const VectorXs>& x,
                           const std::size_t maxiter,
                           const Scalar tol) override {
    Data* d = static_cast<Data*>(data.get());
    model_->quasiStatic(d->action, u, x, maxiter, tol);
  }

  /**
   * @brief Cast the codegen action model to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return ActionModelCodeGenTpl<NewScalar> A codegen action model with the
   * new scalar type.
   */
  template <typename NewScalar>
  ActionModelCodeGenTpl<NewScalar> cast() const {
    typedef ActionModelCodeGenTpl<NewScalar> ReturnType;
    typedef typename ReturnType::ADScalar ADNewScalar;
    ReturnType ret(model_->template cast<NewScalar>(), lib_fname_, nP_,
                   cast_function<ADScalar, ADNewScalar>(updateParams_),
                   compiler_type_, compile_options_);
    return ret;
  }

  /**
   * @brief Return the action model
   */
  const std::shared_ptr<Base>& get_model() const { return model_; }

  /**
   * @brief Return the number of inequality constraints
   */
  virtual std::size_t get_ng() const override { return model_->get_ng(); }

  /**
   * @brief Return the number of equality constraints
   */
  virtual std::size_t get_nh() const override { return model_->get_nh(); }

  /**
   * @brief Return the number of inequality terminal constraints
   */
  virtual std::size_t get_ng_T() const override { return model_->get_ng_T(); }

  /**
   * @brief Return the number of equality terminal constraints
   */
  virtual std::size_t get_nh_T() const override { return model_->get_nh_T(); }

  /**
   * @brief Return the lower bound of the inequality constraints
   */
  virtual const VectorXs& get_g_lb() const override {
    return model_->get_g_lb();
  }

  /**
   * @brief Return the upper bound of the inequality constraints
   */
  virtual const VectorXs& get_g_ub() const override {
    return model_->get_g_ub();
  }

  /**
   * @brief Return the dimension of the dependent vector used by calc and
   * calcDiff functions
   */
  std::size_t get_nX() const { return nX_; }

  /**
   * @brief Return the dimension of the dependent vector used by calc and
   * calcDiff functions in terminal nodes
   */
  std::size_t get_nX_T() const { return nX_T_; }

  /**
   * @brief Return the dimension of the independent vector used by calc function
   */
  std::size_t get_nY1() const { return nY1_; }

  /**
   * @brief Return the dimension of the independent vector used by calc function
   * in terminal nodes
   */
  std::size_t get_nY1_T() const { return nY1_T_; }

  /**
   * @brief Return the dimension of the independent vector used by calcDiff
   * function
   */
  std::size_t get_nY2() const { return nY2_; }

  /**
   * @brief Return the dimension of the independent vector used by calcDiff
   * function in terminal nodes
   */
  std::size_t get_nY2_T() const { return nY2_T_; }

  /**
   * @brief Print relevant information of the action model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override { model_->print(os); }

 protected:
  ActionModelCodeGenTpl()
      : model_(nullptr),
        nP_(0),
        lib_fname_(""),
        compiler_type_(CLANG),
        compile_options_("-Ofast -march=native"),
        updateParams_(EmptyParamsEnv) {
    // Add initialization logic if necessary
  }

  using Base::ng_;     //!< Number of inequality constraints
  using Base::ng_T_;   //!< Number of inequality constraints in terminal nodes
  using Base::nh_;     //!< Number of equality constraints
  using Base::nh_T_;   //!< Number of equality constraints in terminal nodes
  using Base::nu_;     //!< Control dimension
  using Base::state_;  //!< Model of the state

  std::shared_ptr<Base> model_;  //!< Action model to be code generated
  std::shared_ptr<ADBase>
      ad_model_;  //!< Action model needed for code generation
  std::shared_ptr<ADActionDataAbstract>
      ad_data_;  //! Action data needed for code generation

  std::size_t nP_;  //!< Dimension of the parameter variables in calc and
                    //!< calcDiff functions
  std::size_t nX_;  //!< Dimension of the independent variables used by calc and
                    //!< calcDiff functions
  std::size_t nX_T_;  //!< Dimension of the independent variables used by calc
                      //!< and calcDiff functions in terminal nodes
  std::size_t
      nY1_;  //!< Dimension of the dependent variables used by calc function
  std::size_t nY1_T_;  //!< Dimension of the dependent variables used by calc
                       //!< function in terminal nodes
  std::size_t
      nY2_;  //!< Dimension of the dependent variables used by calcDiff function
  std::size_t nY2_T_;  //!< Dimension of the dependent variables used by
                       //!< calcDiff function in terminal nodes
  ADVectorXs ad_X_;    //!< Independent variables used to tape calc and calcDiff
                       //!< functions
  ADVectorXs ad_X_T_;  //!< Independent variables used to tape calc and calcDiff
                       //!< functions in terminal nodes
  ADVectorXs ad_Y1_;   //!< Dependent variables used to tape calc function
  ADVectorXs ad_Y1_T_;  //!< Dependent variables used to tape calc function in
                        //!< terminal nodes
  ADVectorXs ad_Y2_;    //!< Dependent variables used to tape calcDiff function
  ADVectorXs ad_Y2_T_;  //!< Dependent variables used to tape calcDiff function
                        //!< in terminal nodes

  const std::string Y1fun_name_;  //!< Name of the calc function
  const std::string
      Y1Tfun_name_;  //!< Name of the calc function used in terminal nodes
  const std::string Y2fun_name_;  //!< Name of the calcDiff function
  const std::string
      Y2Tfun_name_;  //!< Name of the calcDiff function used in terminal nodes
  const std::string lib_fname_;        //!< Name of the code generated library
  CompilerType compiler_type_;         //!< Type of compiler
  const std::string compile_options_;  //!< Compilation options

  ParamsEnvironment updateParams_;  // Lambda function that updates parameter
                                    // variables before starting record.

  std::unique_ptr<ADFun> ad_calc_;  //! < Function used to code generate calc
  std::unique_ptr<ADFun>
      ad_calc_T_;  //! < Function used to code generate calc in terminal nodes
  std::unique_ptr<ADFun>
      ad_calcDiff_;  //!< Function used to code generate calcDiff
  std::unique_ptr<ADFun> ad_calcDiff_T_;  //!< Function used to code generate
                                          //!< calcDiff in terminal nodes
  std::unique_ptr<CSourceGen>
      calcCG_;  //!< Code generated source code of calc function
  std::unique_ptr<CSourceGen> calcCG_T_;  //!< Code generated source code of
                                          //!< calc function in terminal nodes
  std::unique_ptr<CSourceGen>
      calcDiffCG_;  //!< Code generated source code of calcDiff function
  std::unique_ptr<CSourceGen>
      calcDiffCG_T_;  //!< Code generated source code of calcDiff function in
                      //!< terminal nodes
  std::unique_ptr<LibraryCSourceGen>
      libCG_;  //!< Library of the code generated source code
  std::unique_ptr<LibraryProcessor>
      dynLibManager_;  //!< Dynamic library manager
  std::unique_ptr<DynamicLib> dynLib_;
  std::unique_ptr<GenericModel> calcFun_;  //!< Code generated calc function
  std::unique_ptr<GenericModel>
      calcFun_T_;  //!< Code generated calc function in terminal nodes
  std::unique_ptr<GenericModel>
      calcDiffFun_;  //!< Code generated calcDiff function
  std::unique_ptr<GenericModel>
      calcDiffFun_T_;  //!< Code generated calcDiff function in terminal nodes

 private:
  void recordCalc() {
    const std::size_t nx = state_->get_nx();
    // Define the calc's input as the independent variables
    CppAD::Independent(ad_X_);
    // Record the calc's environment variables
    updateParams_(ad_model_, ad_X_.tail(nP_));
    // Collect computation in calc
    ad_model_->calc(ad_data_, ad_X_.head(nx), ad_X_.segment(nx, nu_));
    tapeCalcOutput();
    // Define calc's output as the dependent variable
    ad_calc_->Dependent(ad_X_, ad_Y1_);
    ad_calc_->optimize("no_compare_op");
  }

  void recordCalc_T() {
    const std::size_t nx = state_->get_nx();
    // Define the calc's input as the independent variables
    CppAD::Independent(ad_X_T_);
    // Record the calc's environment variables
    updateParams_(ad_model_, ad_X_T_.tail(nP_));
    // Collect computation in calc
    ad_model_->calc(ad_data_, ad_X_T_.head(nx));
    tapeCalcOutput_T();
    // Define calc's output as the dependent variable
    ad_calc_T_->Dependent(ad_X_T_, ad_Y1_T_);
    ad_calc_T_->optimize("no_compare_op");
  }

  void recordCalcDiff() {
    const std::size_t nx = state_->get_nx();
    // Define the calcDiff's input as the independent variables
    CppAD::Independent(ad_X_);
    // Record the calcDiff's environment variables
    updateParams_(ad_model_, ad_X_.tail(nP_));
    // Collect computation in calcDiff
    ad_model_->calc(ad_data_, ad_X_.head(nx), ad_X_.segment(nx, nu_));
    ad_model_->calcDiff(ad_data_, ad_X_.head(nx), ad_X_.segment(nx, nu_));
    tapeCalcDiffOutput();
    // Define calcDiff's output as the dependent variable
    ad_calcDiff_->Dependent(ad_X_, ad_Y2_);
    ad_calcDiff_->optimize("no_compare_op");
  }

  void recordCalcDiff_T() {
    const std::size_t nx = state_->get_nx();
    // Define the calcDiff's input as the independent variables
    CppAD::Independent(ad_X_T_);
    // Record the calcDiff's environment variables
    updateParams_(ad_model_, ad_X_T_.tail(nP_));
    // Collect computation in calcDiff
    ad_model_->calc(ad_data_, ad_X_T_.head(nx));
    ad_model_->calcDiff(ad_data_, ad_X_T_.head(nx));
    tapeCalcDiffOutput_T();
    // Define calcDiff's output as the dependent variable
    ad_calcDiff_T_->Dependent(ad_X_T_, ad_Y2_T_);
    ad_calcDiff_T_->optimize("no_compare_op");
  }

  void tapeCalcOutput() {
    Eigen::DenseIndex it_Y1 = 0;
    ad_Y1_[it_Y1] = ad_data_->cost;
    it_Y1 += 1;
    ad_Y1_.segment(it_Y1, state_->get_nx()) = ad_data_->xnext;
    it_Y1 += state_->get_nx();
    ad_Y1_.segment(it_Y1, ng_) = ad_data_->g;
    it_Y1 += ng_;
    ad_Y1_.segment(it_Y1, nh_) = ad_data_->h;
  }

  void tapeCalcOutput_T() {
    Eigen::DenseIndex it_Y1 = 0;
    ad_Y1_T_[it_Y1] = ad_data_->cost;
    it_Y1 += 1;
    ad_Y1_T_.segment(it_Y1, ng_T_) = ad_data_->g;
    it_Y1 += ng_T_;
    ad_Y1_T_.segment(it_Y1, nh_T_) = ad_data_->h;
  }

  void tapeCalcDiffOutput() {
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

  void tapeCalcDiffOutput_T() {
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

  static void EmptyParamsEnv(std::shared_ptr<ADBase>,
                             const Eigen::Ref<const ADVectorXs>&) {}
};

template <typename _Scalar>
struct ActionDataCodeGenTpl : public ActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActionDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit ActionDataCodeGenTpl(Model<Scalar>* const model)
      : Base(model), action(model->get_model()->createData()) {
    ActionModelCodeGenTpl<Scalar>* m =
        static_cast<ActionModelCodeGenTpl<Scalar>*>(model);
    X.resize(m->get_nX());
    X_T.resize(m->get_nX_T());
    Y1.resize(m->get_nY1());
    Y1_T.resize(m->get_nY1_T());
    Y2.resize(m->get_nY2());
    Y2_T.resize(m->get_nY2_T());
    X.setZero();
    X_T.setZero();
    Y1.setZero();
    Y1_T.setZero();
    Y2.setZero();
    Y2_T.setZero();
  }

  using Base::cost;
  using Base::Fu;
  using Base::Fx;
  using Base::g;
  using Base::Gu;
  using Base::Gx;
  using Base::h;
  using Base::Hu;
  using Base::Hx;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::r;
  using Base::xnext;

  VectorXs X;    //!< Independent variables used by calc and calcDiff functions
  VectorXs X_T;  //!< Independent variables used by calc and calcDiff functions
                 //!< in terminal nodes
  VectorXs Y1;   //!< Dependent variables used by calc function
  VectorXs
      Y1_T;     //!< Dependent variables used by calc function in terminal nodes
  VectorXs Y2;  //!< Dependent variables used by calcDiff function
  VectorXs Y2_T;  //!< Dependent variables used by calcDiff function in terminal
                  //!< nodes
  std::shared_ptr<Base> action;  //!< Action data

  void set_Y1() {
    Eigen::DenseIndex it_Y1 = 0;
    const std::size_t nx = xnext.size();
    const std::size_t ng = g.size();
    const std::size_t nh = h.size();
    cost = Y1[it_Y1];
    it_Y1 += 1;
    xnext = Y1.segment(it_Y1, nx);
    it_Y1 += nx;
    g = Y1.segment(it_Y1, ng);
    it_Y1 += ng;
    h = Y1.segment(it_Y1, nh);
  }

  void set_Y1_T() {
    Eigen::DenseIndex it_Y1 = 0;
    const std::size_t ng = g.size();
    const std::size_t nh = h.size();
    cost = Y1_T[it_Y1];
    it_Y1 += 1;
    g = Y1_T.segment(it_Y1, ng);
    it_Y1 += ng;
    h = Y1_T.segment(it_Y1, nh);
  }

  void set_Y2() {
    const std::size_t ndx = Fx.cols();
    const std::size_t nu = Fu.cols();
    const std::size_t ng = g.size();
    const std::size_t nh = h.size();
    Eigen::DenseIndex it_Y2 = 0;
    Fx = Eigen::Map<MatrixXs>(Y2.data() + it_Y2, ndx, ndx);
    it_Y2 += ndx * ndx;
    Fu = Eigen::Map<MatrixXs>(Y2.data() + it_Y2, ndx, nu);
    it_Y2 += ndx * nu;
    Lx = Eigen::Map<VectorXs>(Y2.data() + it_Y2, ndx);
    it_Y2 += ndx;
    Lu = Eigen::Map<VectorXs>(Y2.data() + it_Y2, nu);
    it_Y2 += nu;
    Lxx = Eigen::Map<MatrixXs>(Y2.data() + it_Y2, ndx, ndx);
    it_Y2 += ndx * ndx;
    Lxu = Eigen::Map<MatrixXs>(Y2.data() + it_Y2, ndx, nu);
    it_Y2 += ndx * nu;
    Luu = Eigen::Map<MatrixXs>(Y2.data() + it_Y2, nu, nu);
    it_Y2 += nu * nu;
    Gx = Eigen::Map<MatrixXs>(Y2.data() + it_Y2, ng, ndx);
    it_Y2 += ng * ndx;
    Gu = Eigen::Map<MatrixXs>(Y2.data() + it_Y2, ng, nu);
    it_Y2 += ng * nu;
    Hx = Eigen::Map<MatrixXs>(Y2.data() + it_Y2, nh, ndx);
    it_Y2 += nh * ndx;
    Hu = Eigen::Map<MatrixXs>(Y2.data() + it_Y2, nh, nu);
  }

  void set_Y2_T() {
    const std::size_t ndx = Fx.cols();
    const std::size_t ng = g.size();
    const std::size_t nh = h.size();
    Eigen::DenseIndex it_Y2 = 0;
    Lx = Eigen::Map<VectorXs>(Y2_T.data() + it_Y2, ndx);
    it_Y2 += ndx;
    Lxx = Eigen::Map<MatrixXs>(Y2_T.data() + it_Y2, ndx, ndx);
    it_Y2 += ndx * ndx;
    Gx = Eigen::Map<MatrixXs>(Y2_T.data() + it_Y2, ng, ndx);
    it_Y2 += ng * ndx;
    Hx = Eigen::Map<MatrixXs>(Y2_T.data() + it_Y2, nh, ndx);
  }
};

}  // namespace crocoddyl

CROCODDYL_DECLARE_FLOATINGPOINT_EXTERN_TEMPLATE_CLASS(
    crocoddyl::ActionModelCodeGenTpl)
CROCODDYL_DECLARE_FLOATINGPOINT_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::ActionDataCodeGenTpl)

#endif  // CROCODDYL_WITH_CODEGEN

#endif  // CROCODDYL_CORE_CODEGEN_ACTION_BASE_HPP_
