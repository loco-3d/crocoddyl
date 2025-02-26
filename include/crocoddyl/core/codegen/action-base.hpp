
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

#include <functional>

#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/utils/stop-watch.hpp"
#include "pinocchio/codegen/cppadcg.hpp"

namespace pinocchio {
template <typename NewScalar, typename Scalar>
struct ScalarCast<NewScalar, CppAD::cg::CG<Scalar>> {
  static NewScalar cast(const CppAD::cg::CG<Scalar>& value) {
    return static_cast<NewScalar>(value.getValue());
  }
};
}  // namespace pinocchio

namespace crocoddyl {

enum CompilerType { GCC = 0, CLANG };

template <typename Scalar>
struct ActionDataCodeGenTpl;

template <typename _Scalar>
class ActionModelCodeGenTpl : public ActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ActionModelBase, ActionModelCodeGenTpl)

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
  typedef
      typename PINOCCHIO_EIGEN_PLAIN_ROW_MAJOR_TYPE(ADMatrixXs) RowADMatrixXs;
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
      : Base(model->get_state(), model->get_nu()),
        model_(model),
        ad_model_(model->template cast<ADScalar>()),
        ad_data_(ad_model_->createData()),
        nP_(nP),
        nX_(state_->get_nx() + nu_ + nP_),
        nY1_(state_->get_nx() + 1),
        ad_X_(nX_),
        ad_Y1_(nY1_),
        Y1fun_name_("calc"),
        Y2fun_name_("calcDiff"),
        lib_fname_(lib_fname),
        compiler_type_(compiler),
        compile_options_(compile_options),
        updateParams_(updateParams),
        ad_calc_(std::make_unique<ADFun>()),
        ad_calcDiff_(std::make_unique<ADFun>()) {
    const std::size_t ndx = state_->get_ndx();
    nY2_ = 2 * ndx * ndx + 2 * ndx * nu_ + nu_ * nu_ + ndx + nu_;
    ad_Y2_.resize(nY2_);
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
      : Base(ad_model->get_state()->template cast<Scalar>(),
             ad_model->get_nu()),
        model_(ad_model->template cast<Scalar>()),
        ad_model_(ad_model),
        ad_data_(ad_model_->createData()),
        nP_(nP),
        nX_(state_->get_nx() + nu_ + nP_),
        nY1_(state_->get_nx() + 1),
        ad_X_(nX_),
        ad_Y1_(nY1_),
        Y1fun_name_("calc"),
        Y2fun_name_("calcDiff"),
        lib_fname_(lib_fname),
        compiler_type_(compiler),
        compile_options_(compile_options),
        updateParams_(updateParams),
        ad_calc_(std::make_unique<ADFun>()),
        ad_calcDiff_(std::make_unique<ADFun>()) {
    const std::size_t ndx = state_->get_ndx();
    nY2_ = 2 * ndx * ndx + 2 * ndx * nu_ + nu_ * nu_ + ndx + nu_;
    ad_Y2_.resize(nY2_);
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
        nY1_(other.nY1_),
        nY2_(other.nY2_),
        ad_X_(other.nX_),
        ad_Y1_(other.nY1_),
        ad_Y2_(other.nY2_),
        Y1fun_name_(other.Y1fun_name_),
        Y2fun_name_(other.Y2fun_name_),
        lib_fname_(other.lib_fname_),
        compiler_type_(other.compiler_type_),
        compile_options_(other.compile_options_),
        updateParams_(other.updateParams_),
        ad_calc_(std::make_unique<ADFun>(std::move(*other.ad_calc_))),
        ad_calcDiff_(std::make_unique<ADFun>(std::move(*other.ad_calcDiff_))) {
    initLib();
    loadLib();
  }

  virtual ~ActionModelCodeGenTpl() = default;

  /**
   * @brief Initialize the code-generated library
   */
  void initLib() {
    START_PROFILER("ActionModelCodeGen::initLib");
    recordCalc();
    // Generate source code for calc
    calcCG_ = std::unique_ptr<CSourceGen>(
        new CSourceGen(*ad_calc_.get(), Y1fun_name_));
    calcCG_->setCreateForwardZero(true);
    calcCG_->setCreateJacobian(false);
    // Generate source code for calcDiff
    recordCalcDiff();
    calcDiffCG_ = std::unique_ptr<CSourceGen>(
        new CSourceGen(*ad_calcDiff_.get(), Y2fun_name_));
    calcDiffCG_->setCreateForwardZero(true);
    calcDiffCG_->setCreateJacobian(false);
    // Generate library for calc and calcDiff
    libCG_ = std::unique_ptr<LibraryCSourceGen>(
        new LibraryCSourceGen(*calcCG_, *calcDiffCG_));
    // Crate dynamic library manager
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
    calcDiffFun_ = dynLib_->model(Y2fun_name_.c_str());
  }

  void set_parameters(const std::shared_ptr<ActionDataAbstract>& data,
                      const Eigen::Ref<const VectorXs>& p) const {
    Data* d = static_cast<Data*>(data.get());
    d->X.tail(nP_) = p;
  }

  /**
   * @brief Compute the next state and cost value from a code-generated library
   *
   * @param[in] data  Action data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  void calc(const std::shared_ptr<ActionDataAbstract>& data,
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
   * @brief Compute the derivatives of the dynamics and cost functions from a
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
  void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
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
   * @brief Create the data for the code-generated action
   *
   * @return the action data
   */
  std::shared_ptr<ActionDataAbstract> createData() override {
    return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
  }

  /**
   * @brief Checks that a specific data belongs to this model
   */
  bool checkData(const std::shared_ptr<ActionDataAbstract>& data) override {
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
  void quasiStatic(const std::shared_ptr<ActionDataAbstract>& data,
                   Eigen::Ref<VectorXs> u, const Eigen::Ref<const VectorXs>& x,
                   const std::size_t maxiter, const Scalar tol) override {
    model_->quasiStatic(data, u, x, maxiter, tol);
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
    ReturnType ret(model_->template cast<NewScalar>(), lib_fname_);
    return ret;
  }

  /**
   * @brief Return the number of inequality constraints
   */
  std::size_t get_ng() const override { return model_->get_ng(); }

  /**
   * @brief Return the number of equality constraints
   */
  std::size_t get_nh() const override { return model_->get_nh(); }

  /**
   * @brief Return the number of inequality terminal constraints
   */
  std::size_t get_ng_T() const override { return model_->get_ng_T(); }

  /**
   * @brief Return the number of equality terminal constraints
   */
  std::size_t get_nh_T() const override { return model_->get_nh_T(); }

  /**
   * @brief Return the lower bound of the inequality constraints
   */
  const VectorXs& get_g_lb() const override { return model_->get_g_lb(); }

  /**
   * @brief Return the upper bound of the inequality constraints
   */
  const VectorXs& get_g_ub() const override { return model_->get_g_ub(); }

  /**
   * @brief Return the dimension of the dependent vector used by calc and
   * calcDiff functions
   */
  std::size_t get_nX() const { return nX_; }

  /**
   * @brief Return the dimension of the independent vector used by calc function
   */
  std::size_t get_nY1() const { return nY1_; }

  /**
   * @brief Return the dimension of the independent vector used by calcDiff
   * function
   */
  std::size_t get_nY2() const { return nY2_; }

  /**
   * @brief Print relevant information of the action model
   *
   * @param[out] os  Output stream object
   */
  void print(std::ostream& os) const override { model_->print(os); }

 protected:
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
  std::size_t
      nY1_;  //!< Dimension of the dependent variables used by calc function
  std::size_t
      nY2_;  //!< Dimension of the dependent variables used by calcDiff function
  ADVectorXs ad_X_;   //!< Independent variables used to tape calc and calcDiff
                      //!< functions
  ADVectorXs ad_Y1_;  //!< Dependent variables used to tape calc function
  ADVectorXs ad_Y2_;  //!< Dependent variables used to tape calcDiff function

  const std::string Y1fun_name_;       //!< Name of the calc function
  const std::string Y2fun_name_;       //!< Name of the calcDiff function
  const std::string lib_fname_;        //!< Name of the code generated library
  CompilerType compiler_type_;         //!< Type of compiler
  const std::string compile_options_;  //!< Compilation options

  ParamsEnvironment updateParams_;  // Lambda function that updates parameter
                                    // variables before starting record.

  std::unique_ptr<ADFun> ad_calc_;  //! < Function used to code generate calc
  std::unique_ptr<ADFun>
      ad_calcDiff_;  //!< Function used to code generate calcDiff
  std::unique_ptr<CSourceGen>
      calcCG_;  //!< Code generated source code of calc function
  std::unique_ptr<CSourceGen>
      calcDiffCG_;  //!< Code generated source code of calcDiff function
  std::unique_ptr<LibraryCSourceGen>
      libCG_;  //!< Library of the code generated source code
  std::unique_ptr<LibraryProcessor>
      dynLibManager_;  //!< Dynamic library manager
  std::unique_ptr<DynamicLib> dynLib_;
  std::unique_ptr<GenericModel> calcFun_;  //!< Code generated calc function
  std::unique_ptr<GenericModel>
      calcDiffFun_;  //!< Code generated calcDiff function

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

  void tapeCalcOutput() {
    ad_Y1_[0] = ad_data_->cost;
    ad_Y1_.tail(state_->get_nx()) = ad_data_->xnext;
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
  explicit ActionDataCodeGenTpl(Model<Scalar>* const model) : Base(model) {
    ActionModelCodeGenTpl<Scalar>* m =
        static_cast<ActionModelCodeGenTpl<Scalar>*>(model);
    X.resize(m->get_nX());
    Y1.resize(m->get_nY1());
    Y2.resize(m->get_nY1());
    X.setZero();
    Y1.setZero();
    Y2.setZero();
  }

  using Base::cost;
  using Base::Fu;
  using Base::Fx;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::r;
  using Base::xnext;

  VectorXs X;   //!< Independent variables used by calc and calcDiff functions
  VectorXs Y1;  //!< Dependent variables used by calc function
  VectorXs Y2;  //!< Dependent variables used by calcDiff function

  void set_Y1() {
    cost = Y1[0];
    xnext = Y1.tail(xnext.size());
  }

  void set_Y2() {
    const std::size_t ndx = Fx.cols();
    const std::size_t nu = Fu.cols();
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
  }
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_CODEGEN_ACTION_BASE_HPP_
