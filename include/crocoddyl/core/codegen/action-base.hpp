
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
  typedef ActionDataCodeGenTpl<ADScalar> ADActionDataCodeGen;
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
  typedef std::function<void(std::shared_ptr<ADBase>,
                             const Eigen::Ref<const ADVectorXs>&)>
      ActionEnvironment;

  ActionModelCodeGenTpl(std::shared_ptr<ADBase> admodel,
                        std::shared_ptr<Base> model,
                        const std::string& library_name,
                        const std::size_t n_env = 0,
                        ActionEnvironment fn_record_env = empty_record_env,
                        const std::string& function_name_calc = "calc",
                        const std::string& function_name_calcDiff = "calcDiff")
      : Base(model->get_state(), model->get_nu()),
        model(model),
        ad_model(admodel),
        ad_data(ad_model->createData()),
        n_env(n_env),
        nX(ad_model->get_state()->get_nx() + ad_model->get_nu() + n_env),
        nY1(ad_model->get_state()->get_nx() + 1),
        ad_X(nX),
        ad_Y1(nY1),
        function_name_calc(function_name_calc),
        function_name_calcDiff(function_name_calcDiff),
        library_name(library_name),
        fn_record_env(fn_record_env) {
    const std::size_t ndx = ad_model->get_state()->get_ndx();
    const std::size_t nu = ad_model->get_nu();
    nY2 = 2 * ndx * ndx + 2 * ndx * nu + nu * nu + ndx + nu;
    ad_Y2.resize(2 * ndx * ndx + 2 * ndx * nu + nu * nu + ndx + nu);
    initLib();
    loadLib();
  }

  ActionModelCodeGenTpl(const ActionModelCodeGenTpl<Scalar>& other)
      : Base(other),
        model(other.model),
        ad_model(other.ad_model),
        n_env(other.n_env),
        nX(other.nX),
        nY1(other.nY1),
        nY2(other.nY2),
        ad_X(other.nX),
        ad_Y1(other.nY1),
        ad_Y2(other.nY2),
        function_name_calc(other.function_name_calc),
        function_name_calcDiff(other.function_name_calcDiff),
        library_name(other.library_name),
        fn_record_env(other.fn_record_env),
        ad_calc(std::make_unique<ADFun>(std::move(*other.ad_calc))),
        ad_calcDiff(std::make_unique<ADFun>(std::move(*other.ad_calcDiff))) {
    initLib();
    loadLib();
  }

  static void empty_record_env(std::shared_ptr<ADBase>,
                               const Eigen::Ref<const ADVectorXs>&) {}

  void recordCalc() {
    const std::size_t nx = ad_model->get_state()->get_nx();
    const std::size_t nu = ad_model->get_nu();
    // Define the calc's input as the independent variables
    CppAD::Independent(ad_X);
    // Record the calc's environment variables
    fn_record_env(ad_model, ad_X.tail(n_env));
    // Collect computation in calc
    ad_model->calc(ad_data, ad_X.head(nx), ad_X.segment(nx, nu));
    tapeCalcOutput();
    // Define calc's output as the dependent variable
    ad_calc->Dependent(ad_X, ad_Y1);
    ad_calc->optimize("no_compare_op");
  }

  void recordCalcDiff() {
    const std::size_t nx = ad_model->get_state()->get_nx();
    const std::size_t nu = ad_model->get_nu();
    // Define the calcDiff's input as the independent variables
    CppAD::Independent(ad_X);
    // Record the calcDiff's environment variables
    fn_record_env(ad_model, ad_X.tail(n_env));
    // Collect computation in calcDiff
    ad_model->calc(ad_data, ad_X.head(nx), ad_X.segment(nx, nu));
    ad_model->calcDiff(ad_data, ad_X.head(nx), ad_X.segment(nx, nu));
    tapeCalcDiffOutput();
    // Define calcDiff's output as the dependent variable
    ad_calcDiff->Dependent(ad_X, ad_Y2);
    ad_calcDiff->optimize("no_compare_op");
  }

  void tapeCalcOutput() {
    ad_Y1[0] = ad_data->cost;
    ad_Y1.tail(ad_model->get_state()->get_nx()) = ad_data->xnext;
  }

  void tapeCalcDiffOutput() {
    const std::size_t ndx = ad_model->get_state()->get_ndx();
    const std::size_t nu = ad_model->get_nu();
    ADVectorXs& ad_Y = ad_Y2;
    Eigen::DenseIndex it_Y = 0;
    Eigen::Map<ADMatrixXs>(ad_Y.data() + it_Y, ndx, ndx) = ad_data->Fx;
    it_Y += ndx * ndx;
    Eigen::Map<ADMatrixXs>(ad_Y.data() + it_Y, ndx, nu) = ad_data->Fu;
    it_Y += ndx * nu;
    Eigen::Map<ADVectorXs>(ad_Y.data() + it_Y, ndx) = ad_data->Lx;
    it_Y += ndx;
    Eigen::Map<ADVectorXs>(ad_Y.data() + it_Y, nu) = ad_data->Lu;
    it_Y += nu;
    Eigen::Map<ADMatrixXs>(ad_Y.data() + it_Y, ndx, ndx) = ad_data->Lxx;
    it_Y += ndx * ndx;
    Eigen::Map<ADMatrixXs>(ad_Y.data() + it_Y, ndx, nu) = ad_data->Lxu;
    it_Y += ndx * nu;
    Eigen::Map<ADMatrixXs>(ad_Y.data() + it_Y, nu, nu) = ad_data->Luu;
  }

  void initLib() {
    recordCalc();
    // Generate source code for calc
    calcgen_ptr = std::unique_ptr<CSourceGen>(
        new CSourceGen(*ad_calc.get(), function_name_calc));
    calcgen_ptr->setCreateForwardZero(true);
    calcgen_ptr->setCreateJacobian(false);
    // Generate source code for calcDiff
    recordCalcDiff();
    calcDiffgen_ptr = std::unique_ptr<CSourceGen>(
        new CSourceGen(*ad_calcDiff.get(), function_name_calcDiff));
    calcDiffgen_ptr->setCreateForwardZero(true);
    calcDiffgen_ptr->setCreateJacobian(false);
    // Generate library for calc and calcDiff
    libcgen_ptr = std::unique_ptr<LibraryCSourceGen>(
        new LibraryCSourceGen(*calcgen_ptr, *calcDiffgen_ptr));
    // Crate dynamic library manager
    dynamicLibManager_ptr = std::unique_ptr<LibraryProcessor>(
        new LibraryProcessor(*libcgen_ptr, library_name));
  }

  void compileLib() {
    CppAD::cg::GccCompiler<Scalar> compiler;
    std::vector<std::string> compile_options = compiler.getCompileFlags();
    compile_options[0] = "-O3";
    compiler.setCompileFlags(compile_options);
    dynamicLibManager_ptr->createDynamicLibrary(compiler, false);
  }

  bool existLib() const {
    const std::string filename =
        dynamicLibManager_ptr->getLibraryName() +
        CppAD::cg::system::SystemInfo<>::DYNAMIC_LIB_EXTENSION;
    std::ifstream file(filename.c_str());
    return file.good();
  }

  void loadLib(const bool generate_if_not_exist = true) {
    if (not existLib() && generate_if_not_exist) compileLib();

    const auto it = dynamicLibManager_ptr->getOptions().find("dlOpenMode");
    if (it == dynamicLibManager_ptr->getOptions().end()) {
      dynamicLib_ptr.reset(new CppAD::cg::LinuxDynamicLib<Scalar>(
          dynamicLibManager_ptr->getLibraryName() +
          CppAD::cg::system::SystemInfo<>::DYNAMIC_LIB_EXTENSION));
    } else {
      int dlOpenMode = std::stoi(it->second);
      dynamicLib_ptr.reset(new CppAD::cg::LinuxDynamicLib<Scalar>(
          dynamicLibManager_ptr->getLibraryName() +
              CppAD::cg::system::SystemInfo<>::DYNAMIC_LIB_EXTENSION,
          dlOpenMode));
    }
    calcFun_ptr = dynamicLib_ptr->model(function_name_calc.c_str());
    calcDiffFun_ptr = dynamicLib_ptr->model(function_name_calcDiff.c_str());
  }

  void set_env(const std::shared_ptr<ActionDataAbstract>& data,
               const Eigen::Ref<const VectorXs>& env_val) const {
    Data* d = static_cast<Data*>(data.get());
    d->X.tail(n_env) = env_val;
  }

  void calc(const std::shared_ptr<ActionDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u) override {
    Data* d = static_cast<Data*>(data.get());
    const std::size_t nx = ad_model->get_state()->get_nx();
    const std::size_t nu = ad_model->get_nu();

    d->X.head(nx) = x;
    d->X.segment(nx, nu) = u;
    calcFun_ptr->ForwardZero(d->X, d->Y1);
    d->set_Y1();
  }

  void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u) override {
    Data* d = static_cast<Data*>(data.get());
    const std::size_t nx = ad_model->get_state()->get_nx();
    const std::size_t nu = ad_model->get_nu();

    d->X.head(nx) = x;
    d->X.segment(nx, nu) = u;
    calcDiffFun_ptr->ForwardZero(d->X, d->Y2);
    d->set_Y2();
  }

  std::shared_ptr<ActionDataAbstract> createData() override {
    return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
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
    typedef CppAD::cg::CG<NewScalar> CGNewScalar;
    typedef CppAD::AD<CGNewScalar> ADNewScalar;
    ReturnType ret(ad_model->template cast<ADNewScalar>(),
                   model->template cast<NewScalar>(), library_name);
    return ret;
  }

  /**
   * @brief Return the dimension of the dependent vector used by calc and
   * calcDiff functions
   */
  std::size_t get_nX() const { return nX; }

  /**
   * @brief Return the dimension of the independent vector used by calc function
   */
  std::size_t get_nY1() const { return nY1; }

  /**
   * @brief Return the dimension of the independent vector used by calcDiff
   * function
   */
  std::size_t get_nY2() const { return nY2; }

 protected:
  using Base::has_control_limits_;  //!< Indicates whether any of the control
                                    //!< limits
  using Base::nr_;                  //!< Dimension of the cost residual
  using Base::nu_;                  //!< Control dimension
  using Base::state_;               //!< Model of the state
  using Base::u_lb_;                //!< Lower control limits
  using Base::u_ub_;                //!< Upper control limits

  std::shared_ptr<Base> model;
  std::shared_ptr<ADBase> ad_model;
  std::shared_ptr<ADActionDataAbstract> ad_data;

  std::size_t n_env;  //!< Dimension of the environment variables
  std::size_t nX;  //!< Dimension of the independent variables used by calc and
                   //!< calcDiff functions
  std::size_t
      nY1;  //!< Dimension of the dependent variables used by calc function
  std::size_t
      nY2;  //!< Dimension of the dependent variables used by calcDiff function
  ADVectorXs
      ad_X;  //!< Independent variables used to tape calc and calcDiff functions
  ADVectorXs ad_Y1;  //!< Dependent variables used to tape calc function
  ADVectorXs ad_Y2;  //!< Dependent variables used to tape calcDiff function

  /// \brief Name of the function
  const std::string function_name_calc, function_name_calcDiff;

  /// \brief Name of the library
  const std::string library_name;

  /// \brief A function that updates the environment variables before starting
  /// record.
  ActionEnvironment fn_record_env;

  /// \brief Options to generate or not the source code for the evaluation
  /// function
  bool build_forward;

  std::unique_ptr<ADFun> ad_calc, ad_calcDiff;

  std::unique_ptr<CSourceGen> calcgen_ptr;
  std::unique_ptr<CSourceGen> calcDiffgen_ptr;
  std::unique_ptr<LibraryCSourceGen> libcgen_ptr;
  std::unique_ptr<LibraryProcessor> dynamicLibManager_ptr;
  std::unique_ptr<DynamicLib> dynamicLib_ptr;
  std::unique_ptr<GenericModel> calcFun_ptr;
  std::unique_ptr<GenericModel> calcDiffFun_ptr;
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
    VectorXs& Y = Y2;
    const std::size_t ndx = Fx.rows();
    const std::size_t nu = Fu.cols();

    Eigen::DenseIndex it_Y = 0;
    Fx = Eigen::Map<MatrixXs>(Y.data() + it_Y, ndx, ndx);
    it_Y += ndx * ndx;
    Fu = Eigen::Map<MatrixXs>(Y.data() + it_Y, ndx, nu);
    it_Y += ndx * nu;
    Lx = Eigen::Map<VectorXs>(Y.data() + it_Y, ndx);
    it_Y += ndx;
    Lu = Eigen::Map<VectorXs>(Y.data() + it_Y, nu);
    it_Y += nu;
    Lxx = Eigen::Map<MatrixXs>(Y.data() + it_Y, ndx, ndx);
    it_Y += ndx * ndx;
    Lxu = Eigen::Map<MatrixXs>(Y.data() + it_Y, ndx, nu);
    it_Y += ndx * nu;
    Luu = Eigen::Map<MatrixXs>(Y.data() + it_Y, nu, nu);
  }
};

}  // namespace crocoddyl

#endif  // ifndef CROCODDYL_CORE_CODEGEN_ACTION_BASE_HPP_
