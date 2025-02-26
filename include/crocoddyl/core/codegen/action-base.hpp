
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
  typedef CppAD::cg::LinuxDynamicLib<Scalar> LinuxDynamicLib;
  typedef CppAD::cg::system::SystemInfo<> SystemInfo;
  typedef std::function<void(std::shared_ptr<ADBase>,
                             const Eigen::Ref<const ADVectorXs>&)>
      ParamsEnvironment;

  ActionModelCodeGenTpl(std::shared_ptr<ADBase> admodel,
                        std::shared_ptr<Base> model,
                        const std::string& lib_fname, const std::size_t nP = 0,
                        ParamsEnvironment updateParams = empty_record_env,
                        const std::string& Y1fun_name = "calc",
                        const std::string& Y2fun_name = "calcDiff")
      : Base(model->get_state(), model->get_nu()),
        model(model),
        ad_model(admodel),
        ad_data(ad_model->createData()),
        nP(nP),
        nX(ad_model->get_state()->get_nx() + ad_model->get_nu() + nP),
        nY1(ad_model->get_state()->get_nx() + 1),
        ad_X(nX),
        ad_Y1(nY1),
        Y1fun_name(Y1fun_name),
        Y2fun_name(Y2fun_name),
        lib_fname(lib_fname),
        updateParams(updateParams) {
    const std::size_t ndx = state_->get_ndx();
    nY2 = 2 * ndx * ndx + 2 * ndx * nu_ + nu_ * nu_ + ndx + nu_;
    ad_Y2.resize(nY2);
    initLib();
    loadLib();
  }

  ActionModelCodeGenTpl(const ActionModelCodeGenTpl<Scalar>& other)
      : Base(other),
        model(other.model),
        ad_model(other.ad_model),
        nP(other.nP),
        nX(other.nX),
        nY1(other.nY1),
        nY2(other.nY2),
        ad_X(other.nX),
        ad_Y1(other.nY1),
        ad_Y2(other.nY2),
        Y1fun_name(other.Y1fun_name),
        Y2fun_name(other.Y2fun_name),
        lib_fname(other.lib_fname),
        updateParams(other.updateParams),
        ad_calc(std::make_unique<ADFun>(std::move(*other.ad_calc))),
        ad_calcDiff(std::make_unique<ADFun>(std::move(*other.ad_calcDiff))) {
    initLib();
    loadLib();
  }

  static void empty_record_env(std::shared_ptr<ADBase>,
                               const Eigen::Ref<const ADVectorXs>&) {}

  void recordCalc() {
    const std::size_t nx = state_->get_nx();
    // Define the calc's input as the independent variables
    CppAD::Independent(ad_X);
    // Record the calc's environment variables
    updateParams(ad_model, ad_X.tail(nP));
    // Collect computation in calc
    ad_model->calc(ad_data, ad_X.head(nx), ad_X.segment(nx, nu_));
    tapeCalcOutput();
    // Define calc's output as the dependent variable
    ad_calc->Dependent(ad_X, ad_Y1);
    ad_calc->optimize("no_compare_op");
  }

  void recordCalcDiff() {
    const std::size_t nx = state_->get_nx();
    // Define the calcDiff's input as the independent variables
    CppAD::Independent(ad_X);
    // Record the calcDiff's environment variables
    updateParams(ad_model, ad_X.tail(nP));
    // Collect computation in calcDiff
    ad_model->calc(ad_data, ad_X.head(nx), ad_X.segment(nx, nu_));
    ad_model->calcDiff(ad_data, ad_X.head(nx), ad_X.segment(nx, nu_));
    tapeCalcDiffOutput();
    // Define calcDiff's output as the dependent variable
    ad_calcDiff->Dependent(ad_X, ad_Y2);
    ad_calcDiff->optimize("no_compare_op");
  }

  void tapeCalcOutput() {
    ad_Y1[0] = ad_data->cost;
    ad_Y1.tail(state_->get_nx()) = ad_data->xnext;
  }

  void tapeCalcDiffOutput() {
    const std::size_t ndx = state_->get_ndx();
    Eigen::DenseIndex it_Y2 = 0;
    Eigen::Map<ADMatrixXs>(ad_Y2.data() + it_Y2, ndx, ndx) = ad_data->Fx;
    it_Y2 += ndx * ndx;
    Eigen::Map<ADMatrixXs>(ad_Y2.data() + it_Y2, ndx, nu_) = ad_data->Fu;
    it_Y2 += ndx * nu_;
    Eigen::Map<ADVectorXs>(ad_Y2.data() + it_Y2, ndx) = ad_data->Lx;
    it_Y2 += ndx;
    Eigen::Map<ADVectorXs>(ad_Y2.data() + it_Y2, nu_) = ad_data->Lu;
    it_Y2 += nu_;
    Eigen::Map<ADMatrixXs>(ad_Y2.data() + it_Y2, ndx, ndx) = ad_data->Lxx;
    it_Y2 += ndx * ndx;
    Eigen::Map<ADMatrixXs>(ad_Y2.data() + it_Y2, ndx, nu_) = ad_data->Lxu;
    it_Y2 += ndx * nu_;
    Eigen::Map<ADMatrixXs>(ad_Y2.data() + it_Y2, nu_, nu_) = ad_data->Luu;
  }

  void initLib() {
    recordCalc();
    // Generate source code for calc
    calcCG =
        std::unique_ptr<CSourceGen>(new CSourceGen(*ad_calc.get(), Y1fun_name));
    calcCG->setCreateForwardZero(true);
    calcCG->setCreateJacobian(false);
    // Generate source code for calcDiff
    recordCalcDiff();
    calcDiffCG = std::unique_ptr<CSourceGen>(
        new CSourceGen(*ad_calcDiff.get(), Y2fun_name));
    calcDiffCG->setCreateForwardZero(true);
    calcDiffCG->setCreateJacobian(false);
    // Generate library for calc and calcDiff
    libCG = std::unique_ptr<LibraryCSourceGen>(
        new LibraryCSourceGen(*calcCG, *calcDiffCG));
    // Crate dynamic library manager
    dynLibManager = std::unique_ptr<LibraryProcessor>(
        new LibraryProcessor(*libCG, lib_fname));
  }

  void compileLib() {
    CppAD::cg::GccCompiler<Scalar> compiler;
    std::vector<std::string> compile_options = compiler.getCompileFlags();
    compile_options[0] = "-O3";
    compiler.setCompileFlags(compile_options);
    dynLibManager->createDynamicLibrary(compiler, false);
  }

  bool existLib() const {
    const std::string filename =
        dynLibManager->getLibraryName() + SystemInfo::DYNAMIC_LIB_EXTENSION;
    std::ifstream file(filename.c_str());
    return file.good();
  }

  void loadLib(const bool generate_if_not_exist = true) {
    if (!existLib() && generate_if_not_exist) {
      compileLib();
    }
    const auto it = dynLibManager->getOptions().find("dlOpenMode");
    const std::string filename =
        dynLibManager->getLibraryName() + SystemInfo::DYNAMIC_LIB_EXTENSION;
    if (it == dynLibManager->getOptions().end()) {
      dynLib.reset(new LinuxDynamicLib(filename));
    } else {
      int dlOpenMode = std::stoi(it->second);
      dynLib.reset(new LinuxDynamicLib(filename, dlOpenMode));
    }
    calcFun = dynLib->model(Y1fun_name.c_str());
    calcDiffFun = dynLib->model(Y2fun_name.c_str());
  }

  void set_parameters(const std::shared_ptr<ActionDataAbstract>& data,
                      const Eigen::Ref<const VectorXs>& p) const {
    Data* d = static_cast<Data*>(data.get());
    d->X.tail(nP) = p;
  }

  void calc(const std::shared_ptr<ActionDataAbstract>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u) override {
    Data* d = static_cast<Data*>(data.get());
    const std::size_t nx = ad_model->get_state()->get_nx();
    d->X.head(nx) = x;
    d->X.segment(nx, nu_) = u;
    calcFun->ForwardZero(d->X, d->Y1);
    d->set_Y1();
  }

  void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u) override {
    Data* d = static_cast<Data*>(data.get());
    const std::size_t nx = ad_model->get_state()->get_nx();
    d->X.head(nx) = x;
    d->X.segment(nx, nu_) = u;
    calcDiffFun->ForwardZero(d->X, d->Y2);
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
                   model_->template cast<NewScalar>(), lib_fname);
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
  using Base::nu_;     //!< Control dimension
  using Base::state_;  //!< Model of the state

  std::shared_ptr<Base> model;  //!< Action model to be code generated
  std::shared_ptr<ADBase>
      ad_model;  //!< Action model needed for code generation
  std::shared_ptr<ADActionDataAbstract>
      ad_data;  //! Action data needed for code generation

  std::size_t nP;  //!< Dimension of the parameter variables in calc and
                   //!< calcDiff functions
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

  const std::string Y1fun_name;  //!< Name of the calc function
  const std::string Y2fun_name;  //!< Name of the calcDiff function
  const std::string lib_fname;   //!< Name of the code generated library

  ParamsEnvironment updateParams;  // Lambda function that updates parameter
                                   // variables before starting record.

  std::unique_ptr<ADFun> ad_calc;  //! < Function used to code generate calc
  std::unique_ptr<ADFun>
      ad_calcDiff;  //!< Function used to code generate calcDiff
  std::unique_ptr<CSourceGen>
      calcCG;  //!< Code generated source code of calc function
  std::unique_ptr<CSourceGen>
      calcDiffCG;  //!< Code generated source code of calcDiff function
  std::unique_ptr<LibraryCSourceGen>
      libCG;  //!< Library of the code generated source code
  std::unique_ptr<LibraryProcessor> dynLibManager;  //!< Dynamic library manager
  std::unique_ptr<DynamicLib> dynLib;
  std::unique_ptr<GenericModel> calcFun;  //!< Code generated calc function
  std::unique_ptr<GenericModel>
      calcDiffFun;  //!< Code generated calcDiff function
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

#endif  // ifndef CROCODDYL_CORE_CODEGEN_ACTION_BASE_HPP_
