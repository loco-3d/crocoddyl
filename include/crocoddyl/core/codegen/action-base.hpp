
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, INRIA, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_CODEGEN_ACTION_BASE_HPP_
#define CROCODDYL_CORE_CODEGEN_ACTION_BASE_HPP_

#include <functional>
#include "pinocchio/codegen/cppadcg.hpp"

#include "crocoddyl/core/action-base.hpp"

namespace crocoddyl {

template <typename Scalar>
struct ActionDataCodeGenTpl;

template <typename _Scalar>
class ActionModelCodeGenTpl : public ActionModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef ActionModelAbstractTpl<Scalar> Base;
  typedef ActionDataCodeGenTpl<Scalar> Data;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef typename MathBaseTpl<Scalar>::VectorXs VectorXs;
  typedef typename MathBaseTpl<Scalar>::MatrixXs MatrixXs;

  typedef CppAD::cg::CG<Scalar> CGScalar;
  typedef CppAD::AD<CGScalar> ADScalar;
  typedef crocoddyl::ActionModelAbstractTpl<ADScalar> ADBase;
  typedef ActionDataAbstractTpl<ADScalar> ADActionDataAbstract;
  typedef ActionDataCodeGenTpl<ADScalar> ADActionDataCodeGen;
  typedef typename MathBaseTpl<ADScalar>::VectorXs ADVectorXs;
  typedef typename MathBaseTpl<ADScalar>::MatrixXs ADMatrixXs;
  typedef typename MathBaseTpl<ADScalar>::Vector3s ADVector3s;
  typedef typename MathBaseTpl<ADScalar>::Matrix3s ADMatrix3s;

  typedef typename PINOCCHIO_EIGEN_PLAIN_ROW_MAJOR_TYPE(ADMatrixXs) RowADMatrixXs;

  typedef CppAD::ADFun<CGScalar> ADFun;

  ActionModelCodeGenTpl(boost::shared_ptr<ADBase> admodel, boost::shared_ptr<Base> model,
                        const std::string& library_name, const std::size_t n_env = 0,
                        std::function<void(boost::shared_ptr<ADBase>, const Eigen::Ref<const ADVectorXs>&)>
                            fn_record_env = empty_record_env,
                        const std::string& function_name_calc = "calc",
                        const std::string& function_name_calcDiff = "calcDiff")
      : Base(model->get_state(), model->get_nu()),
        model(model),
        ad_model(admodel),
        ad_data(ad_model->createData()),
        function_name_calc(function_name_calc),
        function_name_calcDiff(function_name_calcDiff),
        library_name(library_name),
        n_env(n_env),
        fn_record_env(fn_record_env),
        ad_X(ad_model->get_state()->get_nx() + ad_model->get_nu() + n_env),
        ad_X2(ad_model->get_state()->get_nx() + ad_model->get_nu() + n_env),
        ad_calcout(ad_model->get_state()->get_nx() + 1) {
    std::size_t ndx = ad_model->get_state()->get_ndx();
    std::size_t nu = ad_model->get_nu();
    ad_calcDiffout.resize(2 * ndx * ndx + 2 * ndx * nu + nu * nu + ndx + nu);
    initLib();
    loadLib();
  }

  static void empty_record_env(boost::shared_ptr<ADBase>, const Eigen::Ref<const ADVectorXs>&) {}

  void recordCalc() {
    CppAD::Independent(ad_X);
    std::size_t nx = ad_model->get_state()->get_nx();
    std::size_t nu = ad_model->get_nu();

    fn_record_env(ad_model, ad_X.tail(n_env));

    ad_model->calc(ad_data, ad_X.head(nx), ad_X.segment(nx, nu));
    collect_calcout();
    // ad_calcout.template head<1>()[0] = ad_data->cost;
    // ad_calcout.tail(ad_model->get_state()->get_nx()) = ad_data->xnext;
    ad_calc.Dependent(ad_X, ad_calcout);
    ad_calc.optimize("no_compare_op");
  }

  void collect_calcout() {
    ad_calcout[0] = ad_data->cost;
    ad_calcout.tail(ad_model->get_state()->get_nx()) = ad_data->xnext;
  }

  void collect_calcDiffout() {
    ADVectorXs& ad_Y = ad_calcDiffout;

    std::size_t ndx = ad_model->get_state()->get_ndx();
    std::size_t nu = ad_model->get_nu();
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

  void recordCalcDiff() {
    CppAD::Independent(ad_X2);
    std::size_t nx = ad_model->get_state()->get_nx();
    std::size_t nu = ad_model->get_nu();

    fn_record_env(ad_model, ad_X2.tail(n_env));

    ad_model->calc(ad_data, ad_X2.head(nx), ad_X2.segment(nx, nu));
    ad_model->calcDiff(ad_data, ad_X2.head(nx), ad_X2.segment(nx, nu));

    collect_calcDiffout();
    ad_calcDiff.Dependent(ad_X2, ad_calcDiffout);
    ad_calcDiff.optimize("no_compare_op");
  }

  void initLib() {
    recordCalc();

    // generates source code
    calcgen_ptr = std::unique_ptr<CppAD::cg::ModelCSourceGen<Scalar> >(
        new CppAD::cg::ModelCSourceGen<Scalar>(ad_calc, function_name_calc));
    calcgen_ptr->setCreateForwardZero(true);
    calcgen_ptr->setCreateJacobian(false);

    // generates source code
    recordCalcDiff();
    calcDiffgen_ptr = std::unique_ptr<CppAD::cg::ModelCSourceGen<Scalar> >(
        new CppAD::cg::ModelCSourceGen<Scalar>(ad_calcDiff, function_name_calcDiff));
    calcDiffgen_ptr->setCreateForwardZero(true);
    calcDiffgen_ptr->setCreateJacobian(false);

    libcgen_ptr = std::unique_ptr<CppAD::cg::ModelLibraryCSourceGen<Scalar> >(
        new CppAD::cg::ModelLibraryCSourceGen<Scalar>(*calcgen_ptr, *calcDiffgen_ptr));

    dynamicLibManager_ptr = std::unique_ptr<CppAD::cg::DynamicModelLibraryProcessor<Scalar> >(
        new CppAD::cg::DynamicModelLibraryProcessor<Scalar>(*libcgen_ptr, library_name));
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
        dynamicLibManager_ptr->getLibraryName() + CppAD::cg::system::SystemInfo<>::DYNAMIC_LIB_EXTENSION;
    std::ifstream file(filename.c_str());
    return file.good();
  }

  void loadLib(const bool generate_if_not_exist = true) {
    if (not existLib() && generate_if_not_exist) compileLib();

    const auto it = dynamicLibManager_ptr->getOptions().find("dlOpenMode");
    if (it == dynamicLibManager_ptr->getOptions().end()) {
      dynamicLib_ptr.reset(new CppAD::cg::LinuxDynamicLib<Scalar>(
          dynamicLibManager_ptr->getLibraryName() + CppAD::cg::system::SystemInfo<>::DYNAMIC_LIB_EXTENSION));
    } else {
      int dlOpenMode = std::stoi(it->second);
      dynamicLib_ptr.reset(new CppAD::cg::LinuxDynamicLib<Scalar>(
          dynamicLibManager_ptr->getLibraryName() + CppAD::cg::system::SystemInfo<>::DYNAMIC_LIB_EXTENSION,
          dlOpenMode));
    }

    calcFun_ptr = dynamicLib_ptr->model(function_name_calc.c_str());
    calcDiffFun_ptr = dynamicLib_ptr->model(function_name_calcDiff.c_str());
  }

  void set_env(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& env_val) const {
    Data* d = static_cast<Data*>(data.get());
    d->xu.tail(n_env) = env_val;
  }

  void calc(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u) {
    Data* d = static_cast<Data*>(data.get());
    std::size_t nx = ad_model->get_state()->get_nx();
    std::size_t nu = ad_model->get_nu();

    d->xu.head(nx) = x;
    d->xu.segment(nx, nu) = u;

    calcFun_ptr->ForwardZero(d->xu, d->calcout);
    d->distribute_calcout();
  }

  void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u) {
    Data* d = static_cast<Data*>(data.get());
    std::size_t nx = ad_model->get_state()->get_nx();
    std::size_t nu = ad_model->get_nu();

    d->xu.head(nx) = x;
    d->xu.segment(nx, nu) = u;
    calcDiffFun_ptr->ForwardZero(d->xu, d->calcDiffout);
    d->distribute_calcDiffout();
  }

  boost::shared_ptr<ActionDataAbstract> createData() {
    return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
  }

  /// \brief Dimension of the input vector
  Eigen::DenseIndex getInputDimension() const { return ad_X.size(); }

 protected:
  using Base::has_control_limits_;  //!< Indicates whether any of the control limits
  using Base::nr_;                  //!< Dimension of the cost residual
  using Base::nu_;                  //!< Control dimension
  using Base::state_;               //!< Model of the state
  using Base::u_lb_;                //!< Lower control limits
  using Base::u_ub_;                //!< Upper control limits
  using Base::unone_;               //!< Neutral state

  boost::shared_ptr<Base> model;
  boost::shared_ptr<ADBase> ad_model;
  boost::shared_ptr<ADActionDataAbstract> ad_data;

  /// \brief Name of the function
  const std::string function_name_calc, function_name_calcDiff;

  /// \brief Name of the library
  const std::string library_name;

  /// \brief Size of the environment variables
  const std::size_t n_env;

  /// \brief A function that updates the environment variables before starting record.
  std::function<void(boost::shared_ptr<ADBase>, const Eigen::Ref<const ADVectorXs>&)> fn_record_env;

  /// \brief Options to generate or not the source code for the evaluation function
  bool build_forward;

  ADVectorXs ad_X, ad_X2;

  ADVectorXs ad_calcout;
  ADVectorXs ad_calcDiffout;

  ADFun ad_calc, ad_calcDiff;

  std::unique_ptr<CppAD::cg::ModelCSourceGen<Scalar> > calcgen_ptr, calcDiffgen_ptr;
  std::unique_ptr<CppAD::cg::ModelLibraryCSourceGen<Scalar> > libcgen_ptr;
  std::unique_ptr<CppAD::cg::DynamicModelLibraryProcessor<Scalar> > dynamicLibManager_ptr;
  std::unique_ptr<CppAD::cg::DynamicLib<Scalar> > dynamicLib_ptr;
  std::unique_ptr<CppAD::cg::GenericModel<Scalar> > calcFun_ptr, calcDiffFun_ptr;

};  // struct CodeGenBase

template <typename _Scalar>
struct ActionDataCodeGenTpl : public ActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActionDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

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

  VectorXs xu, calcout;

  VectorXs calcDiffout;

  void distribute_calcout() {
    cost = calcout[0];
    xnext = calcout.tail(xnext.size());
  }

  void distribute_calcDiffout() {
    VectorXs& Y = calcDiffout;
    std::size_t ndx = Fx.rows();
    std::size_t nu = Fu.cols();

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

  template <template <typename Scalar> class Model>
  explicit ActionDataCodeGenTpl(Model<Scalar>* const model) : Base(model), calcout(model->get_state()->get_nx() + 1) {
    ActionModelCodeGenTpl<Scalar>* m = static_cast<ActionModelCodeGenTpl<Scalar>*>(model);
    xu.resize(m->getInputDimension());
    xu.setZero();
    calcout.setZero();
    std::size_t ndx = model->get_state()->get_ndx();
    std::size_t nu = model->get_nu();
    calcDiffout.resize(2 * ndx * ndx + 2 * ndx * nu + nu * nu + ndx + nu);
    calcDiffout.setZero();
  }
};

}  // namespace crocoddyl

#endif  // ifndef CROCODDYL_CORE_CODEGEN_ACTION_BASE_HPP_
