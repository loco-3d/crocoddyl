
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, LAAS-CNRS, INRIA, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_CODEGEN_ACTION_HPP_
#define CROCODDYL_CORE_CODEGEN_ACTION_HPP_

#ifdef CROCODDYL_WITH_CODEGEN

#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/codegen/common.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"
#include "crocoddyl/core/utils/stop-watch.hpp"

namespace crocoddyl {

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
  typedef ParameterManagerTpl<Scalar> ParameterManager;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  typedef CppAD::cg::CG<Scalar> CGScalar;
  typedef CppAD::AD<CGScalar> ADScalar;
  typedef MathBaseTpl<ADScalar> ADMathBase;
  typedef ActionModelAbstractTpl<ADScalar> ADBase;
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
   * @param[in] autodiff      Generate autodiff Jacobians and use calcDiff
   * Gauss-Newton Hessian blocks (default false)
   * @param[in] np            Dimension of the parameter variables in the calc
   * and calcDiff functions
   * @param[in] updateParams  Function used to update the calc and calcDiff's
   * parameters (default empty function)
   * @param[in] compiler      Type of compiler GCC or CLANG (default: compiler
   * used to build Crocoddyl)
   * @param[in] compile_options  Compilation flags (default: "-O3 -ffast-math
   * -march=native")
   */
  ActionModelCodeGenTpl(
      std::shared_ptr<Base> model, const std::string& lib_fname,
      bool autodiff = false, const std::size_t np = 0,
      ParamsEnvironment updateParams = EmptyParamsEnv,
      CompilerType compiler = defaultCompilerType(),
      const std::string& compile_options = "-O3 -ffast-math -march=native");

  /**
   * @brief Initialize the code generated action model from an AD model
   *
   * @param[in] ad_model      Action model used to code generate
   * @param[in] lib_fname     Name of the code generated library
   * @param[in] autodiff      Generate autodiff Jacobians and use calcDiff
   * Gauss-Newton Hessian blocks (default false)
   * @param[in] np            Dimension of the parameter variables in the calc
   * and calcDiff functions
   * @param[in] updateParams  Function used to update the calc and calcDiff's
   * parameters (default empty function)
   * @param[in] compiler      Type of compiler GCC or CLANG (default: compiler
   * used to build Crocoddyl)
   * @param[in] compile_options  Compilation flags (default: "-O3 -ffast-math
   * -march=native")
   */
  ActionModelCodeGenTpl(
      std::shared_ptr<ADBase> ad_model, const std::string& lib_fname,
      bool autodiff = false, const std::size_t np = 0,
      ParamsEnvironment updateParams = EmptyParamsEnv,
      CompilerType compiler = defaultCompilerType(),
      const std::string& compile_options = "-O3 -ffast-math -march=native");

  /**
   * @brief Initialize the code generated action model from an pre-compiled
   * library
   *
   * @param lib_fname  Name of the code generated library
   * @param model      Action model which we want to code generate
   */
  ActionModelCodeGenTpl(const std::string& lib_fname,
                        std::shared_ptr<Base> model);

  /**
   * @brief Initialize the code generated action model from an pre-compiled
   * library
   *
   * @param lib_fname  Name of the code generated library
   * @param model      Action model which we want to code generate
   */
  ActionModelCodeGenTpl(const std::string& lib_fname,
                        std::shared_ptr<ADBase> ad_model);

  /**
   * @brief Copy constructor
   * @param other  Action model to be copied
   */
  ActionModelCodeGenTpl(const ActionModelCodeGenTpl<Scalar>& other);

  virtual ~ActionModelCodeGenTpl() = default;

  /**
   * @brief Initialize the code-generated library
   */
  void initLib();

  /**
   * @brief Compile the code-generated library
   */
  void compileLib();

  /**
   * @brief Check if the code-generated library exists
   *
   * @param lib_fname  Name of the code generated library
   * @return Return true if the code-generated library exists, otherwise false.
   */
  bool existLib(const std::string& lib_fname) const;

  /**
   * @brief Load the code-generated library
   *
   * @param lib_fname  Name of the code generated library
   */
  void loadLib(const std::string& lib_fname);

  /**
   * @brief Update the parameters of the codegen action
   *
   * @param data  Action data
   * @param p     Parameters vector (dimension np)
   */
  void update_p(const std::shared_ptr<ActionDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& p) override;

  void set_params(const std::shared_ptr<ActionDataAbstract>& data,
                  std::shared_ptr<ParameterManager> params) override;

  /**
   * @brief Compute the next state and cost value using a code-generated library
   *
   * @param[in] data  Action data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Compute the cost value and constraint infeasibilities for terminal
   * nodes using a code-generated library
   */
  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;

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
                        const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Compute the derivatives of cost functions for terminal nodes using a
   * code-generated library
   */
  virtual void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Create the data for the code-generated action
   *
   * @return the action data
   */
  virtual std::shared_ptr<ActionDataAbstract> createData() override;
  virtual std::shared_ptr<ActionDataAbstract> createData(
      const std::shared_ptr<ParameterDataManagerTpl<Scalar>>& params_data)
      override;

  /**
   * @brief Checks that a specific data belongs to this model
   */
  virtual bool checkData(
      const std::shared_ptr<ActionDataAbstract>& data) override;

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
   * @param[in] maxiter Maximum allowed number of iterations (default 100)
   * @param[in] tol     Tolerance (default 1e-9)
   */
  virtual void quasiStatic(const std::shared_ptr<ActionDataAbstract>& data,
                           Eigen::Ref<VectorXs> u,
                           const Eigen::Ref<const VectorXs>& x,
                           const std::size_t /*maxiter = 100*/,
                           const Scalar /*tol*/) override;

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
  ActionModelCodeGenTpl<NewScalar> cast() const;

  /**
   * @brief Return the action model
   */
  const std::shared_ptr<Base>& get_model() const;

  std::size_t get_np() const override;

  /**
   * @brief Return the number of inequality constraints
   */
  virtual std::size_t get_ng() const override;

  /**
   * @brief Return the number of equality constraints
   */
  virtual std::size_t get_nh() const override;

  /**
   * @brief Return the number of inequality terminal constraints
   */
  virtual std::size_t get_ng_T() const override;

  /**
   * @brief Return the number of equality terminal constraints
   */
  virtual std::size_t get_nh_T() const override;

  /**
   * @brief Return the lower bound of the inequality constraints
   */
  virtual const VectorXs& get_g_lb() const override;

  /**
   * @brief Return the upper bound of the inequality constraints
   */
  virtual const VectorXs& get_g_ub() const override;

  /**
   * @brief Return the dimension of the dependent vector used by calc and
   * calcDiff functions
   */
  std::size_t get_nX() const;

  /**
   * @brief Return the dimension of the dependent vector used by calc and
   * calcDiff functions in terminal nodes
   */
  std::size_t get_nX_T() const;

  /**
   * @brief Return the dimension of the independent vector used by the
   * autodiff calcDiff function
   */
  std::size_t get_nX2() const;

  /**
   * @brief Return the dimension of the independent vector used by the
   * autodiff calcDiff function in terminal nodes
   */
  std::size_t get_nX2_T() const;

  /**
   * @brief Return the dimension of the dependent vector used by the quasiStatic
   * function
   */
  std::size_t get_nX3() const;

  /**
   * @brief Return the dimension of the independent vector used by the calc
   * function
   */
  std::size_t get_nY1() const;

  /**
   * @brief Return the dimension of the independent vector used by the calc
   * function in terminal nodes
   */
  std::size_t get_nY1_T() const;

  /**
   * @brief Return the dimension of the independent vector used by the calcDiff
   * function
   */
  std::size_t get_nY2() const;

  /**
   * @brief Return the dimension of the independent vector used by the calcDiff
   * function in terminal nodes
   */
  std::size_t get_nY2_T() const;

  /**
   * @brief Return the dimension of the local cost-Hessian block vector
   */
  std::size_t get_nYh() const;

  /**
   * @brief Return the dimension of the terminal local cost-Hessian block vector
   */
  std::size_t get_nYh_T() const;

  /**
   * @brief Return the dimension of the independent vector used by the
   * quasiStatic function
   */
  std::size_t get_nY3() const;

  /**
   * @brief Print relevant information of the action model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override;

 protected:
  ActionModelCodeGenTpl();

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
  std::shared_ptr<ADActionDataAbstract>
      ad_data_pert_;  //! Perturbed action data used by autodiff tapes
  bool autodiff_;  //! True if we are genering the automatic derivatives of calc
                   //! funnctions

  std::size_t np_;  //!< Dimension of the parameter variables in the calc and
                    //!< calcDiff functions
  std::size_t nX_;  //!< Dimension of the independent variables used by the calc
                    //!< and calcDiff functions
  std::size_t nX_T_;   //!< Dimension of the independent variables used by the
                       //!< calc and calcDiff functions in terminal nodes
  std::size_t nX2_;    //!< Dimension of the independent variables used by the
                       //!< autodiff calcDiff function
  std::size_t nX2_T_;  //!< Dimension of the independent variables used by the
                       //!< autodiff calcDiff function in terminal nodes
  std::size_t nX3_;    //!< Dimension of the independent variables used by the
                       //!< quasiStatic function
  std::size_t
      nY1_;  //!< Dimension of the dependent variables used by the calc function
  std::size_t nY1_T_;   //!< Dimension of the dependent variables used by the
                        //!< calc function in terminal nodes
  std::size_t nY2_;     //!< Dimension of the dependent variables used by the
                        //!< calcDiff function
  std::size_t nY2_T_;   //!< Dimension of the dependent variables used by the
                        //!< calcDiff function in terminal nodes
  std::size_t nYh_;     //!< Dimension of the local cost-Hessian block output
  std::size_t nYh_T_;   //!< Dimension of the terminal local cost-Hessian blocks
  std::size_t nY3_;     //!< Dimension of the dependent variables used by the
                        //!< quasiStatic function
  ADVectorXs ad_X_;     //!< Independent variables used to tape the calc and
                        //!< calcDiff functions
  ADVectorXs ad_X_T_;   //!< Independent variables used to tape the calc and
                        //!< calcDiff functions in terminal nodes
  ADVectorXs ad_X2_;    //!< Independent variables used to tape the autodiff
                        //!< calcDiff function
  ADVectorXs ad_X2_T_;  //!< Independent variables used to tape the autodiff
                        //!< calcDiff function in terminal nodes
  ADVectorXs ad_X3_;    //!< Independent variables used to tape quasiStatic
                        //!< function
  ADVectorXs ad_Y1_;    //!< Dependent variables used to tape the calc function
  ADVectorXs ad_Y1_T_;  //!< Dependent variables used to tape the calc function
                        //!< in terminal nodes
  ADVectorXs
      ad_Y2_;  //!< Dependent variables used to tape the calcDiff function
  ADVectorXs ad_Y2_T_;  //!< Dependent variables used to tape the calcDiff
                        //!< function in terminal nodes
  ADVectorXs
      ad_Y3_;  //!< Dependent variables used to tape the quasiStatic function

  const std::string Y1fun_name_;  //!< Name of the calc function
  const std::string
      Y1Tfun_name_;  //!< Name of the calc function used in terminal nodes
  const std::string Y2fun_name_;  //!< Name of the calcDiff function
  const std::string
      Y2Tfun_name_;  //!< Name of the calcDiff function used in terminal nodes
  const std::string
      Y2Costfun_name_;  //!< Name of the local cost Hessian block function
  const std::string
      Y2CostTfun_name_;  //!< Name of the terminal local cost Hessian function
  const std::string Y3fun_name_;       //!< Name of the quasiStatic function
  const std::string lib_fname_;        //!< Name of the code generated library
  CompilerType compiler_type_;         //!< Type of compiler
  const std::string compile_options_;  //!< Compilation options

  ParamsEnvironment updateParams_;  // Lambda function that updates parameter
                                    // variables before starting record.
  std::shared_ptr<ParameterManager>
      params_;  //!< Parameter manager used to initialize generated data

  std::unique_ptr<ADFun> ad_calc_;  //! < Function used to code generate calc
  std::unique_ptr<ADFun>
      ad_calc_T_;  //! < Function used to code generate calc in terminal nodes
  std::unique_ptr<ADFun>
      ad_calcDiff_;  //!< Function used to code generate calcDiff
  std::unique_ptr<ADFun> ad_calcDiff_T_;  //!< Function used to code generate
                                          //!< the calcDiff in terminal nodes
  std::unique_ptr<ADFun>
      ad_calcDiffCost_;  //!< Local cost Hessian block function
  std::unique_ptr<ADFun>
      ad_calcDiffCost_T_;  //!< Terminal local cost Hessian block function
  std::unique_ptr<ADFun>
      ad_quasiStatic_;  //! < Function used to code generate quasiStatic
  std::unique_ptr<CSourceGen>
      calcCG_;  //!< Code generated source code of the calc function
  std::unique_ptr<CSourceGen>
      calcCG_T_;  //!< Code generated source code of
                  //!< the calc function in terminal nodes
  std::unique_ptr<CSourceGen>
      calcDiffCG_;  //!< Code generated source code of the calcDiff function
  std::unique_ptr<CSourceGen>
      calcDiffCG_T_;  //!< Code generated source code of the calcDiff function
                      //!< in terminal nodes
  std::unique_ptr<CSourceGen>
      calcDiffCostCG_;  //!< Code generated local cost Hessian block function
  std::unique_ptr<CSourceGen>
      calcDiffCostCG_T_;  //!< Code generated terminal local cost Hessian
                          //!< block function
  std::unique_ptr<CSourceGen> quasiStaticCG_;  //!< Code generated source code
                                               //!< of the quasiStatic function
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
  std::unique_ptr<GenericModel>
      calcDiffCostFun_;  //!< Code generated local cost Hessian block function
  std::unique_ptr<GenericModel> calcDiffCostFun_T_;  //!< Code generated
                                                     //!< terminal local cost
                                                     //!< Hessian block function
  std::unique_ptr<GenericModel>
      quasiStaticFun_;  //!< Code generated quasiStatic function

 private:
  void recordCalc();
  void recordCalc_T();
  void recordCalcDiff();
  void recordCalcDiff_T();
  void recordCalcDiffCost();
  void recordCalcDiffCost_T();
  void recordQuasiStatic();
  void recordParams(const Eigen::Ref<const ADVectorXs>& p);
  void recordParams(const std::shared_ptr<ADActionDataAbstract>& data,
                    const Eigen::Ref<const ADVectorXs>& p);
  void syncDataParameters(Data* const data) const;

  void tapeCalcOutput();
  void tapeCalcOutput_T();
  void tapeCalcDiffOutput();
  void tapeCalcDiffOutput_T();
  void tapeCalcDiffHessianOutput(ADVectorXs& y);
  void tapeCalcDiffHessianOutput_T(ADVectorXs& y);

  VectorXs wCostHess_;
  VectorXs wCostHess_T_;
  VectorXs wCostHessScalar_;
  VectorXs wCostHessScalar_T_;

  static void EmptyParamsEnv(std::shared_ptr<ADBase>,
                             const Eigen::Ref<const ADVectorXs>&);
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
    X_T.resize(m->get_nX_T());
    X2.resize(m->get_nX2());
    X2_T.resize(m->get_nX2_T());
    X3.resize(m->get_nX3());
    Y1.resize(m->get_nY1());
    J1.resize(m->get_nY1() * m->get_nX());
    H1.resize(m->get_nX() * m->get_nX());
    Y1_T.resize(m->get_nY1_T());
    J1_T.resize(m->get_nY1_T() * m->get_nX_T());
    H1_T.resize(m->get_nX_T() * m->get_nX_T());
    J2.resize(m->get_nY2() * m->get_nX2());
    H2.resize(m->get_nX2() * m->get_nX2());
    J2_T.resize(m->get_nY2_T() * m->get_nX2_T());
    H2_T.resize(m->get_nX2_T() * m->get_nX2_T());
    Y2.resize(m->get_nY2());
    Y2_T.resize(m->get_nY2_T());
    Yh.resize(m->get_nYh());
    Yh_T.resize(m->get_nYh_T());
    Y3.resize(m->get_nY3());
    X.setZero();
    X_T.setZero();
    X2.setZero();
    X2_T.setZero();
    X3.setZero();
    Y1.setZero();
    J1.setZero();
    H1.setZero();
    Y1_T.setZero();
    J1_T.setZero();
    H1_T.setZero();
    J2.setZero();
    H2.setZero();
    J2_T.setZero();
    H2_T.setZero();
    Y2.setZero();
    Y2_T.setZero();
    Yh.setZero();
    Yh_T.setZero();
    Y3.setZero();
  }

  using Base::cost;
  using Base::Fp;
  using Base::Fu;
  using Base::Fx;
  using Base::g;
  using Base::Gp;
  using Base::Gu;
  using Base::Gx;
  using Base::h;
  using Base::Hp;
  using Base::Hu;
  using Base::Hx;
  using Base::Lp;
  using Base::Lpp;
  using Base::Lpu;
  using Base::Lpx;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::r;
  using Base::xnext;

  VectorXs
      X;  //!< Independent variables used by the calc and calcDiff functions
  VectorXs X_T;   //!< Independent variables used by the calc and calcDiff
                  //!< functions in terminal nodes
  VectorXs X2;    //!< Independent variables used by the autodiff calcDiff
                  //!< function
  VectorXs X2_T;  //!< Independent variables used by the autodiff calcDiff
                  //!< function in terminal nodes
  VectorXs X3;    //!< Independent variables used by the quasiStatic function
  VectorXs Y1;    //!< Dependent variables used by the calc function
  VectorXs J1;    //!< Autodiff Jacobian of the calc function
  VectorXs H1;    //!< Autodiff Hessianb of the calc function
  VectorXs Y1_T;  //!< Dependent variables used by the calc function in terminal
  VectorXs J1_T;  //!< Autodiff Jacobian of the calc function in terminal nodes
  VectorXs H1_T;  //!< Autodiff Hessianb of the calc function in terminal nodes
                  //!< nodes
  VectorXs J2;    //!< Autodiff Jacobian of the tangent calcDiff function
  VectorXs H2;    //!< Legacy tangent calcDiff Hessian buffer
  VectorXs J2_T;  //!< Autodiff Jacobian of the terminal tangent calcDiff
                  //!< function
  VectorXs H2_T;  //!< Legacy terminal tangent calcDiff Hessian buffer
  VectorXs Y2;    //!< Dependent variables used by the calcDiff function
  VectorXs Y2_T;  //!< Dependent variables used by the calcDiff function in
                  //!< terminal nodes
  VectorXs Yh;    //!< Local cost-Hessian blocks used by autodiff calcDiff
  VectorXs Yh_T;  //!< Terminal local cost-Hessian blocks used by autodiff
  VectorXs Y3;    //!< Dependent variables used by the quasiStatic function
  std::shared_ptr<ParameterDataManagerTpl<Scalar>> params_data;

  template <template <typename Scalar> class Model>
  void set_Y1(Model<Scalar>* const model) {
    const std::size_t nx = model->get_state()->get_nx();
    const std::size_t ng = model->get_ng();
    const std::size_t nh = model->get_nh();
    Eigen::DenseIndex it_Y1 = 0;
    cost = Y1[it_Y1];
    it_Y1 += 1;
    xnext = Y1.segment(it_Y1, nx);
    it_Y1 += nx;
    g = Y1.segment(it_Y1, ng);
    it_Y1 += ng;
    h = Y1.segment(it_Y1, nh);
  }

  template <template <typename Scalar> class Model>
  void set_D1(Model<Scalar>* const model) {
    const std::size_t ndx = model->get_state()->get_ndx();
    const std::size_t nu = model->get_nu();
    const std::size_t ng = model->get_ng();
    const std::size_t nh = model->get_nh();
    const std::size_t np = model->get_np();
    const std::size_t nxu = ndx + nu + np;
    Eigen::DenseIndex it_J1 = 0;
    Lx = Eigen::Map<VectorXs>(J1.data() + it_J1, ndx);
    it_J1 += ndx;
    Lu = Eigen::Map<VectorXs>(J1.data() + it_J1, nu);
    it_J1 += nu;
    Lp = Eigen::Map<VectorXs>(J1.data() + it_J1, np);
    it_J1 += np;
    Eigen::Map<MatrixXs> J1_map(J1.data() + it_J1, nxu, ndx);
    Fx = J1_map.topRows(ndx).transpose();
    Fu = J1_map.middleRows(ndx, nu).transpose();
    Fp = J1_map.bottomRows(np).transpose();
    it_J1 += ndx * nxu;
    Eigen::Map<MatrixXs> G_map(J1.data() + it_J1, nxu, ng);
    Gx = G_map.topRows(ndx).transpose();
    Gu = G_map.middleRows(ndx, nu).transpose();
    Gp = G_map.bottomRows(np).transpose();
    it_J1 += ng * nxu;
    Eigen::Map<MatrixXs> H_map(J1.data() + it_J1, nxu, nh);
    Hx = H_map.topRows(ndx).transpose();
    Hu = H_map.middleRows(ndx, nu).transpose();
    Hp = H_map.bottomRows(np).transpose();
    Eigen::Map<MatrixXs> H1_map(H1.data(), nxu, nxu);
    Lxx = H1_map.topLeftCorner(ndx, ndx);
    Luu = H1_map.middleCols(ndx, nu).middleRows(ndx, nu);
    Lxu = H1_map.middleCols(ndx, nu).topRows(ndx);
    Lpp = H1_map.bottomRightCorner(np, np);
    Lpx = H1_map.bottomRows(np).leftCols(ndx);
    Lpu = H1_map.bottomRows(np).middleCols(ndx, nu);
  }

  template <template <typename Scalar> class Model>
  void set_Y1_T(Model<Scalar>* const model) {
    const std::size_t ng = model->get_ng_T();
    const std::size_t nh = model->get_nh_T();
    Eigen::DenseIndex it_Y1 = 0;
    cost = Y1_T[it_Y1];
    it_Y1 += 1;
    g.setZero();
    if (ng != 0u) {
      g.head(ng) = Y1_T.segment(it_Y1, ng);
    }
    it_Y1 += ng;
    h.setZero();
    if (nh != 0u) {
      h.head(nh) = Y1_T.segment(it_Y1, nh);
    }
  }

  template <template <typename Scalar> class Model>
  void set_D1_T(Model<Scalar>* const model) {
    const std::size_t ndx = model->get_state()->get_ndx();
    const std::size_t ng = model->get_ng_T();
    const std::size_t nh = model->get_nh_T();
    const std::size_t np = model->get_np();
    const std::size_t nxp = ndx + np;
    Eigen::DenseIndex it_J1 = 0;
    Lx = Eigen::Map<VectorXs>(J1_T.data() + it_J1, ndx);
    it_J1 += ndx;
    Lp = Eigen::Map<VectorXs>(J1_T.data() + it_J1, np);
    it_J1 += np;
    Gx.setZero();
    Gp.setZero();
    if (ng != 0u) {
      Eigen::Map<MatrixXs> G_map(J1_T.data() + it_J1, nxp, ng);
      Gx.topRows(ng) = G_map.topRows(ndx).transpose();
      Gp.topRows(ng) = G_map.bottomRows(np).transpose();
    }
    it_J1 += ng * nxp;
    Hx.setZero();
    Hp.setZero();
    if (nh != 0u) {
      Eigen::Map<MatrixXs> H_map(J1_T.data() + it_J1, nxp, nh);
      Hx.topRows(nh) = H_map.topRows(ndx).transpose();
      Hp.topRows(nh) = H_map.bottomRows(np).transpose();
    }
    Lxx = Eigen::Map<MatrixXs>(H1_T.data(), nxp, nxp).topLeftCorner(ndx, ndx);
    Lpp = Eigen::Map<MatrixXs>(H1_T.data(), nxp, nxp).bottomRightCorner(np, np);
    Lpx = Eigen::Map<MatrixXs>(H1_T.data(), nxp, nxp)
              .bottomRows(np)
              .leftCols(ndx);
  }

  template <template <typename Scalar> class Model>
  void set_D2(Model<Scalar>* const model) {
    const std::size_t ndx = model->get_state()->get_ndx();
    const std::size_t nu = model->get_nu();
    const std::size_t ng = model->get_ng();
    const std::size_t nh = model->get_nh();
    const std::size_t np = model->get_np();
    const std::size_t nbase = model->get_nX();
    const std::size_t ninput = model->get_nX2();
    const std::size_t npert = ndx + nu + np;
    Eigen::Map<MatrixXs> J_map(J2.data(), ninput, model->get_nY2());
    const auto J_pert = J_map.middleRows(nbase, npert);
    Lx = J_pert.col(0).head(ndx);
    Lu = J_pert.col(0).segment(ndx, nu);
    Lp = J_pert.col(0).tail(np);
    Eigen::Map<MatrixXs> F_map(J2.data() + ninput, ninput, ndx);
    Fx = F_map.middleRows(nbase, npert).topRows(ndx).transpose();
    Fu = F_map.middleRows(nbase + ndx, nu).transpose();
    Fp = F_map.middleRows(nbase + ndx + nu, np).transpose();
    Eigen::Map<MatrixXs> G_map(J2.data() + ninput * (1 + ndx), ninput, ng);
    Gx = G_map.middleRows(nbase, npert).topRows(ndx).transpose();
    Gu = G_map.middleRows(nbase + ndx, nu).transpose();
    Gp = G_map.middleRows(nbase + ndx + nu, np).transpose();
    Eigen::Map<MatrixXs> H_map(J2.data() + ninput * (1 + ndx + ng), ninput, nh);
    Hx = H_map.middleRows(nbase, npert).topRows(ndx).transpose();
    Hu = H_map.middleRows(nbase + ndx, nu).transpose();
    Hp = H_map.middleRows(nbase + ndx + nu, np).transpose();
    Eigen::Map<MatrixXs> H2_map(H2.data(), ninput, ninput);
    Lxx = H2_map.block(nbase, nbase, ndx, ndx);
    Lxu = H2_map.block(nbase, nbase + ndx, ndx, nu);
    Luu = H2_map.block(nbase + ndx, nbase + ndx, nu, nu);
    Lpp = H2_map.block(nbase + ndx + nu, nbase + ndx + nu, np, np);
    Lpx = H2_map.block(nbase + ndx + nu, nbase, np, ndx);
    Lpu = H2_map.block(nbase + ndx + nu, nbase + ndx, np, nu);
  }

  template <template <typename Scalar> class Model>
  void set_D2_T(Model<Scalar>* const model) {
    const std::size_t ndx = model->get_state()->get_ndx();
    const std::size_t ng = model->get_ng_T();
    const std::size_t nh = model->get_nh_T();
    const std::size_t np = model->get_np();
    const std::size_t nbase = model->get_nX_T();
    const std::size_t ninput = model->get_nX2_T();
    const std::size_t npert = ndx + np;
    Eigen::Map<MatrixXs> J_map(J2_T.data(), ninput, model->get_nY2_T());
    const auto J_pert = J_map.middleRows(nbase, npert);
    Lx = J_pert.col(0).head(ndx);
    Lp = J_pert.col(0).tail(np);
    Eigen::Map<MatrixXs> F_map(J2_T.data() + ninput, ninput, ndx);
    Fx = F_map.middleRows(nbase, ndx).transpose();
    Gx.setZero();
    Gp.setZero();
    if (ng != 0u) {
      Eigen::Map<MatrixXs> G_map(J2_T.data() + ninput * (1 + ndx), ninput, ng);
      Gx.topRows(ng) = G_map.middleRows(nbase, ndx).transpose();
      Gp.topRows(ng) = G_map.middleRows(nbase + ndx, np).transpose();
    }
    Hx.setZero();
    Hp.setZero();
    if (nh != 0u) {
      Eigen::Map<MatrixXs> H_map(J2_T.data() + ninput * (1 + ndx + ng), ninput,
                                 nh);
      Hx.topRows(nh) = H_map.middleRows(nbase, ndx).transpose();
      Hp.topRows(nh) = H_map.middleRows(nbase + ndx, np).transpose();
    }
    Eigen::Map<MatrixXs> H2_map(H2_T.data(), ninput, ninput);
    Lxx = H2_map.block(nbase, nbase, ndx, ndx);
    Lpp = H2_map.block(nbase + ndx, nbase + ndx, np, np);
    Lpx = H2_map.block(nbase + ndx, nbase, np, ndx);
  }

  template <template <typename Scalar> class Model>
  void set_Yh_hessian(Model<Scalar>* const model) {
    const std::size_t ndx = model->get_state()->get_ndx();
    const std::size_t nu = model->get_nu();
    const std::size_t np = model->get_np();
    Eigen::DenseIndex it_Yh = 0;
    Lxx = Eigen::Map<MatrixXs>(Yh.data() + it_Yh, ndx, ndx);
    it_Yh += ndx * ndx;
    Lxu = Eigen::Map<MatrixXs>(Yh.data() + it_Yh, ndx, nu);
    it_Yh += ndx * nu;
    Luu = Eigen::Map<MatrixXs>(Yh.data() + it_Yh, nu, nu);
    it_Yh += nu * nu;
    Lpp = Eigen::Map<MatrixXs>(Yh.data() + it_Yh, np, np);
    it_Yh += np * np;
    Lpx = Eigen::Map<MatrixXs>(Yh.data() + it_Yh, np, ndx);
    it_Yh += np * ndx;
    Lpu = Eigen::Map<MatrixXs>(Yh.data() + it_Yh, np, nu);
  }

  template <template <typename Scalar> class Model>
  void set_Yh_hessian_T(Model<Scalar>* const model) {
    const std::size_t ndx = model->get_state()->get_ndx();
    const std::size_t np = model->get_np();
    Eigen::DenseIndex it_Yh = 0;
    Lxx = Eigen::Map<MatrixXs>(Yh_T.data() + it_Yh, ndx, ndx);
    it_Yh += ndx * ndx;
    Lpp = Eigen::Map<MatrixXs>(Yh_T.data() + it_Yh, np, np);
    it_Yh += np * np;
    Lpx = Eigen::Map<MatrixXs>(Yh_T.data() + it_Yh, np, ndx);
  }

  template <template <typename Scalar> class Model>
  void set_Y2(Model<Scalar>* const model) {
    const std::size_t ndx = model->get_state()->get_ndx();
    const std::size_t nu = model->get_nu();
    const std::size_t ng = model->get_ng();
    const std::size_t nh = model->get_nh();
    const std::size_t np = model->get_np();
    Eigen::DenseIndex it_Y2 = 0;
    Fx = Eigen::Map<MatrixXs>(Y2.data() + it_Y2, ndx, ndx);
    it_Y2 += ndx * ndx;
    Fu = Eigen::Map<MatrixXs>(Y2.data() + it_Y2, ndx, nu);
    it_Y2 += ndx * nu;
    Fp = Eigen::Map<MatrixXs>(Y2.data() + it_Y2, ndx, np);
    it_Y2 += ndx * np;
    Lx = Eigen::Map<VectorXs>(Y2.data() + it_Y2, ndx);
    it_Y2 += ndx;
    Lu = Eigen::Map<VectorXs>(Y2.data() + it_Y2, nu);
    it_Y2 += nu;
    Lp = Eigen::Map<VectorXs>(Y2.data() + it_Y2, np);
    it_Y2 += np;
    Lxx = Eigen::Map<MatrixXs>(Y2.data() + it_Y2, ndx, ndx);
    it_Y2 += ndx * ndx;
    Lxu = Eigen::Map<MatrixXs>(Y2.data() + it_Y2, ndx, nu);
    it_Y2 += ndx * nu;
    Luu = Eigen::Map<MatrixXs>(Y2.data() + it_Y2, nu, nu);
    it_Y2 += nu * nu;
    Lpp = Eigen::Map<MatrixXs>(Y2.data() + it_Y2, np, np);
    it_Y2 += np * np;
    Lpx = Eigen::Map<MatrixXs>(Y2.data() + it_Y2, np, ndx);
    it_Y2 += np * ndx;
    Lpu = Eigen::Map<MatrixXs>(Y2.data() + it_Y2, np, nu);
    it_Y2 += np * nu;
    Gx = Eigen::Map<MatrixXs>(Y2.data() + it_Y2, ng, ndx);
    it_Y2 += ng * ndx;
    Gu = Eigen::Map<MatrixXs>(Y2.data() + it_Y2, ng, nu);
    it_Y2 += ng * nu;
    Gp = Eigen::Map<MatrixXs>(Y2.data() + it_Y2, ng, np);
    it_Y2 += ng * np;
    Hx = Eigen::Map<MatrixXs>(Y2.data() + it_Y2, nh, ndx);
    it_Y2 += nh * ndx;
    Hu = Eigen::Map<MatrixXs>(Y2.data() + it_Y2, nh, nu);
    it_Y2 += nh * nu;
    Hp = Eigen::Map<MatrixXs>(Y2.data() + it_Y2, nh, np);
  }

  template <template <typename Scalar> class Model>
  void set_Y2_T(Model<Scalar>* const model) {
    const std::size_t ndx = model->get_state()->get_ndx();
    const std::size_t ng = model->get_ng_T();
    const std::size_t nh = model->get_nh_T();
    const std::size_t np = model->get_np();
    Eigen::DenseIndex it_Y2 = 0;
    Lx = Eigen::Map<VectorXs>(Y2_T.data() + it_Y2, ndx);
    it_Y2 += ndx;
    Lp = Eigen::Map<VectorXs>(Y2_T.data() + it_Y2, np);
    it_Y2 += np;
    Lxx = Eigen::Map<MatrixXs>(Y2_T.data() + it_Y2, ndx, ndx);
    it_Y2 += ndx * ndx;
    Lpp = Eigen::Map<MatrixXs>(Y2_T.data() + it_Y2, np, np);
    it_Y2 += np * np;
    Lpx = Eigen::Map<MatrixXs>(Y2_T.data() + it_Y2, np, ndx);
    it_Y2 += np * ndx;
    Gx.setZero();
    if (ng != 0u) {
      Gx.topRows(ng) = Eigen::Map<MatrixXs>(Y2_T.data() + it_Y2, ng, ndx);
    }
    it_Y2 += ng * ndx;
    Gp.setZero();
    if (ng != 0u) {
      Gp.topRows(ng) = Eigen::Map<MatrixXs>(Y2_T.data() + it_Y2, ng, np);
    }
    it_Y2 += ng * np;
    Hx.setZero();
    if (nh != 0u) {
      Hx.topRows(nh) = Eigen::Map<MatrixXs>(Y2_T.data() + it_Y2, nh, ndx);
    }
    it_Y2 += nh * ndx;
    Hp.setZero();
    if (nh != 0u) {
      Hp.topRows(nh) = Eigen::Map<MatrixXs>(Y2_T.data() + it_Y2, nh, np);
    }
  }
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/codegen/action.hxx"

#endif  // CROCODDYL_WITH_CODEGEN

#endif  // CROCODDYL_CORE_CODEGEN_ACTION_HPP_
