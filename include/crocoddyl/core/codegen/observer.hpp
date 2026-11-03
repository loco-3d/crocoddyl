///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, LAAS-CNRS, INRIA, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_CODEGEN_OBSERVER_HPP_
#define CROCODDYL_CORE_CODEGEN_OBSERVER_HPP_

#ifdef CROCODDYL_WITH_CODEGEN

#include "crocoddyl/core/codegen/common.hpp"
#include "crocoddyl/core/integ-observer-base.hpp"
#include "crocoddyl/core/observer-base.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"
#include "crocoddyl/core/utils/stop-watch.hpp"
#include "crocoddyl/multibody/residuals/power.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ObserverModelCodeGenTpl : public ObserverModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_BASE_DERIVED_FLOATINGPOINT_CAST(ActionModelBase,
                                            ObserverModelCodeGenTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ObserverModelAbstractTpl<Scalar> Base;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef ObserverDataAbstractTpl<Scalar> ObserverDataAbstract;
  typedef ObserverDataCodeGenTpl<Scalar> Data;
  typedef ParameterManagerTpl<Scalar> ParameterManager;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  typedef CppAD::cg::CG<Scalar> CGScalar;
  typedef CppAD::AD<CGScalar> ADScalar;
  typedef MathBaseTpl<ADScalar> ADMathBase;
  typedef ObserverModelAbstractTpl<ADScalar> ADBase;
  typedef ObserverDataAbstractTpl<ADScalar> ADObserverDataAbstract;
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

  ObserverModelCodeGenTpl(
      std::shared_ptr<Base> model, const std::string& lib_fname,
      bool autodiff = false, const std::size_t np = 0,
      ParamsEnvironment updateParams = EmptyParamsEnv,
      CompilerType compiler = defaultCompilerType(),
      const std::string& compile_options = "-O3 -ffast-math -march=native");

  ObserverModelCodeGenTpl(
      std::shared_ptr<Base> model, const std::string& lib_fname, bool autodiff,
      const std::size_t np, const std::size_t nenv,
      ParamsEnvironment updateParams,
      CompilerType compiler = defaultCompilerType(),
      const std::string& compile_options = "-O3 -ffast-math -march=native",
      const std::string& state_observation_cost = std::string(),
      const std::string& weight_cost = std::string());

  ObserverModelCodeGenTpl(
      std::shared_ptr<Base> model, const std::string& lib_fname, bool autodiff,
      const std::size_t np, const std::size_t nenv,
      const std::string& state_observation_cost,
      const std::string& weight_cost = std::string(),
      CompilerType compiler = defaultCompilerType(),
      const std::string& compile_options = "-O3 -ffast-math -march=native");

  ObserverModelCodeGenTpl(
      std::shared_ptr<ADBase> ad_model, const std::string& lib_fname,
      bool autodiff = false, const std::size_t np = 0,
      ParamsEnvironment updateParams = EmptyParamsEnv,
      CompilerType compiler = defaultCompilerType(),
      const std::string& compile_options = "-O3 -ffast-math -march=native");

  ObserverModelCodeGenTpl(
      std::shared_ptr<ADBase> ad_model, const std::string& lib_fname,
      bool autodiff, const std::size_t np, const std::size_t nenv,
      ParamsEnvironment updateParams,
      CompilerType compiler = defaultCompilerType(),
      const std::string& compile_options = "-O3 -ffast-math -march=native",
      const std::string& state_observation_cost = std::string(),
      const std::string& weight_cost = std::string());

  ObserverModelCodeGenTpl(
      std::shared_ptr<ADBase> ad_model, const std::string& lib_fname,
      bool autodiff, const std::size_t np, const std::size_t nenv,
      const std::string& state_observation_cost,
      const std::string& weight_cost = std::string(),
      CompilerType compiler = defaultCompilerType(),
      const std::string& compile_options = "-O3 -ffast-math -march=native");

  ObserverModelCodeGenTpl(const std::string& lib_fname,
                          std::shared_ptr<Base> model);
  ObserverModelCodeGenTpl(const std::string& lib_fname,
                          std::shared_ptr<ADBase> ad_model);
  ObserverModelCodeGenTpl(const ObserverModelCodeGenTpl<Scalar>& other);
  virtual ~ObserverModelCodeGenTpl() = default;

  void initLib();
  void compileLib();
  bool existLib(const std::string& lib_fname) const;
  void loadLib(const std::string& lib_fname);

  void update_p(const std::shared_ptr<ActionDataAbstract>& data,
                const Eigen::Ref<const VectorXs>& p) override;
  void update_env(const std::shared_ptr<ActionDataAbstract>& data,
                  const Eigen::Ref<const VectorXs>& env);
  void update_tau(const Eigen::Ref<const VectorXs>& tau_meas) override;
  void set_params(const std::shared_ptr<ActionDataAbstract>& data,
                  std::shared_ptr<ParameterManager> params) override;

  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;
  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;
  virtual void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;
  virtual void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x) override;

  virtual std::shared_ptr<ActionDataAbstract> createData() override;
  virtual std::shared_ptr<ActionDataAbstract> createData(
      const std::shared_ptr<ParameterDataManagerTpl<Scalar>>& params_data)
      override;
  virtual bool checkData(
      const std::shared_ptr<ActionDataAbstract>& data) override;

  const std::shared_ptr<Base>& get_model() const;
  std::size_t get_np() const override;
  std::size_t get_nenv() const;
  virtual std::size_t get_ng() const override;
  virtual std::size_t get_nh() const override;
  virtual std::size_t get_ng_T() const override;
  virtual std::size_t get_nh_T() const override;
  virtual const VectorXs& get_g_lb() const override;
  virtual const VectorXs& get_g_ub() const override;
  std::size_t get_nX() const;
  std::size_t get_nX_T() const;
  std::size_t get_nX2() const;
  std::size_t get_nX2_T() const;
  std::size_t get_nY1() const;
  std::size_t get_nY1_T() const;
  std::size_t get_nY2() const;
  std::size_t get_nY2_T() const;
  std::size_t get_nYh() const;
  std::size_t get_nYh_T() const;
  virtual void print(std::ostream& os) const override;

 protected:
  ObserverModelCodeGenTpl();

  using Base::ng_;
  using Base::ng_T_;
  using Base::nh_;
  using Base::nh_T_;
  using Base::ntau_;
  using Base::nu_;
  using Base::state_;
  using Base::tau_meas_;

  std::shared_ptr<Base> model_;
  std::shared_ptr<ADBase> ad_model_;
  std::shared_ptr<ADObserverDataAbstract> ad_data_;
  std::shared_ptr<ADObserverDataAbstract> ad_data_pert_;
  bool autodiff_;
  std::size_t np_;
  std::size_t nenv_;
  std::size_t nX_;
  std::size_t nX_T_;
  std::size_t nX2_;
  std::size_t nX2_T_;
  std::size_t nY1_;
  std::size_t nY1_T_;
  std::size_t nY2_;
  std::size_t nY2_T_;
  std::size_t nYh_;
  std::size_t nYh_T_;
  ADVectorXs ad_X_;
  ADVectorXs ad_X_T_;
  ADVectorXs ad_X2_;
  ADVectorXs ad_X2_T_;
  ADVectorXs ad_Y1_;
  ADVectorXs ad_Y1_T_;
  ADVectorXs ad_Y2_;
  ADVectorXs ad_Y2_T_;

  const std::string Y1fun_name_;
  const std::string Y1Tfun_name_;
  const std::string Y2fun_name_;
  const std::string Y2Tfun_name_;
  const std::string Y2Costfun_name_;
  const std::string Y2CostTfun_name_;
  const std::string lib_fname_;
  CompilerType compiler_type_;
  const std::string compile_options_;
  ParamsEnvironment updateParams_;
  std::shared_ptr<ParameterManager> params_;

  std::unique_ptr<ADFun> ad_calc_;
  std::unique_ptr<ADFun> ad_calc_T_;
  std::unique_ptr<ADFun> ad_calcDiff_;
  std::unique_ptr<ADFun> ad_calcDiff_T_;
  std::unique_ptr<ADFun> ad_calcDiffCost_;
  std::unique_ptr<ADFun> ad_calcDiffCost_T_;
  std::unique_ptr<CSourceGen> calcCG_;
  std::unique_ptr<CSourceGen> calcCG_T_;
  std::unique_ptr<CSourceGen> calcDiffCG_;
  std::unique_ptr<CSourceGen> calcDiffCG_T_;
  std::unique_ptr<CSourceGen> calcDiffCostCG_;
  std::unique_ptr<CSourceGen> calcDiffCostCG_T_;
  std::unique_ptr<LibraryCSourceGen> libCG_;
  std::unique_ptr<LibraryProcessor> dynLibManager_;
  std::unique_ptr<DynamicLib> dynLib_;
  std::unique_ptr<GenericModel> calcFun_;
  std::unique_ptr<GenericModel> calcFun_T_;
  std::unique_ptr<GenericModel> calcDiffFun_;
  std::unique_ptr<GenericModel> calcDiffFun_T_;
  std::unique_ptr<GenericModel> calcDiffCostFun_;
  std::unique_ptr<GenericModel> calcDiffCostFun_T_;

 private:
  void recordCalc();
  void recordCalc_T();
  void recordCalcDiff();
  void recordCalcDiff_T();
  void recordCalcDiffCost();
  void recordCalcDiffCost_T();
  void recordParams(const Eigen::Ref<const ADVectorXs>& p,
                    const Eigen::Ref<const ADVectorXs>& env);
  void recordParams(const std::shared_ptr<ADObserverDataAbstract>& data,
                    const Eigen::Ref<const ADVectorXs>& p,
                    const Eigen::Ref<const ADVectorXs>& env);
  void setupADParameters();
  void syncDataParameters(Data* const data) const;
  static ParamsEnvironment makeStateObservationEnvironment(
      const std::string& state_observation_cost,
      const std::string& weight_cost);

  void tapeCalcOutput();
  void tapeCalcOutput_T();
  void tapeCalcDiffOutput();
  void tapeCalcDiffOutput_T();
  void tapeCalcDiffDirectOutput(ADVectorXs& y);
  void tapeCalcDiffDirectOutput_T(ADVectorXs& y);
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
struct ObserverDataCodeGenTpl : public ObserverDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ObserverDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit ObserverDataCodeGenTpl(Model<Scalar>* const model) : Base(model) {
    ObserverModelCodeGenTpl<Scalar>* m =
        static_cast<ObserverModelCodeGenTpl<Scalar>*>(model);
    X.resize(m->get_nX());
    X_T.resize(m->get_nX_T());
    X2.resize(m->get_nX2());
    X2_T.resize(m->get_nX2_T());
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
    X.setZero();
    X_T.setZero();
    X2.setZero();
    X2_T.setZero();
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
  }

  using Base::cost;
  using Base::dissipative_E;
  using Base::Ep;
  using Base::Eu;
  using Base::Ex;
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
  using Base::xnext;

  VectorXs X;
  VectorXs X_T;
  VectorXs X2;
  VectorXs X2_T;
  VectorXs Y1;
  VectorXs J1;
  VectorXs H1;
  VectorXs Y1_T;
  VectorXs J1_T;
  VectorXs H1_T;
  VectorXs J2;
  VectorXs H2;
  VectorXs J2_T;
  VectorXs H2_T;
  VectorXs Y2;
  VectorXs Y2_T;
  VectorXs Yh;
  VectorXs Yh_T;
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
    it_Y1 += nh;
    dissipative_E[0] = Y1[it_Y1];
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
    it_Y1 += nh;
    dissipative_E[0] = Y1_T[it_Y1];
  }

  template <template <typename Scalar> class Model>
  void set_D1(Model<Scalar>* const model) {
    const std::size_t ndx = model->get_state()->get_ndx();
    const std::size_t nu = model->get_nu();
    const std::size_t ng = model->get_ng();
    const std::size_t nh = model->get_nh();
    const std::size_t np = model->get_np();
    const std::size_t ntau = model->get_ntau();
    const std::size_t ninput = ndx + nu + np + ntau;
    Eigen::DenseIndex it_J1 = 0;
    Lx = Eigen::Map<VectorXs>(J1.data() + it_J1, ndx);
    it_J1 += ndx;
    Lu = Eigen::Map<VectorXs>(J1.data() + it_J1, nu);
    it_J1 += nu;
    Lp = Eigen::Map<VectorXs>(J1.data() + it_J1, np);
    it_J1 += np + ntau;
    Eigen::Map<MatrixXs> J1_map(J1.data() + it_J1, ninput, ndx);
    Fx = J1_map.topRows(ndx).transpose();
    Fu = J1_map.middleRows(ndx, nu).transpose();
    Fp = J1_map.middleRows(ndx + nu, np).transpose();
    it_J1 += ndx * ninput;
    Eigen::Map<MatrixXs> G_map(J1.data() + it_J1, ninput, ng);
    Gx = G_map.topRows(ndx).transpose();
    Gu = G_map.middleRows(ndx, nu).transpose();
    Gp = G_map.middleRows(ndx + nu, np).transpose();
    it_J1 += ng * ninput;
    Eigen::Map<MatrixXs> H_map(J1.data() + it_J1, ninput, nh);
    Hx = H_map.topRows(ndx).transpose();
    Hu = H_map.middleRows(ndx, nu).transpose();
    Hp = H_map.middleRows(ndx + nu, np).transpose();
    it_J1 += nh * ninput;
    Eigen::Map<VectorXs> dE_map(J1.data() + it_J1, ninput);
    Ex = dE_map.head(ndx).transpose();
    Eu = dE_map.segment(ndx, nu).transpose();
    Ep = dE_map.segment(ndx + nu, np).transpose();
    Eigen::Map<MatrixXs> H1_map(H1.data(), ninput, ninput);
    Lxx = H1_map.topLeftCorner(ndx, ndx);
    Luu = H1_map.middleCols(ndx, nu).middleRows(ndx, nu);
    Lxu = H1_map.middleCols(ndx, nu).topRows(ndx);
    Lpp = H1_map.middleRows(ndx + nu, np).middleCols(ndx + nu, np);
    Lpx = H1_map.middleRows(ndx + nu, np).leftCols(ndx);
    Lpu = H1_map.middleRows(ndx + nu, np).middleCols(ndx, nu);
  }

  template <template <typename Scalar> class Model>
  void set_D1_T(Model<Scalar>* const model) {
    const std::size_t ndx = model->get_state()->get_ndx();
    const std::size_t ng = model->get_ng_T();
    const std::size_t nh = model->get_nh_T();
    const std::size_t np = model->get_np();
    const std::size_t ntau = model->get_ntau();
    const std::size_t ninput = ndx + np + ntau;
    Eigen::DenseIndex it_J1 = 0;
    Lx = Eigen::Map<VectorXs>(J1_T.data() + it_J1, ndx);
    it_J1 += ndx;
    Lp = Eigen::Map<VectorXs>(J1_T.data() + it_J1, np);
    it_J1 += np + ntau;
    Gx.setZero();
    Gp.setZero();
    if (ng != 0u) {
      Eigen::Map<MatrixXs> G_map(J1_T.data() + it_J1, ninput, ng);
      Gx.topRows(ng) = G_map.topRows(ndx).transpose();
      Gp.topRows(ng) = G_map.middleRows(ndx, np).transpose();
    }
    it_J1 += ng * ninput;
    Hx.setZero();
    Hp.setZero();
    if (nh != 0u) {
      Eigen::Map<MatrixXs> H_map(J1_T.data() + it_J1, ninput, nh);
      Hx.topRows(nh) = H_map.topRows(ndx).transpose();
      Hp.topRows(nh) = H_map.middleRows(ndx, np).transpose();
    }
    it_J1 += nh * ninput;
    Eigen::Map<VectorXs> dE_map(J1_T.data() + it_J1, ninput);
    Ex = dE_map.head(ndx).transpose();
    Eu.setZero();
    Ep = dE_map.segment(ndx, np).transpose();
    Eigen::Map<MatrixXs> H1_map(H1_T.data(), ninput, ninput);
    Lxx = H1_map.topLeftCorner(ndx, ndx);
    Lpp = H1_map.middleRows(ndx, np).middleCols(ndx, np);
    Lpx = H1_map.middleRows(ndx, np).leftCols(ndx);
    Fx.setIdentity();
    Fp.setZero();
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
    Fx = F_map.middleRows(nbase, ndx).transpose();
    Fu = F_map.middleRows(nbase + ndx, nu).transpose();
    Fp = F_map.middleRows(nbase + ndx + nu, np).transpose();
    Eigen::Map<MatrixXs> G_map(J2.data() + ninput * (1 + ndx), ninput, ng);
    Gx = G_map.middleRows(nbase, ndx).transpose();
    Gu = G_map.middleRows(nbase + ndx, nu).transpose();
    Gp = G_map.middleRows(nbase + ndx + nu, np).transpose();
    Eigen::Map<MatrixXs> H_map(J2.data() + ninput * (1 + ndx + ng), ninput, nh);
    Hx = H_map.middleRows(nbase, ndx).transpose();
    Hu = H_map.middleRows(nbase + ndx, nu).transpose();
    Hp = H_map.middleRows(nbase + ndx + nu, np).transpose();
    Eigen::Map<VectorXs> dE_map(J2.data() + ninput * (1 + ndx + ng + nh),
                                ninput);
    Ex = dE_map.segment(nbase, ndx).transpose();
    Eu = dE_map.segment(nbase + ndx, nu).transpose();
    Ep = dE_map.segment(nbase + ndx + nu, np).transpose();
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
    Fp = F_map.middleRows(nbase + ndx, np).transpose();
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
    Eigen::Map<VectorXs> dE_map(J2_T.data() + ninput * (1 + ndx + ng + nh),
                                ninput);
    Ex = dE_map.segment(nbase, ndx).transpose();
    Eu.setZero();
    Ep = dE_map.segment(nbase + ndx, np).transpose();
    Eigen::Map<MatrixXs> H2_map(H2_T.data(), ninput, ninput);
    Lxx = H2_map.block(nbase, nbase, ndx, ndx);
    Lpp = H2_map.block(nbase + ndx, nbase + ndx, np, np);
    Lpx = H2_map.block(nbase + ndx, nbase, np, ndx);
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
    it_Y2 += nh * np;
    Ex = Eigen::Map<MatrixXs>(Y2.data() + it_Y2, 1, ndx);
    it_Y2 += ndx;
    Eu = Eigen::Map<MatrixXs>(Y2.data() + it_Y2, 1, nu);
    it_Y2 += nu;
    Ep = Eigen::Map<MatrixXs>(Y2.data() + it_Y2, 1, np);
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
    it_Y2 += nh * np;
    Ex = Eigen::Map<MatrixXs>(Y2_T.data() + it_Y2, 1, ndx);
    it_Y2 += ndx;
    Eu.setZero();
    Ep = Eigen::Map<MatrixXs>(Y2_T.data() + it_Y2, 1, np);
    Fx.setIdentity();
    Fp.setZero();
  }

  template <template <typename Scalar> class Model>
  void set_Yh(Model<Scalar>* const model) {
    const std::size_t ndx = model->get_state()->get_ndx();
    const std::size_t nu = model->get_nu();
    const std::size_t ng = model->get_ng();
    const std::size_t nh = model->get_nh();
    const std::size_t np = model->get_np();
    Eigen::DenseIndex it_Yh = 0;
    Fx = Eigen::Map<MatrixXs>(Yh.data() + it_Yh, ndx, ndx);
    it_Yh += ndx * ndx;
    Fu = Eigen::Map<MatrixXs>(Yh.data() + it_Yh, ndx, nu);
    it_Yh += ndx * nu;
    Fp = Eigen::Map<MatrixXs>(Yh.data() + it_Yh, ndx, np);
    it_Yh += ndx * np;
    Lx = Eigen::Map<VectorXs>(Yh.data() + it_Yh, ndx);
    it_Yh += ndx;
    Lu = Eigen::Map<VectorXs>(Yh.data() + it_Yh, nu);
    it_Yh += nu;
    Lp = Eigen::Map<VectorXs>(Yh.data() + it_Yh, np);
    it_Yh += np;
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
    it_Yh += np * nu;
    Gx = Eigen::Map<MatrixXs>(Yh.data() + it_Yh, ng, ndx);
    it_Yh += ng * ndx;
    Gu = Eigen::Map<MatrixXs>(Yh.data() + it_Yh, ng, nu);
    it_Yh += ng * nu;
    Gp = Eigen::Map<MatrixXs>(Yh.data() + it_Yh, ng, np);
    it_Yh += ng * np;
    Hx = Eigen::Map<MatrixXs>(Yh.data() + it_Yh, nh, ndx);
    it_Yh += nh * ndx;
    Hu = Eigen::Map<MatrixXs>(Yh.data() + it_Yh, nh, nu);
    it_Yh += nh * nu;
    Hp = Eigen::Map<MatrixXs>(Yh.data() + it_Yh, nh, np);
    it_Yh += nh * np;
    Ex = Eigen::Map<MatrixXs>(Yh.data() + it_Yh, 1, ndx);
    it_Yh += ndx;
    Eu = Eigen::Map<MatrixXs>(Yh.data() + it_Yh, 1, nu);
    it_Yh += nu;
    Ep = Eigen::Map<MatrixXs>(Yh.data() + it_Yh, 1, np);
  }

  template <template <typename Scalar> class Model>
  void set_Yh_hessian(Model<Scalar>* const model) {
    const std::size_t ndx = model->get_state()->get_ndx();
    const std::size_t nu = model->get_nu();
    const std::size_t np = model->get_np();
    const std::size_t hessian_size =
        ndx * ndx + ndx * nu + nu * nu + np * np + np * ndx + np * nu;
    Eigen::DenseIndex it_Yh = 0;
    if (model->get_nYh() != hessian_size) {
      it_Yh += ndx * ndx;  // Fx
      it_Yh += ndx * nu;   // Fu
      it_Yh += ndx * np;   // Fp
      it_Yh += ndx;        // Lx
      it_Yh += nu;         // Lu
      it_Yh += np;         // Lp
    }
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
  void set_Yh_T(Model<Scalar>* const model) {
    const std::size_t ndx = model->get_state()->get_ndx();
    const std::size_t ng = model->get_ng_T();
    const std::size_t nh = model->get_nh_T();
    const std::size_t np = model->get_np();
    Eigen::DenseIndex it_Yh = 0;
    Lx = Eigen::Map<VectorXs>(Yh_T.data() + it_Yh, ndx);
    it_Yh += ndx;
    Lp = Eigen::Map<VectorXs>(Yh_T.data() + it_Yh, np);
    it_Yh += np;
    Lxx = Eigen::Map<MatrixXs>(Yh_T.data() + it_Yh, ndx, ndx);
    it_Yh += ndx * ndx;
    Lpp = Eigen::Map<MatrixXs>(Yh_T.data() + it_Yh, np, np);
    it_Yh += np * np;
    Lpx = Eigen::Map<MatrixXs>(Yh_T.data() + it_Yh, np, ndx);
    it_Yh += np * ndx;
    Gx.setZero();
    if (ng != 0u) {
      Gx.topRows(ng) = Eigen::Map<MatrixXs>(Yh_T.data() + it_Yh, ng, ndx);
    }
    it_Yh += ng * ndx;
    Gp.setZero();
    if (ng != 0u) {
      Gp.topRows(ng) = Eigen::Map<MatrixXs>(Yh_T.data() + it_Yh, ng, np);
    }
    it_Yh += ng * np;
    Hx.setZero();
    if (nh != 0u) {
      Hx.topRows(nh) = Eigen::Map<MatrixXs>(Yh_T.data() + it_Yh, nh, ndx);
    }
    it_Yh += nh * ndx;
    Hp.setZero();
    if (nh != 0u) {
      Hp.topRows(nh) = Eigen::Map<MatrixXs>(Yh_T.data() + it_Yh, nh, np);
    }
    it_Yh += nh * np;
    Ex = Eigen::Map<MatrixXs>(Yh_T.data() + it_Yh, 1, ndx);
    it_Yh += ndx;
    Eu.setZero();
    Ep = Eigen::Map<MatrixXs>(Yh_T.data() + it_Yh, 1, np);
    Fx.setIdentity();
    Fp.setZero();
  }

  template <template <typename Scalar> class Model>
  void set_Yh_hessian_T(Model<Scalar>* const model) {
    const std::size_t ndx = model->get_state()->get_ndx();
    const std::size_t np = model->get_np();
    const std::size_t hessian_size = ndx * ndx + np * np + np * ndx;
    Eigen::DenseIndex it_Yh = 0;
    if (model->get_nYh_T() != hessian_size) {
      it_Yh += ndx;  // Lx
      it_Yh += np;   // Lp
    }
    Lxx = Eigen::Map<MatrixXs>(Yh_T.data() + it_Yh, ndx, ndx);
    it_Yh += ndx * ndx;
    Lpp = Eigen::Map<MatrixXs>(Yh_T.data() + it_Yh, np, np);
    it_Yh += np * np;
    Lpx = Eigen::Map<MatrixXs>(Yh_T.data() + it_Yh, np, ndx);
  }
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/codegen/observer.hxx"

#endif  // CROCODDYL_WITH_CODEGEN

#endif  // CROCODDYL_CORE_CODEGEN_OBSERVER_HPP_
