///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_OBSERVER_RK_HPP_
#define CROCODDYL_CORE_OBSERVER_RK_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/integ-observer-base.hpp"
#include "crocoddyl/core/integrator/rk.hpp"
#include "crocoddyl/core/utils/conversions.hpp"

namespace crocoddyl {

/**
 * @brief Runge-Kutta integrated observer model
 *
 * This is the observer counterpart of the Python PDDP RK observer. It
 * discretizes continuous-time constrained dynamics for estimation with
 * observer controls \f$w=[\eta,u]\f$, where \f$\eta\f$ is process noise in the
 * state tangent space and \f$u\f$ is the dynamics control.
 */
template <typename _Scalar>
class IntegratedObserverModelRKTpl
    : public IntegratedObserverModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ActionModelBase, IntegratedObserverModelRKTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef IntegratedObserverModelAbstractTpl<Scalar> Base;
  typedef IntegratedObserverDataRKTpl<Scalar> Data;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef DynamicsModelAbstractTpl<Scalar> DynamicsModelAbstract;
  typedef DynamicsDataAbstractTpl<Scalar> DynamicsDataAbstract;
  typedef CostModelSumTpl<Scalar> CostModelSum;
  typedef CostDataSumTpl<Scalar> CostDataSum;
  typedef ConstraintModelManagerTpl<Scalar> ConstraintModelManager;
  typedef ConstraintDataManagerTpl<Scalar> ConstraintDataManager;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef typename Base::ParameterManager ParameterManager;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  IntegratedObserverModelRKTpl(
      std::shared_ptr<DynamicsModelAbstract> dynamics,
      std::shared_ptr<CostModelSum> costs,
      std::shared_ptr<ConstraintModelManager> constraints = nullptr,
      const Scalar time_step = Scalar(1e-3), const RKType rktype = four);
  virtual ~IntegratedObserverModelRKTpl() = default;

  using Base::createData;

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

  virtual void set_params(const std::shared_ptr<ActionDataAbstract>& data,
                          std::shared_ptr<ParameterManager> params) override;

  virtual void update_p(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& p) override;

  virtual std::shared_ptr<ActionDataAbstract> createData() override;
  virtual std::shared_ptr<ActionDataAbstract> createData(
      const std::shared_ptr<ParameterDataManager>& params_data) override;

  virtual bool checkData(
      const std::shared_ptr<ActionDataAbstract>& data) override;

  virtual void quasiStatic(const std::shared_ptr<ActionDataAbstract>& data,
                           Eigen::Ref<VectorXs> u,
                           const Eigen::Ref<const VectorXs>& x,
                           const std::size_t maxiter = 100,
                           const Scalar tol = Scalar(1e-9)) override;

  std::size_t get_ni() const;

  RKType get_rk_type() const;

  virtual void print(std::ostream& os) const override;

  template <typename NewScalar>
  IntegratedObserverModelRKTpl<NewScalar> cast() const;

 protected:
  using Base::constraints_;
  using Base::costs_;
  using Base::dynamics_;
  using Base::np_;
  using Base::nr_;
  using Base::params_;
  using Base::state_;
  using Base::time_step_;
  using Base::u_zero_;

  std::shared_ptr<Data> cast_data(
      const std::shared_ptr<ActionDataAbstract>& data);

 private:
  void set_rk_type(const RKType rktype);

  RKType rk_type_;
  std::vector<Scalar> rk_c_;
  std::size_t ni_;
};

template <typename _Scalar>
struct IntegratedObserverDataRKTpl
    : public IntegratedObserverDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef IntegratedObserverDataAbstractTpl<Scalar> Base;
  typedef DynamicsDataAbstractTpl<Scalar> DynamicsDataAbstract;
  typedef CostDataSumTpl<Scalar> CostDataSum;
  typedef ConstraintDataManagerTpl<Scalar> ConstraintDataManager;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename S> class Model>
  explicit IntegratedObserverDataRKTpl(
      Model<Scalar>* const model,
      const std::shared_ptr<ParameterDataManager>& params_data =
          std::shared_ptr<ParameterDataManager>());
  virtual ~IntegratedObserverDataRKTpl() = default;

  template <class Model>
  void resize(Model* const model, const bool running_node = true) {
    Base::resize(model, running_node);
    const std::size_t ni = model->get_ni();
    const std::size_t nx = model->get_state()->get_nx();
    const std::size_t ndx = model->get_state()->get_ndx();
    const std::size_t nv = model->get_state()->get_nv();
    const std::size_t nu = model->get_nu();
    const std::size_t np = model->get_np();

    resize_vector(integral, ni, Scalar(0.));
    resize_vector(ki, ni, ndx);
    resize_vector(y, ni, nx);
    resize_vector(dx_rk, ni, ndx);
    resize_matrix_vector(dki_dx, ni, ndx, ndx);
    resize_matrix_vector(dki_du, ni, ndx, nu);
    resize_matrix_vector(dki_dp, ni, ndx, np);
    resize_matrix_vector(dy_dx, ni, ndx, ndx);
    resize_matrix_vector(dy_du, ni, ndx, nu);
    resize_matrix_vector(dy_dp, ni, ndx, np);
    resize_vector(dli_dx, ni, ndx);
    resize_vector(dli_du, ni, nu);
    resize_vector(dli_dp, ni, np);
    resize_matrix_vector(ddli_ddx, ni, ndx, ndx);
    resize_matrix_vector(ddli_ddu, ni, nu, nu);
    resize_matrix_vector(ddli_ddp, ni, np, np);
    resize_matrix_vector(ddli_dxdu, ni, ndx, nu);
    resize_matrix_vector(ddli_dpdx, ni, np, ndx);
    resize_matrix_vector(ddli_dpdu, ni, np, nu);
    resize_matrix_vector(Luu_partialx, ni, nu, nu);
    resize_matrix_vector(Lpp_partialx, ni, np, np);

    ddx_dx.resize(ndx, ndx);
    ddx_du.resize(ndx, nu);
    ddx_dp.resize(ndx, np);
    Jfirst.resize(ndx, ndx);
    Jsecond.resize(ndx, ndx);
    tmp_ndx_ndx.resize(ndx, ndx);
    tmp_ndx_nu.resize(ndx, nu);
    tmp_ndx_np.resize(ndx, np);
    tmp_nu_nu.resize(nu, nu);
    tmp_np_np.resize(np, np);
    setZero();
    if (ni != 0u) {
      dy_dx[0].diagonal().setOnes();
      dki_dx[0].topRightCorner(nv, nv).diagonal().setOnes();
    }
  }

  virtual void setZero() override {
    Base::setZero();
    for (std::size_t i = 0; i < integral.size(); ++i) {
      integral[i] = Scalar(0.);
      ki[i].setZero();
      y[i].setZero();
      dx_rk[i].setZero();
      dki_dx[i].setZero();
      dki_du[i].setZero();
      dki_dp[i].setZero();
      dy_dx[i].setZero();
      dy_du[i].setZero();
      dy_dp[i].setZero();
      dli_dx[i].setZero();
      dli_du[i].setZero();
      dli_dp[i].setZero();
      ddli_ddx[i].setZero();
      ddli_ddu[i].setZero();
      ddli_ddp[i].setZero();
      ddli_dxdu[i].setZero();
      ddli_dpdx[i].setZero();
      ddli_dpdu[i].setZero();
      Luu_partialx[i].setZero();
      Lpp_partialx[i].setZero();
    }
    ddx_dx.setZero();
    ddx_du.setZero();
    ddx_dp.setZero();
    Jfirst.setZero();
    Jsecond.setZero();
    tmp_ndx_ndx.setZero();
    tmp_ndx_nu.setZero();
    tmp_ndx_np.setZero();
    tmp_nu_nu.setZero();
    tmp_np_np.setZero();
  }

  std::vector<std::shared_ptr<DynamicsDataAbstract> > dynamics_stage;
  std::vector<std::shared_ptr<CostDataSum> > costs_stage;
  std::vector<Scalar> integral;
  std::vector<VectorXs> ki;
  std::vector<VectorXs> y;
  std::vector<VectorXs> dx_rk;
  std::vector<MatrixXs> dki_dx;
  std::vector<MatrixXs> dki_du;
  std::vector<MatrixXs> dki_dp;
  std::vector<MatrixXs> dy_dx;
  std::vector<MatrixXs> dy_du;
  std::vector<MatrixXs> dy_dp;
  std::vector<VectorXs> dli_dx;
  std::vector<VectorXs> dli_du;
  std::vector<VectorXs> dli_dp;
  std::vector<MatrixXs> ddli_ddx;
  std::vector<MatrixXs> ddli_ddu;
  std::vector<MatrixXs> ddli_ddp;
  std::vector<MatrixXs> ddli_dxdu;
  std::vector<MatrixXs> ddli_dpdx;
  std::vector<MatrixXs> ddli_dpdu;
  std::vector<MatrixXs> Luu_partialx;
  std::vector<MatrixXs> Lpp_partialx;
  MatrixXs ddx_dx;
  MatrixXs ddx_du;
  MatrixXs ddx_dp;
  MatrixXs Jfirst;
  MatrixXs Jsecond;
  MatrixXs tmp_ndx_ndx;
  MatrixXs tmp_ndx_nu;
  MatrixXs tmp_ndx_np;
  MatrixXs tmp_nu_nu;
  MatrixXs tmp_np_np;

  using Base::constraints;
  using Base::cost;
  using Base::costs;
  using Base::dE_dp;
  using Base::dE_dv;
  using Base::dissipative_E;
  using Base::dx;
  using Base::dynamics;
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

 private:
  static void resize_vector(std::vector<Scalar>& values, const std::size_t size,
                            const Scalar value) {
    values.resize(size, value);
  }

  static void resize_vector(std::vector<VectorXs>& values,
                            const std::size_t size, const std::size_t rows) {
    values.resize(size);
    for (std::size_t i = 0; i < size; ++i) {
      values[i].resize(rows);
    }
  }

  static void resize_matrix_vector(std::vector<MatrixXs>& values,
                                   const std::size_t size,
                                   const std::size_t rows,
                                   const std::size_t cols) {
    values.resize(size);
    for (std::size_t i = 0; i < size; ++i) {
      values[i].resize(rows, cols);
    }
  }
};

}  // namespace crocoddyl

#include "crocoddyl/core/observer/rk.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::IntegratedObserverModelRKTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::IntegratedObserverDataRKTpl)

#endif  // CROCODDYL_CORE_OBSERVER_RK_HPP_
