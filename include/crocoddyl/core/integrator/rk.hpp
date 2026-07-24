///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, University of Edinburgh, University of Trento,
//                          LAAS-CNRS, IRI: CSIC-UPC, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_INTEGRATOR_RK_HPP_
#define CROCODDYL_CORE_INTEGRATOR_RK_HPP_

#include "crocoddyl/core/constraints/constraint-manager.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/integ-action-base.hpp"
#include "crocoddyl/core/params/integrator-timeopt.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"

namespace crocoddyl {

enum RKType { two = 2, three = 3, four = 4 };

/**
 * @brief Standard RK integrator
 *
 * It applies RK2, RK3 or RK4 to either a differential action model or the
 * compositional continuous dynamics, cost and constraint backend. The latter
 * propagates action and dynamics parameters, including shared-time
 * sensitivities, through all RK stages without model-global numerical caches.
 *
 * This standard RK scheme introduces also the possibility to parametrize the
 * control trajectory inside an integration step, for instance using
 * polynomials. This requires introducing some notation to clarify the
 * difference between the control inputs of the differential model and the
 * control inputs to the integrated model. We have decided to use
 * \f$\mathbf{w}\f$ to refer to the control inputs of the differential model and
 * \f$\mathbf{u}\f$ for the control inputs of the integrated action model.
 *
 * \sa `IntegratedActionModelAbstractTpl`, `calc()`, `calcDiff()`,
 * `createData()`
 */
template <typename _Scalar>
class IntegratedActionModelRKTpl
    : public IntegratedActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ActionModelBase, IntegratedActionModelRKTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef IntegratedActionModelAbstractTpl<Scalar> Base;
  typedef IntegratedActionDataRKTpl<Scalar> Data;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef DifferentialActionModelAbstractTpl<Scalar>
      DifferentialActionModelAbstract;
  typedef DynamicsModelAbstractTpl<Scalar> DynamicsModelAbstract;
  typedef DifferentialActionDataAbstractTpl<Scalar>
      DifferentialActionDataAbstract;
  typedef DynamicsDataAbstractTpl<Scalar> DynamicsDataAbstract;
  typedef CostModelSumTpl<Scalar> CostModelSum;
  typedef CostDataSumTpl<Scalar> CostDataSum;
  typedef ConstraintModelManagerTpl<Scalar> ConstraintModelManager;
  typedef ConstraintDataManagerTpl<Scalar> ConstraintDataManager;
  typedef ControlParametrizationModelAbstractTpl<Scalar>
      ControlParametrizationModelAbstract;
  typedef ControlParametrizationDataAbstractTpl<Scalar>
      ControlParametrizationDataAbstract;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef typename Base::ParameterManager ParameterManager;
  typedef IntegratorTimeTpl<Scalar> IntegratorTime;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the RK integrator
   *
   * @param[in] model      Differential action model
   * @param[in] control    Control parametrization
   * @param[in] rktype     Type of RK integrator
   * @param[in] time_step  Step time (default 1e-3)
   * @param[in] with_cost_residual  Compute cost residual (default true)
   */
  IntegratedActionModelRKTpl(
      std::shared_ptr<DifferentialActionModelAbstract> model,
      std::shared_ptr<ControlParametrizationModelAbstract> control,
      const RKType rktype, const Scalar time_step = Scalar(1e-3),
      const bool with_cost_residual = true);

  /**
   * @brief Initialize the RK integrator
   *
   * This initialization uses `ControlParametrizationPolyZeroTpl` for the
   * control parametrization.
   *
   * @param[in] model      Differential action model
   * @param[in] rktype     Type of RK integrator
   * @param[in] time_step  Step time (default 1e-3)
   * @param[in] with_cost_residual  Compute cost residual (default true)
   */
  IntegratedActionModelRKTpl(
      std::shared_ptr<DifferentialActionModelAbstract> model,
      const RKType rktype, const Scalar time_step = Scalar(1e-3),
      const bool with_cost_residual = true);

  /**
   * @brief Initialize from continuous dynamics, costs and constraints
   *
   * Null control and time arguments select a zero-order control
   * parametrization and a private default integration time, respectively.
   */
  IntegratedActionModelRKTpl(
      std::shared_ptr<DynamicsModelAbstract> dynamics,
      std::shared_ptr<CostModelSum> costs,
      std::shared_ptr<ConstraintModelManager> constraints = nullptr,
      std::shared_ptr<ControlParametrizationModelAbstract> control = nullptr,
      std::shared_ptr<IntegratorTime> integrator_time = nullptr,
      const RKType rktype = four);

  /**
   * @brief Copy constructor
   */
  IntegratedActionModelRKTpl(const IntegratedActionModelRKTpl<Scalar>& other)
      : Base(other), rk_type_(other.rk_type_) {
    set_rk_type(rk_type_);
  }

  virtual ~IntegratedActionModelRKTpl() = default;

  using Base::createData;

  /**
   * @brief Integrate the differential action model using RK scheme
   *
   * @param[in] data  RK integrator data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Integrate the total cost value for nodes that depends only on the
   * state using RK scheme
   *
   * It computes the total cost and defines the next state as the current one.
   * This function is used in the terminal nodes of an optimal control problem.
   *
   * @param[in] data  RK integrator data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Compute the partial derivatives of the RK integrator
   *
   * @param[in] data  RK integrator data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Compute the partial derivatives of the cost
   *
   * It updates the derivatives of the cost function with respect to the state
   * only. This function is used in the terminal nodes of an optimal control
   * problem.
   *
   * @param[in] data  RK integrator data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Create the RK integrator data
   *
   * @return the RK integrator data
   */
  virtual std::shared_ptr<ActionDataAbstract> createData() override;
  virtual std::shared_ptr<ActionDataAbstract> createData(
      const std::shared_ptr<ParameterDataManager>& params_data) override;

  /**
   * @brief Cast the RK integrated-action model to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return IntegratedActionModelRKTpl<NewScalar> An action model with the
   * new scalar type.
   */
  template <typename NewScalar>
  IntegratedActionModelRKTpl<NewScalar> cast() const;

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
   * @param[in]  data     RK integrator data
   * @param[out] u        Quasic static commands
   * @param[in]  x        State point (velocity has to be zero)
   * @param[in]  maxiter  Maximum allowed number of iterations
   * @param[in]  tol      Tolerance
   */
  virtual void quasiStatic(const std::shared_ptr<ActionDataAbstract>& data,
                           Eigen::Ref<VectorXs> u,
                           const Eigen::Ref<const VectorXs>& x,
                           const std::size_t maxiter = 100,
                           const Scalar tol = Scalar(1e-9)) override;

  /** @brief Attach a parameter manager to every RK stage */
  void set_params(const std::shared_ptr<ActionDataAbstract>& data,
                  std::shared_ptr<ParameterManager> params) override;

  /** @brief Update active parameters in every RK stage */
  virtual void update_p(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& p) override;

  /**
   * @brief Return the number of nodes of the integrator
   */
  std::size_t get_ni() const;

  /**
   * @brief Print relevant information of the RK integrator model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override;

 protected:
  using Base::constraints_;      //!< Constraint manager
  using Base::control_;          //!< Control parametrization
  using Base::costs_;            //!< Cost model stack
  using Base::differential_;     //!< Differential action model
  using Base::dynamics_;         //!< Dynamics model
  using Base::integrator_time_;  //!< Integrator time description
  using Base::ng_;               //!< Number of inequality constraints
  using Base::nh_;               //!< Number of equality constraints
  using Base::nu_;               //!< Dimension of the control
  using Base::params_;
  using Base::refresh_integrator_time;
  using Base::state_;       //!< Model of the state
  using Base::time_step2_;  //!< Square of the time step used for integration
  using Base::time_step_;   //!< Time step used for integration
  using Base::with_cost_residual_;  //!< Flag indicating whether a cost residual
                                    //!< is used

 private:
  /**
   * @brief Modify the RK type
   */
  void set_rk_type(const RKType rktype);

  RKType rk_type_;
  std::vector<Scalar> rk_c_;
  std::size_t ni_;
};

/**
 * @brief Data and per-stage workspaces for RK2, RK3 and RK4
 *
 * Each instance owns its stage vectors, Jacobians, Hessian workspaces and
 * time-parameter column bookkeeping. Backend model data and the optional
 * parameter payload are shared. `resize()` preserves allocation capacity for
 * repeated calculations with the same layout.
 */
template <typename _Scalar>
struct IntegratedActionDataRKTpl
    : public IntegratedActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef IntegratedActionDataAbstractTpl<Scalar> Base;
  typedef DifferentialActionDataAbstractTpl<Scalar>
      DifferentialActionDataAbstract;
  typedef DynamicsDataAbstractTpl<Scalar> DynamicsDataAbstract;
  typedef CostDataSumTpl<Scalar> CostDataSum;
  typedef ConstraintDataManagerTpl<Scalar> ConstraintDataManager;
  typedef ControlParametrizationDataAbstractTpl<Scalar>
      ControlParametrizationDataAbstract;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit IntegratedActionDataRKTpl(
      Model<Scalar>* const model,
      const std::shared_ptr<ParameterDataManager>& params_data =
          std::shared_ptr<ParameterDataManager>())
      : Base(model),
        integral(model->get_ni(), Scalar(0.)),
        dx(model->get_state()->get_ndx()),
        ki(model->get_ni(), VectorXs::Zero(model->get_state()->get_ndx())),
        y(model->get_ni(), VectorXs::Zero(model->get_state()->get_nx())),
        ws(model->get_ni(), VectorXs::Zero(model->get_control()->get_nw())),
        dx_rk(model->get_ni(), VectorXs::Zero(model->get_state()->get_ndx())),
        dki_dx(model->get_ni(), MatrixXs::Zero(model->get_state()->get_ndx(),
                                               model->get_state()->get_ndx())),
        dki_du(model->get_ni(),
               MatrixXs::Zero(model->get_state()->get_ndx(), model->get_nu())),
        dki_dp(model->get_ni(),
               MatrixXs::Zero(model->get_state()->get_ndx(), model->get_np())),
        dyi_dx(model->get_ni(), MatrixXs::Zero(model->get_state()->get_ndx(),
                                               model->get_state()->get_ndx())),
        dyi_du(model->get_ni(),
               MatrixXs::Zero(model->get_state()->get_ndx(), model->get_nu())),
        dyi_dp(model->get_ni(),
               MatrixXs::Zero(model->get_state()->get_ndx(), model->get_np())),
        dli_dx(model->get_ni(), VectorXs::Zero(model->get_state()->get_ndx())),
        dli_du(model->get_ni(), VectorXs::Zero(model->get_nu())),
        dli_dp(model->get_ni(), VectorXs::Zero(model->get_np())),
        ddli_ddx(model->get_ni(),
                 MatrixXs::Zero(model->get_state()->get_ndx(),
                                model->get_state()->get_ndx())),
        ddli_ddw(model->get_ni(),
                 MatrixXs::Zero(model->get_control()->get_nw(),
                                model->get_control()->get_nw())),
        ddli_ddu(model->get_ni(),
                 MatrixXs::Zero(model->get_nu(), model->get_nu())),
        ddli_ddp(model->get_ni(),
                 MatrixXs::Zero(model->get_np(), model->get_np())),
        ddli_dxdw(model->get_ni(),
                  MatrixXs::Zero(model->get_state()->get_ndx(),
                                 model->get_control()->get_nw())),
        ddli_dxdu(model->get_ni(), MatrixXs::Zero(model->get_state()->get_ndx(),
                                                  model->get_nu())),
        ddli_dpdx(
            model->get_ni(),
            MatrixXs::Zero(model->get_np(), model->get_state()->get_ndx())),
        ddli_dpdu(model->get_ni(),
                  MatrixXs::Zero(model->get_np(), model->get_nu())),
        ddli_dwdu(
            model->get_ni(),
            MatrixXs::Zero(model->get_control()->get_nw(), model->get_nu())),
        Luu_partialx(model->get_ni(),
                     MatrixXs::Zero(model->get_nu(), model->get_nu())),
        Lpp_partialx(model->get_ni(),
                     MatrixXs::Zero(model->get_np(), model->get_np())),
        Lxu_i(model->get_ni(),
              MatrixXs::Zero(model->get_state()->get_ndx(), model->get_nu())),
        Lxx_partialx(model->get_ni(),
                     MatrixXs::Zero(model->get_state()->get_ndx(),
                                    model->get_state()->get_ndx())),
        Lxx_partialu(
            model->get_ni(),
            MatrixXs::Zero(model->get_state()->get_ndx(), model->get_nu())),
        ddx_dp(MatrixXs::Zero(model->get_state()->get_ndx(), model->get_np())),
        tmp_ndx_np(
            MatrixXs::Zero(model->get_state()->get_ndx(), model->get_np())) {
    dx.setZero();

    differential.reserve(model->get_ni());
    dynamics.reserve(model->get_ni());
    costs.reserve(model->get_ni());
    control.reserve(model->get_ni());
    constraints.reset();
    params = params_data;

    for (std::size_t i = 0; i < model->get_ni(); ++i) {
      control.push_back(std::shared_ptr<ControlParametrizationDataAbstract>(
          model->get_control()->createData()));
    }

    if (model->get_dynamics() != nullptr) {
      for (std::size_t i = 0; i < model->get_ni(); ++i) {
        dynamics.push_back(std::shared_ptr<DynamicsDataAbstract>(
            params_data != nullptr
                ? model->get_dynamics()->createData(params_data)
                : model->get_dynamics()->createData()));
        costs.push_back(std::shared_ptr<CostDataSum>(
            model->get_costs()->createData(dynamics[i]->shared)));
      }
      if (model->get_constraints() != nullptr) {
        constraints = model->get_constraints()->createData(dynamics[0]->shared);
      }
    } else {
      for (std::size_t i = 0; i < model->get_ni(); ++i) {
        differential.push_back(std::shared_ptr<DifferentialActionDataAbstract>(
            model->get_differential()->createData()));
      }
    }

    const std::size_t nv = model->get_state()->get_nv();
    dyi_dx[0].diagonal().setOnes();
    dki_dx[0].topRightCorner(nv, nv).diagonal().setOnes();
  }
  virtual ~IntegratedActionDataRKTpl() = default;

  template <class Model>
  void resize(Model* const model, const bool running_node = true) {
    const std::size_t ni = model->get_ni();
    const std::size_t nx = model->get_state()->get_nx();
    const std::size_t ndx = model->get_state()->get_ndx();
    const std::size_t nv = model->get_state()->get_nv();
    const std::size_t nu = model->get_nu();
    const std::size_t nw = model->get_control()->get_nw();

    ActionDataAbstractTpl<Scalar>::resize(model, running_node);
    integral.resize(ni, Scalar(0.));
    dx.resize(ndx);
    ki.resize(ni, VectorXs::Zero(ndx));
    y.resize(ni, VectorXs::Zero(nx));
    ws.resize(ni, VectorXs::Zero(nw));
    dx_rk.resize(ni, VectorXs::Zero(ndx));
    dki_dx.resize(ni, MatrixXs::Zero(ndx, ndx));
    dki_du.resize(ni, MatrixXs::Zero(ndx, nu));
    dki_dp.resize(ni, MatrixXs::Zero(ndx, model->get_np()));
    dyi_dx.resize(ni, MatrixXs::Zero(ndx, ndx));
    dyi_du.resize(ni, MatrixXs::Zero(ndx, nu));
    dyi_dp.resize(ni, MatrixXs::Zero(ndx, model->get_np()));
    dli_dx.resize(ni, VectorXs::Zero(ndx));
    dli_du.resize(ni, VectorXs::Zero(nu));
    dli_dp.resize(ni, VectorXs::Zero(model->get_np()));
    ddli_ddx.resize(ni, MatrixXs::Zero(ndx, ndx));
    ddli_ddw.resize(ni, MatrixXs::Zero(nw, nw));
    ddli_ddu.resize(ni, MatrixXs::Zero(nu, nu));
    ddli_ddp.resize(ni, MatrixXs::Zero(model->get_np(), model->get_np()));
    ddli_dxdw.resize(ni, MatrixXs::Zero(ndx, nw));
    ddli_dxdu.resize(ni, MatrixXs::Zero(ndx, nu));
    ddli_dpdx.resize(ni, MatrixXs::Zero(model->get_np(), ndx));
    ddli_dpdu.resize(ni, MatrixXs::Zero(model->get_np(), nu));
    ddli_dwdu.resize(ni, MatrixXs::Zero(nw, nu));
    Luu_partialx.resize(ni, MatrixXs::Zero(nu, nu));
    Lpp_partialx.resize(ni, MatrixXs::Zero(model->get_np(), model->get_np()));
    Lxu_i.resize(ni, MatrixXs::Zero(ndx, nu));
    Lxx_partialx.resize(ni, MatrixXs::Zero(ndx, ndx));
    Lxx_partialu.resize(ni, MatrixXs::Zero(ndx, nu));
    ddx_dp.resize(ndx, model->get_np());
    tmp_ndx_np.resize(ndx, model->get_np());
    dx.setZero();
    for (std::size_t i = 0; i < ni; ++i) {
      ki[i].resize(ndx);
      y[i].resize(nx);
      ws[i].resize(nw);
      dx_rk[i].resize(ndx);
      dki_dx[i].resize(ndx, ndx);
      dki_du[i].resize(ndx, nu);
      dki_dp[i].resize(ndx, model->get_np());
      dyi_dx[i].resize(ndx, ndx);
      dyi_du[i].resize(ndx, nu);
      dyi_dp[i].resize(ndx, model->get_np());
      dli_dx[i].resize(ndx);
      dli_du[i].resize(nu);
      dli_dp[i].resize(model->get_np());
      ddli_ddx[i].resize(ndx, ndx);
      ddli_ddw[i].resize(nw, nw);
      ddli_ddu[i].resize(nu, nu);
      ddli_ddp[i].resize(model->get_np(), model->get_np());
      ddli_dxdw[i].resize(ndx, nw);
      ddli_dxdu[i].resize(ndx, nu);
      ddli_dpdx[i].resize(model->get_np(), ndx);
      ddli_dpdu[i].resize(model->get_np(), nu);
      ddli_dwdu[i].resize(nw, nu);
      Luu_partialx[i].resize(nu, nu);
      Lpp_partialx[i].resize(model->get_np(), model->get_np());
      Lxu_i[i].resize(ndx, nu);
      Lxx_partialx[i].resize(ndx, ndx);
      Lxx_partialu[i].resize(ndx, nu);
      ki[i].setZero();
      y[i].setZero();
      ws[i].setZero();
      dx_rk[i].setZero();
      dki_dx[i].setZero();
      dki_du[i].setZero();
      dki_dp[i].setZero();
      dyi_dx[i].setZero();
      dyi_du[i].setZero();
      dyi_dp[i].setZero();
      dli_dx[i].setZero();
      dli_du[i].setZero();
      dli_dp[i].setZero();
      ddli_ddx[i].setZero();
      ddli_ddw[i].setZero();
      ddli_ddu[i].setZero();
      ddli_ddp[i].setZero();
      ddli_dxdw[i].setZero();
      ddli_dxdu[i].setZero();
      ddli_dpdx[i].setZero();
      ddli_dpdu[i].setZero();
      ddli_dwdu[i].setZero();
      Luu_partialx[i].setZero();
      Lpp_partialx[i].setZero();
      Lxu_i[i].setZero();
      Lxx_partialx[i].setZero();
      Lxx_partialu[i].setZero();
    }
    ddx_dp.setZero();
    tmp_ndx_np.setZero();
    if (ni != 0) {
      dyi_dx[0].setZero();
      dyi_dx[0].diagonal().setOnes();
      dki_dx[0].setZero();
      dki_dx[0].topRightCorner(nv, nv).diagonal().setOnes();
    }
  }

  std::vector<std::shared_ptr<DifferentialActionDataAbstract> >
      differential;  //!< List of differential model data
  std::vector<std::shared_ptr<DynamicsDataAbstract> >
      dynamics;  //!< List of dynamics model data
  std::vector<std::shared_ptr<CostDataSum> >
      costs;  //!< List of cost-model data
  std::shared_ptr<ConstraintDataManager>
      constraints;                               //!< Constraint-manager data
  std::shared_ptr<ParameterDataManager> params;  //!< Shared parameter payload
  std::vector<std::size_t>
      timeopt_param_cols;  //!< Active time-parameter column indices
  std::vector<std::shared_ptr<ControlParametrizationDataAbstract> >
      control;  //!< List of control parametrization data
  std::vector<Scalar> integral;
  VectorXs dx;               //!< State rate
  std::vector<VectorXs> ki;  //!< List of RK terms related to system dynamics
  std::vector<VectorXs>
      y;  //!< List of states where f is evaluated in the RK integration
  std::vector<VectorXs> ws;  //!< Control inputs evaluated in the RK integration
  std::vector<VectorXs> dx_rk;

  std::vector<MatrixXs>
      dki_dx;  //!< List of partial derivatives of RK nodes with respect to the
               //!< state of the RK integration. dki/dx
  std::vector<MatrixXs>
      dki_du;  //!< List of partial derivatives of RK nodes with respect to the
               //!< control parameters of the RK integration. dki/du
  std::vector<MatrixXs>
      dki_dp;  //!< List of partial derivatives of RK nodes with respect to the
               //!< model parameters. dki/dp

  std::vector<MatrixXs>
      dyi_dx;  //!< List of partial derivatives of RK dynamics with respect to
               //!< the state of the RK integrator. dyi/dx
  std::vector<MatrixXs>
      dyi_du;  //!< List of partial derivatives of RK dynamics with respect to
               //!< the control parameters of the RK integrator. dyi/du
  std::vector<MatrixXs>
      dyi_dp;  //!< List of partial derivatives of RK dynamics with respect to
               //!< the model parameters. dyi/dp

  std::vector<VectorXs>
      dli_dx;  //!< List of partial derivatives of the cost with respect to the
               //!< state of the RK integration. dli_dx
  std::vector<VectorXs>
      dli_du;  //!< List of partial derivatives of the cost with respect to the
               //!< control input of the RK integration. dli_du
  std::vector<VectorXs>
      dli_dp;  //!< List of partial derivatives of the cost with respect to the
               //!< model parameters. dli_dp

  std::vector<MatrixXs>
      ddli_ddx;  //!< List of second partial derivatives of the cost with
                 //!< respect to the state of the RK integration. ddli_ddx
  std::vector<MatrixXs>
      ddli_ddw;  //!< List of second partial derivatives of the cost with
                 //!< respect to the control parameters of the RK integration.
                 //!< ddli_ddw
  std::vector<MatrixXs> ddli_ddu;  //!< List of second partial derivatives of
                                   //!< the cost with respect to the control
                                   //!< input of the RK integration. ddli_ddu
  std::vector<MatrixXs> ddli_ddp;  //!< List of second partial derivatives of
                                   //!< the cost with respect to parameters.
  std::vector<MatrixXs>
      ddli_dxdw;  //!< List of second partial derivatives of the cost with
                  //!< respect to the state and control input of the RK
                  //!< integration. ddli_dxdw
  std::vector<MatrixXs>
      ddli_dxdu;  //!< List of second partial derivatives of the cost with
                  //!< respect to the state and control parameters of the RK
                  //!< integration. ddli_dxdu
  std::vector<MatrixXs>
      ddli_dpdx;  //!< List of second partial derivatives of the cost with
                  //!< respect to parameters and state. ddli_dpdx
  std::vector<MatrixXs>
      ddli_dpdu;  //!< List of second partial derivatives of the cost with
                  //!< respect to parameters and control. ddli_dpdu
  std::vector<MatrixXs>
      ddli_dwdu;  //!< List of second partial derivatives of the cost with
                  //!< respect to the control parameters and inputs control of
                  //!< the RK integration. ddli_dwdu

  std::vector<MatrixXs> Luu_partialx;
  std::vector<MatrixXs> Lpp_partialx;
  std::vector<MatrixXs> Lxu_i;
  std::vector<MatrixXs> Lxx_partialx;
  std::vector<MatrixXs> Lxx_partialu;
  MatrixXs ddx_dp;
  MatrixXs tmp_ndx_np;

  using Base::cost;
  using Base::Fp;
  using Base::Fu;
  using Base::Fx;
  using Base::Gp;
  using Base::Hp;
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
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/integrator/rk.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::IntegratedActionModelRKTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::IntegratedActionDataRKTpl)

#endif  // CROCODDYL_CORE_INTEGRATOR_RK4_HPP_
