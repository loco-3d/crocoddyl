///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, LAAS-CNRS, University of Edinburgh,
//                          University of Oxford, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_INTEGRATOR_EULER_HPP_
#define CROCODDYL_CORE_INTEGRATOR_EULER_HPP_

#include "crocoddyl/core/constraints/constraint-manager.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/integ-action-base.hpp"
#include "crocoddyl/core/params/integrator-timeopt.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"

namespace crocoddyl {

/**
 * @brief Symplectic Euler integrator
 *
 * It applies a symplectic Euler integration scheme to either a differential
 * action model or the compositional continuous dynamics, cost and constraint
 * backend. The latter propagates action and dynamics parameters, including a
 * shared time parameter, through every first- and second-order action block.
 *
 * This symplectic Euler scheme introduces also the possibility to parametrize
 * the control trajectory inside an integration step, for instance using
 * polynomials. This requires introducing some notation to clarify the
 * difference between the control inputs of the differential model and the
 * control inputs to the integrated model. We have decided to use
 * \f$\mathbf{w}\f$ to refer to the control inputs of the differential model and
 * \f$\mathbf{u}\f$ for the control inputs of the integrated action model. Note
 * that the zero-order (e.g., `ControlParametrizationModelPolyZeroTpl`) are the
 * only ones that make sense to use within this integrator.
 *
 * \sa `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class IntegratedActionModelEulerTpl
    : public IntegratedActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ActionModelBase, IntegratedActionModelEulerTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef IntegratedActionModelAbstractTpl<Scalar> Base;
  typedef IntegratedActionDataEulerTpl<Scalar> Data;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef DifferentialActionModelAbstractTpl<Scalar>
      DifferentialActionModelAbstract;
  typedef DynamicsModelAbstractTpl<Scalar> DynamicsModelAbstract;
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
   * @brief Initialize the symplectic Euler integrator
   *
   * @param[in] model               Differential action model
   * @param[in] control             Control parametrization
   * @param[in] time_step           Step time (default 1e-3)
   * @param[in] with_cost_residual  Compute cost residual (default true)
   */
  IntegratedActionModelEulerTpl(
      std::shared_ptr<DifferentialActionModelAbstract> model,
      std::shared_ptr<ControlParametrizationModelAbstract> control,
      const Scalar time_step = Scalar(1e-3),
      const bool with_cost_residual = true);

  /**
   * @brief Initialize the symplectic Euler integrator
   *
   * This initialization uses `ControlParametrizationPolyZeroTpl` for the
   * control parametrization.
   *
   * @param[in] model               Differential action model
   * @param[in] time_step           Step time (default 1e-3)
   * @param[in] with_cost_residual  Compute cost residual (default true)
   */
  IntegratedActionModelEulerTpl(
      std::shared_ptr<DifferentialActionModelAbstract> model,
      const Scalar time_step = Scalar(1e-3),
      const bool with_cost_residual = true);

  /**
   * @brief Initialize from continuous dynamics, costs and constraints
   *
   * Null control and time arguments select a zero-order control
   * parametrization and a private default integration time, respectively.
   */
  IntegratedActionModelEulerTpl(
      std::shared_ptr<DynamicsModelAbstract> dynamics,
      std::shared_ptr<CostModelSum> costs,
      std::shared_ptr<ConstraintModelManager> constraints = nullptr,
      std::shared_ptr<ControlParametrizationModelAbstract> control = nullptr,
      std::shared_ptr<IntegratorTime> integrator_time = nullptr);
  virtual ~IntegratedActionModelEulerTpl() = default;

  using Base::createData;

  /**
   * @brief Integrate the differential action model using symplectic Euler
   * scheme
   *
   * @param[in] data  Symplectic Euler data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Integrate the total cost value for nodes that depends only on the
   * state using symplectic Euler scheme
   *
   * It computes the total cost and defines the next state as the current one.
   * This function is used in the terminal nodes of an optimal control problem.
   *
   * @param[in] data  Symplectic Euler data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Compute the partial derivatives of the symplectic Euler integrator
   *
   * @param[in] data  Symplectic Euler data
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
   * @param[in] data  Symplectic Euler data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Create the symplectic Euler data
   *
   * @return the symplectic Euler data
   */
  virtual std::shared_ptr<ActionDataAbstract> createData() override;
  virtual std::shared_ptr<ActionDataAbstract> createData(
      const std::shared_ptr<ParameterDataManager>& params_data) override;

  /**
   * @brief Cast the Euler integrated-action model to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return IntegratedActionModelEulerTpl<NewScalar> An action model with the
   * new scalar type.
   */
  template <typename NewScalar>
  IntegratedActionModelEulerTpl<NewScalar> cast() const;

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
   * @param[in] data    Symplectic Euler data
   * @param[out] u      Quasic static commands
   * @param[in] x       State point (velocity has to be zero)
   * @param[in] maxiter Maximum allowed number of iterations
   * @param[in] tol     Tolerance
   */
  virtual void quasiStatic(const std::shared_ptr<ActionDataAbstract>& data,
                           Eigen::Ref<VectorXs> u,
                           const Eigen::Ref<const VectorXs>& x,
                           const std::size_t maxiter = 100,
                           const Scalar tol = Scalar(1e-9)) override;

  /** @brief Attach a parameter manager and rebuild shared backend data */
  void set_params(const std::shared_ptr<ActionDataAbstract>& data,
                  std::shared_ptr<ParameterManager> params) override;

  /** @brief Update active action and dynamics parameters */
  virtual void update_p(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& p) override;

  /**
   * @brief Print relevant information of the Euler integrator model
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
};

/**
 * @brief Data for the symplectic Euler integrated action
 *
 * Backend data, control data, and the optional parameter payload are shared;
 * the Euler rate, control-chain-rule matrices and parameter-gradient scratch
 * are owned by each data object. Repeated same-layout calculations reuse all
 * workspaces.
 */
template <typename _Scalar>
struct IntegratedActionDataEulerTpl
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
  explicit IntegratedActionDataEulerTpl(
      Model<Scalar>* const model,
      const std::shared_ptr<ParameterDataManager>& params_data =
          std::shared_ptr<ParameterDataManager>())
      : Base(model) {
    control = model->get_control()->createData();
    differential.reset();
    dynamics.reset();
    costs.reset();
    constraints.reset();
    if (model->get_dynamics() != nullptr) {
      params = params_data;
      dynamics = params_data != nullptr
                     ? model->get_dynamics()->createData(params_data)
                     : model->get_dynamics()->createData();
      costs = model->get_costs()->createData(dynamics->shared);
      if (model->get_constraints() != nullptr) {
        constraints = model->get_constraints()->createData(dynamics->shared);
      }
    } else if (model->get_differential() != nullptr) {
      differential = model->get_differential()->createData();
    }
    const std::size_t ndx = model->get_state()->get_ndx();
    const std::size_t nv = model->get_state()->get_nv();
    dx = VectorXs::Zero(ndx);
    da_du = MatrixXs::Zero(nv, model->get_nu());
    Lwu = MatrixXs::Zero(model->get_control()->get_nw(), model->get_nu());
  }
  virtual ~IntegratedActionDataEulerTpl() = default;

  template <class Model>
  void resize(Model* const model, const bool running_node = true) {
    ActionDataAbstractTpl<Scalar>::resize(model, running_node);
    dx.resize(model->get_state()->get_ndx());
    da_du.resize(model->get_state()->get_nv(), model->get_nu());
    Lwu.resize(model->get_control()->get_nw(), model->get_nu());
    dx.setZero();
    da_du.setZero();
    Lwu.setZero();
  }

  std::shared_ptr<DifferentialActionDataAbstract>
      differential;                                //!< Differential model data
  std::shared_ptr<DynamicsDataAbstract> dynamics;  //!< Dynamics model data
  std::shared_ptr<CostDataSum> costs;              //!< Cost model data
  std::shared_ptr<ConstraintDataManager>
      constraints;  //!< Constraint-manager data
  std::shared_ptr<ControlParametrizationDataAbstract>
      control;  //!< Control parametrization data
  std::shared_ptr<ParameterDataManager> params;  //!< Shared parameter payload
  VectorXs dx;                                   //!< Integrated state increment
  MatrixXs da_du;  //!< Acceleration derivative through control parametrization
  MatrixXs Lwu;    //!< Hessian of the cost function with respect to the control
                   //!< input (w) and control parameters (u)

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
#include "crocoddyl/core/integrator/euler.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::IntegratedActionModelEulerTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::IntegratedActionDataEulerTpl)

#endif  // CROCODDYL_CORE_INTEGRATOR_EULER_HPP_
