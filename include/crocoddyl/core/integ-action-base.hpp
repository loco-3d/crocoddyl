///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2026, LAAS-CNRS, University of Edinburgh,
//                          University of Oxford, University of Trento,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_INTEGRATED_ACTION_BASE_HPP_
#define CROCODDYL_CORE_INTEGRATED_ACTION_BASE_HPP_

#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/control-base.hpp"
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/core/dynamics-base.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/integrator/time.hpp"
#include "crocoddyl/core/utils/deprecate.hpp"

namespace crocoddyl {
namespace detail {

template <typename Model>
Model* check_integrated_action_data_model(Model* const model) {
  if (model == nullptr) {
    throw_pretty("Invalid argument: model is null");
  }
  return model;
}

}  // namespace detail

/**
 * @brief Abstract class for an integrated action model
 *
 * An integrated action model is a special kind of action model that is obtained
 * by applying a numerical integration scheme to a continuous-time model.
 * Legacy Crocoddyl action integrators own a
 * `DifferentialActionModelAbstractTpl`. The compositional backend instead
 * shares a continuous `DynamicsModelAbstractTpl`, a `CostModelSumTpl`, and an
 * optional `ConstraintModelManagerTpl`. Both backends share the control
 * parametrization and `IntegratorTimeTpl`; copies retain shared ownership of
 * these objects, while scalar casts create scalar-compatible copies.
 *
 * Running nodes evaluate dynamics, cost, constraints and their state,
 * control, and parameter derivatives. Terminal nodes evaluate the state-only
 * cost and terminal constraints and leave control-only blocks untouched.
 * Parameterized compositional models must call `set_params()` before
 * `update_p()`, `calc()` or `calcDiff()` on parameter-aware data.
 *
 * \sa `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class IntegratedActionModelAbstractTpl
    : public ActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActionModelAbstractTpl<Scalar> Base;
  typedef IntegratedActionDataAbstractTpl<Scalar> Data;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef DifferentialActionModelAbstractTpl<Scalar>
      DifferentialActionModelAbstract;
  typedef DynamicsModelAbstractTpl<Scalar> DynamicsModelAbstract;
  typedef CostModelSumTpl<Scalar> CostModelSum;
  typedef ConstraintModelManagerTpl<Scalar> ConstraintModelManager;
  typedef ControlParametrizationModelAbstractTpl<Scalar>
      ControlParametrizationModelAbstract;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef ParameterManagerTpl<Scalar> ParameterManager;
  typedef IntegratorTimeTpl<Scalar> IntegratorTime;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize a legacy integrator with explicit control parametrization
   *
   * @param[in] model               Differential action model
   * @param[in] control             Control parametrization
   * @param[in] time_step           Integration step
   * @param[in] with_cost_residual  Whether to expose cost residuals
   */
  IntegratedActionModelAbstractTpl(
      std::shared_ptr<DifferentialActionModelAbstract> model,
      std::shared_ptr<ControlParametrizationModelAbstract> control,
      const Scalar time_step = Scalar(1e-3),
      const bool with_cost_residual = true);

  /**
   * @brief Initialize a legacy integrator with zero-order control
   *
   * @param[in] model               Differential action model
   * @param[in] time_step           Integration step
   * @param[in] with_cost_residual  Whether to expose cost residuals
   */
  IntegratedActionModelAbstractTpl(
      std::shared_ptr<DifferentialActionModelAbstract> model,
      const Scalar time_step = Scalar(1e-3),
      const bool with_cost_residual = true);

  /**
   * @brief Initialize an integrator from dynamics, costs and constraints
   *
   * A null control selects `ControlParametrizationModelPolyZeroTpl`. A null
   * integration-time object creates a private default time description.
   * Passing a non-null time object shares its ownership and live time step.
   */
  IntegratedActionModelAbstractTpl(
      std::shared_ptr<DynamicsModelAbstract> dynamics,
      std::shared_ptr<CostModelSum> costs,
      std::shared_ptr<ConstraintModelManager> constraints = nullptr,
      std::shared_ptr<ControlParametrizationModelAbstract> control = nullptr,
      std::shared_ptr<IntegratorTime> integrator_time = nullptr);

  virtual ~IntegratedActionModelAbstractTpl() = default;

  using Base::createData;
  /** @brief Allocate data for the selected backend. */
  virtual std::shared_ptr<ActionDataAbstract> createData() override;
  /** @brief Allocate data sharing an existing parameter payload. */
  virtual std::shared_ptr<ActionDataAbstract> createData(
      const std::shared_ptr<ParameterDataManager>& params_data) override;

  /** @brief Return the live number of running inequality rows. */
  virtual std::size_t get_ng() const override;

  /** @brief Return the live number of running equality rows. */
  virtual std::size_t get_nh() const override;

  /** @brief Return the live number of terminal inequality rows. */
  virtual std::size_t get_ng_T() const override;

  /** @brief Return the live number of terminal equality rows. */
  virtual std::size_t get_nh_T() const override;

  /** @brief Return live running inequality lower bounds. */
  virtual const VectorXs& get_g_lb() const override;

  /** @brief Return live running inequality upper bounds. */
  virtual const VectorXs& get_g_ub() const override;

  /** @brief Return the legacy differential backend, or nullptr. */
  const std::shared_ptr<DifferentialActionModelAbstract>& get_differential()
      const;

  /** @brief Return the compositional dynamics backend, or nullptr. */
  const std::shared_ptr<DynamicsModelAbstract>& get_dynamics() const;

  /** @brief Return the compositional cost stack, or nullptr. */
  const std::shared_ptr<CostModelSum>& get_costs() const;

  /** @brief Return the compositional constraint manager, or nullptr. */
  const std::shared_ptr<ConstraintModelManager>& get_constraints() const;

  /** @brief Return the shared control parametrization. */
  const std::shared_ptr<ControlParametrizationModelAbstract>& get_control()
      const;

  /** @brief Return the shared, mutable integration-time description. */
  const std::shared_ptr<IntegratorTime>& get_integrator_time() const;

  /** @brief Return the current live integration step. */
  const Scalar get_dt() const;

  /** @brief Set the shared integration step. */
  void set_dt(const Scalar dt);

  DEPRECATED("The DifferentialActionModel should be set at construction time",
             void set_differential(
                 std::shared_ptr<DifferentialActionModelAbstract> model));

 protected:
  using Base::g_lb_;
  using Base::g_ub_;
  using Base::has_control_limits_;
  using Base::ng_;
  using Base::ng_T_;
  using Base::nh_;
  using Base::nh_T_;
  using Base::np_;
  using Base::nr_;
  using Base::nu_;
  using Base::state_;
  using Base::u_lb_;
  using Base::u_ub_;

  IntegratedActionModelAbstractTpl()
      : differential_(nullptr),
        dynamics_(nullptr),
        costs_(nullptr),
        constraints_(nullptr),
        control_(nullptr),
        params_(nullptr),
        integrator_time_(std::make_shared<IntegratorTime>(Scalar(0.), false)),
        time_step_(0.),
        time_step2_(0.),
        with_cost_residual_(false) {}

  void init();
  void refresh_integrator_time();
  void refresh_constraint_bounds() const;

  std::shared_ptr<DifferentialActionModelAbstract> differential_;
  std::shared_ptr<DynamicsModelAbstract> dynamics_;
  std::shared_ptr<CostModelSum> costs_;
  std::shared_ptr<ConstraintModelManager> constraints_;
  std::shared_ptr<ControlParametrizationModelAbstract> control_;
  std::shared_ptr<ParameterManager> params_;
  std::shared_ptr<IntegratorTime> integrator_time_;
  Scalar time_step_;
  Scalar time_step2_;
  bool with_cost_residual_;
  mutable VectorXs g_lb_live_;
  mutable VectorXs g_ub_live_;
};

/**
 * @brief Common data for differential- and dynamics-backed integrators
 *
 * The action derivatives and constraint blocks are resized to the current
 * running or terminal layout. Concrete data owns its numerical workspaces and
 * shares the backend model data and optional parameter-manager payload.
 */
template <typename _Scalar>
struct IntegratedActionDataAbstractTpl : public ActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActionDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit IntegratedActionDataAbstractTpl(Model<Scalar>* const model)
      : Base(detail::check_integrated_action_data_model(model)) {}
  virtual ~IntegratedActionDataAbstractTpl() = default;

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
#include "crocoddyl/core/integ-action-base.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::IntegratedActionModelAbstractTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::IntegratedActionDataAbstractTpl)

#endif  // CROCODDYL_CORE_INTEGRATED_ACTION_BASE_HPP_
