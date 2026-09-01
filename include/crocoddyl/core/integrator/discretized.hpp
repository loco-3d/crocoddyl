///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_INTEGRATOR_DISCRETIZED_HPP_
#define CROCODDYL_CORE_INTEGRATOR_DISCRETIZED_HPP_

#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/constraints/constraint-manager.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/dynamics-base.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/integ-action-base.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"

namespace crocoddyl {

/**
 * @brief Action model for discrete-time dynamics
 *
 * This model wraps a `DynamicsModelAbstractTpl` whose `DynamicsType` is
 * `DiscreteTime`. The dynamics model directly computes the next state
 * \f$\mathbf{x}_{k+1} \in \mathbb{R}^{nx}\f$ and its Jacobians in the tangent
 * space, so no numerical integration step is required.
 *
 * Concretely, `calc()` sets
 * \f[
 *   \mathbf{x}_{k+1} = \mathbf{f}(\mathbf{x}_k, \mathbf{u}_k),
 * \f]
 * and `calcDiff()` reads the Jacobians \f$\mathbf{F_x}\f$ and
 * \f$\mathbf{F_u}\f$ directly from the dynamics data without any integration
 * chain rule. The cost and constraints are not scaled by a time step.
 * Parameter managers are forwarded to the discrete dynamics and shared with
 * the action data. Running and terminal constraint layouts remain distinct;
 * terminal evaluation sets \f$\mathbf{x}_{k+1}=\mathbf{x}_k\f$ and does not
 * evaluate the discrete transition.
 *
 * \sa `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class DiscretizedActionModelTpl : public ActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ActionModelBase, DiscretizedActionModelTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActionModelAbstractTpl<Scalar> Base;
  typedef DiscretizedActionDataTpl<Scalar> Data;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef DynamicsModelAbstractTpl<Scalar> DynamicsModelAbstract;
  typedef DynamicsDataAbstractTpl<Scalar> DynamicsDataAbstract;
  typedef CostModelSumTpl<Scalar> CostModelSum;
  typedef CostDataSumTpl<Scalar> CostDataSum;
  typedef ConstraintModelManagerTpl<Scalar> ConstraintModelManager;
  typedef ConstraintDataManagerTpl<Scalar> ConstraintDataManager;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef ParameterManagerTpl<Scalar> ParameterManager;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the discretized action model
   *
   * @param[in] dynamics     Discrete-time dynamics model
   * @param[in] costs        Cost model stack
   * @param[in] constraints  Constraint manager (default nullptr)
   */
  DiscretizedActionModelTpl(
      std::shared_ptr<DynamicsModelAbstract> dynamics,
      std::shared_ptr<CostModelSum> costs,
      std::shared_ptr<ConstraintModelManager> constraints = nullptr);
  virtual ~DiscretizedActionModelTpl() = default;

  using Base::createData;

  /**
   * @brief Compute the next state and cost value
   *
   * @param[in] data  Discretized action data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{nx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Compute the total cost value for terminal nodes
   *
   * @param[in] data  Discretized action data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{nx}\f$
   */
  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Compute the partial derivatives of the discrete-time dynamics and
   * cost
   *
   * @param[in] data  Discretized action data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{nx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Compute the partial derivatives of the cost for terminal nodes
   *
   * @param[in] data  Discretized action data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{nx}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Create the discretized action data
   *
   * @return the discretized action data
   */
  virtual std::shared_ptr<ActionDataAbstract> createData() override;
  virtual std::shared_ptr<ActionDataAbstract> createData(
      const std::shared_ptr<ParameterDataManager>& params_data) override;

  /**
   * @brief Cast the discretized action model to a different scalar type
   *
   * @tparam NewScalar  New scalar type
   * @return DiscretizedActionModelTpl<NewScalar>
   */
  template <typename NewScalar>
  DiscretizedActionModelTpl<NewScalar> cast() const;

  /**
   * @brief Checks that a specific data belongs to this model
   */
  virtual bool checkData(
      const std::shared_ptr<ActionDataAbstract>& data) override;

  /**
   * @brief Computes the quasic static commands
   *
   * @param[in] data    Discretized action data
   * @param[out] u      Quasi-static commands
   * @param[in] x       State point (velocity must be zero)
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

  /** @brief Update active parameters in the discrete dynamics */
  virtual void update_p(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& p) override;

  virtual std::size_t get_ng() const override;
  virtual std::size_t get_nh() const override;
  virtual std::size_t get_ng_T() const override;
  virtual std::size_t get_nh_T() const override;
  virtual const VectorXs& get_g_lb() const override;
  virtual const VectorXs& get_g_ub() const override;

  /**
   * @brief Return the dynamics model
   */
  const std::shared_ptr<DynamicsModelAbstract>& get_dynamics() const;

  /**
   * @brief Return the cost model stack
   */
  const std::shared_ptr<CostModelSum>& get_costs() const;

  /**
   * @brief Return the constraint manager (may be nullptr)
   */
  const std::shared_ptr<ConstraintModelManager>& get_constraints() const;

  /**
   * @brief Return the parameter manager (may be nullptr)
   */
  const std::shared_ptr<ParameterManager>& get_params() const;

  /**
   * @brief Print relevant information of the discretized action model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override;

 protected:
  void refresh_constraint_bounds() const;

  std::shared_ptr<DynamicsModelAbstract> dynamics_;      //!< Dynamics model
  std::shared_ptr<CostModelSum> costs_;                  //!< Cost model stack
  std::shared_ptr<ConstraintModelManager> constraints_;  //!< Constraint manager
  std::shared_ptr<ParameterManager> params_;             //!< Parameter manager
  mutable VectorXs g_lb_live_;  //!< Live running/terminal inequality bounds
  mutable VectorXs g_ub_live_;  //!< Live running/terminal inequality bounds

  using Base::ng_;
  using Base::nh_;
  using Base::np_;
  using Base::nr_;
  using Base::nu_;
  using Base::state_;
};

/**
 * @brief Data for a discrete dynamics action
 *
 * The specialized dynamics, costs and constraint manager share the dynamics
 * collector. The single constraint-data object is resized between running and
 * terminal layouts. An optional externally supplied parameter payload is
 * retained by shared ownership. The action derivative storage itself is owned
 * by each data object.
 */
template <typename _Scalar>
struct DiscretizedActionDataTpl : public ActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActionDataAbstractTpl<Scalar> Base;
  typedef DynamicsDataAbstractTpl<Scalar> DynamicsDataAbstract;
  typedef CostDataSumTpl<Scalar> CostDataSum;
  typedef ConstraintDataManagerTpl<Scalar> ConstraintDataManager;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit DiscretizedActionDataTpl(
      Model<Scalar>* const model,
      const std::shared_ptr<ParameterDataManager>& params_data =
          std::shared_ptr<ParameterDataManager>())
      : Base(detail::check_integrated_action_data_model(model)) {
    dynamics = params_data != nullptr
                   ? model->get_dynamics()->createData(params_data)
                   : model->get_dynamics()->createData();
    costs = model->get_costs()->createData(dynamics->shared);
    params = params_data;
    if (model->get_constraints() != nullptr) {
      constraints = model->get_constraints()->createData(dynamics->shared);
    }
  }
  virtual ~DiscretizedActionDataTpl() = default;

  std::shared_ptr<DynamicsDataAbstract> dynamics;      //!< Dynamics data
  std::shared_ptr<CostDataSum> costs;                  //!< Cost data
  std::shared_ptr<ConstraintDataManager> constraints;  //!< Constraint data
  std::shared_ptr<ParameterDataManager> params;  //!< Shared parameter data

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
#include "crocoddyl/core/integrator/discretized.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::DiscretizedActionModelTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::DiscretizedActionDataTpl)

#endif  // CROCODDYL_CORE_INTEGRATOR_DISCRETIZED_HPP_
