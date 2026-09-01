///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_OBSERVER_DISCRETIZED_HPP_
#define CROCODDYL_CORE_OBSERVER_DISCRETIZED_HPP_

#include "crocoddyl/core/constraints/constraint-manager.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/dynamics-base.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/observer-base.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"

namespace crocoddyl {

/**
 * @brief Action model for a discretized observer
 *
 * This model wraps a `DynamicsModelAbstractTpl` whose `DynamicsType` is
 * `DiscreteTime` for use in optimal-estimation problems. The observer control
 * vector \f$w\in\mathbb{R}^{ndx}\f$ represents process noise in tangent space;
 * the underlying dynamics is driven solely by the measured torques (updated via
 * `update_tau`) and does not consume \f$w\f$ directly.
 *
 * Concretely, `calc()` sets
 * \f[
 *   \mathbf{x}_{k+1} = \mathbf{f}(\mathbf{x}_k),
 * \f]
 * and `calcDiff()` reads \f$\mathbf{F_x}\f$ and \f$\mathbf{F_p}\f$ directly
 * from the dynamics data. \f$\mathbf{F_w} = 0\f$ because the dynamics mean is
 * independent of process noise.
 *
 * \sa `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class DiscretizedObserverModelTpl : public ObserverModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ActionModelBase, DiscretizedObserverModelTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ObserverModelAbstractTpl<Scalar> Base;
  typedef DiscretizedObserverDataTpl<Scalar> Data;
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
   * @brief Initialize the discretized observer model
   *
   * @param[in] dynamics     Discrete-time dynamics model
   * @param[in] costs        Cost model stack (nu must equal state.ndx)
   * @param[in] ntau         Measured-torque dimension; it must equal
   *                         dynamics.nu
   * @param[in] constraints  Constraint manager (default nullptr)
   */
  DiscretizedObserverModelTpl(
      std::shared_ptr<DynamicsModelAbstract> dynamics,
      std::shared_ptr<CostModelSum> costs, const std::size_t ntau,
      std::shared_ptr<ConstraintModelManager> constraints = nullptr);
  virtual ~DiscretizedObserverModelTpl() = default;

  using Base::createData;

  /**
   * @brief Compute the next state and cost value
   *
   * The native running dynamics overload is evaluated with its data-owned zero
   * control. Process noise does not enter the mean transition; measured
   * torques are injected through `update_tau`.
   *
   * @param[in] data  Discretized observer data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{nx}\f$
   * @param[in] w     Process noise \f$\mathbf{w}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& w) override;

  /**
   * @brief Compute the total cost value for terminal nodes
   *
   * @param[in] data  Discretized observer data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{nx}\f$
   */
  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Compute the partial derivatives of the discrete-time dynamics and
   * cost
   *
   * \f$\mathbf{F_w}=0\f$ because the dynamics does not depend on process noise.
   *
   * @param[in] data  Discretized observer data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{nx}\f$
   * @param[in] w     Process noise \f$\mathbf{w}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& w) override;

  /**
   * @brief Compute the partial derivatives of the cost for terminal nodes
   *
   * @param[in] data  Discretized observer data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{nx}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Create the discretized observer data
   */
  virtual std::shared_ptr<ActionDataAbstract> createData() override;
  virtual std::shared_ptr<ActionDataAbstract> createData(
      const std::shared_ptr<ParameterDataManager>& params_data) override;

  /**
   * @brief Cast the discretized observer model to a different scalar type
   */
  template <typename NewScalar>
  DiscretizedObserverModelTpl<NewScalar> cast() const;

  /**
   * @brief Checks that a specific data belongs to this model
   */
  virtual bool checkData(
      const std::shared_ptr<ActionDataAbstract>& data) override;

  virtual std::size_t get_ng() const override;
  virtual std::size_t get_nh() const override;
  virtual std::size_t get_ng_T() const override;
  virtual std::size_t get_nh_T() const override;
  virtual const VectorXs& get_g_lb() const override;
  virtual const VectorXs& get_g_ub() const override;

  /**
   * @brief Computes the quasi-static commands
   *
   * @param[in] data    Discretized observer data
   * @param[out] w      Quasi-static process noise, set to zero
   * @param[in] x       State point (velocity must be zero)
   * @param[in] maxiter Maximum allowed number of iterations
   * @param[in] tol     Tolerance
   */
  virtual void quasiStatic(const std::shared_ptr<ActionDataAbstract>& data,
                           Eigen::Ref<VectorXs> w,
                           const Eigen::Ref<const VectorXs>& x,
                           const std::size_t maxiter = 100,
                           const Scalar tol = Scalar(1e-9)) override;

  /**
   * @brief Update the observer parameters and resize data accordingly
   *
   * @param[in] data    Discretized observer data
   * @param[in] params  Parameter manager
   */
  void set_params(const std::shared_ptr<ActionDataAbstract>& data,
                  std::shared_ptr<ParameterManager> params) override;

  /**
   * @brief Update the value of the observer parameters
   *
   * @param[in] data  Discretized observer data
   * @param[in] p     Parameter vector
   */
  virtual void update_p(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& p) override;

  /**
   * @brief Update the measured torques and propagate to the dynamics model
   *
   * @param[in] tau_meas  Measured torque vector
   */
  virtual void update_tau(const Eigen::Ref<const VectorXs>& tau_meas) override;

  /**
   * @brief Print relevant information of the discretized observer model
   */
  virtual void print(std::ostream& os) const override;

  const std::shared_ptr<DynamicsModelAbstract>& get_dynamics() const;
  const std::shared_ptr<CostModelSum>& get_costs() const;
  const std::shared_ptr<ConstraintModelManager>& get_constraints() const;
  const std::shared_ptr<ParameterManager>& get_params() const;

 protected:
  using Base::ng_;
  using Base::ng_T_;
  using Base::nh_;
  using Base::nh_T_;
  using Base::np_;
  using Base::nu_;
  using Base::state_;
  using Base::tau_meas_;

 private:
  std::shared_ptr<Data> cast_data(
      const std::shared_ptr<ActionDataAbstract>& data) const;

  void refresh_constraint_bounds() const;

  std::shared_ptr<DynamicsModelAbstract> dynamics_;
  std::shared_ptr<CostModelSum> costs_;
  std::shared_ptr<ConstraintModelManager> constraints_;
  std::shared_ptr<ParameterManager> params_;
  mutable VectorXs g_lb_live_;
  mutable VectorXs g_ub_live_;
};

template <typename _Scalar>
struct DiscretizedObserverDataTpl : public ObserverDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ObserverDataAbstractTpl<Scalar> Base;
  typedef DynamicsDataAbstractTpl<Scalar> DynamicsDataAbstract;
  typedef CostDataSumTpl<Scalar> CostDataSum;
  typedef ConstraintDataManagerTpl<Scalar> ConstraintDataManager;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit DiscretizedObserverDataTpl(
      Model<Scalar>* const model,
      const std::shared_ptr<ParameterDataManager>& params_data =
          std::shared_ptr<ParameterDataManager>())
      : Base(model) {
    dynamics = params_data != nullptr
                   ? model->get_dynamics()->createData(params_data)
                   : model->get_dynamics()->createData();
    costs = model->get_costs()->createData(dynamics->shared);
    if (model->get_constraints() != nullptr) {
      constraints = model->get_constraints()->createData(dynamics->shared);
    }
    u_zero = VectorXs::Zero(model->get_dynamics()->get_nu());
  }
  virtual ~DiscretizedObserverDataTpl() = default;

  std::shared_ptr<DynamicsDataAbstract> dynamics;
  std::shared_ptr<CostDataSum> costs;
  std::shared_ptr<ConstraintDataManager> constraints;
  VectorXs u_zero;  //!< Zero command used by the discrete dynamics

  using Base::cost;
  using Base::dissipative_E;
  using Base::Ep;
  using Base::Eu;
  using Base::Ex;
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

#include "crocoddyl/core/observer/discretized.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::DiscretizedObserverModelTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::DiscretizedObserverDataTpl)

#endif  // CROCODDYL_CORE_OBSERVER_DISCRETIZED_HPP_
