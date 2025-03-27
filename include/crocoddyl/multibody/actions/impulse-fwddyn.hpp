///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          University of Oxford, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTIONS_IMPULSE_FWDDYN_HPP_
#define CROCODDYL_MULTIBODY_ACTIONS_IMPULSE_FWDDYN_HPP_

#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/constraints/constraint-manager.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/multibody/actions/impulse-fwddyn.hpp"
#include "crocoddyl/multibody/actuations/floating-base.hpp"
#include "crocoddyl/multibody/data/impulses.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/impulses/multiple-impulses.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

/**
 * @brief Action model for impulse forward dynamics in multibody systems.
 *
 * This class implements impulse forward dynamics given a stack of
 * rigid-impulses described in `ImpulseModelMultipleTpl`, i.e., \f[
 * \left[\begin{matrix}\mathbf{v}^+ \\ -\boldsymbol{\Lambda}\end{matrix}\right]
 * = \left[\begin{matrix}\mathbf{M} & \mathbf{J}^{\top}_c \\ {\mathbf{J}_{c}} &
 * \mathbf{0} \end{matrix}\right]^{-1}
 * \left[\begin{matrix}\mathbf{M}\mathbf{v}^- \\ -e\mathbf{J}_c\mathbf{v}^-
 * \\\end{matrix}\right], \f] where \f$\mathbf{q}\in Q\f$,
 * \f$\mathbf{v}\in\mathbb{R}^{nv}\f$ are the configuration point and
 * generalized velocity (its tangent vector), respectively; \f$\mathbf{v}^+\f$,
 * \f$\mathbf{v}^-\f$ are the discontinuous changes in the generalized velocity
 * (i.e., velocity before and after impact, respectively);
 * \f$\mathbf{J}_c\in\mathbb{R}^{nc\times nv}\f$ is the contact Jacobian
 * expressed in the local frame; and
 * \f$\boldsymbol{\Lambda}\in\mathbb{R}^{nc}\f$ is the impulse vector.
 *
 * The derivatives of the next state and contact impulses are computed
 * efficiently based on the analytical derivatives of Recursive Newton Euler
 * Algorithm (RNEA) as described in \cite mastalli-icra20. Note that the
 * algorithm for computing the RNEA derivatives is described in \cite
 * carpentier-rss18.
 *
 * The stack of cost and constraint functions are implemented in
 * `CostModelSumTpl` and `ConstraintModelAbstractTpl`, respectively. The
 * computation of the impulse dynamics and its derivatives are carrying out
 * inside `calc()` and `calcDiff()` functions, respectively. It is also
 * important to remark that `calcDiff()` computes the derivatives using the
 * latest stored values by `calc()`. Thus, we need to run `calc()` first.
 *
 * \sa `ActionModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ActionModelImpulseFwdDynamicsTpl
    : public ActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ActionModelBase, ActionModelImpulseFwdDynamicsTpl)

  typedef _Scalar Scalar;
  typedef ActionModelAbstractTpl<Scalar> Base;
  typedef ActionDataImpulseFwdDynamicsTpl<Scalar> Data;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelSumTpl<Scalar> CostModelSum;
  typedef ConstraintModelManagerTpl<Scalar> ConstraintModelManager;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef ImpulseModelMultipleTpl<Scalar> ImpulseModelMultiple;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the impulse forward-dynamics action model
   *
   * It describes the impulse dynamics of a multibody system under rigid-contact
   * constraints defined by `ImpulseModelMultipleTpl`. It computes the cost
   * described in `CostModelSumTpl`.
   *
   * @param[in] state            State of the multibody system
   * @param[in] actuation        Actuation model
   * @param[in] impulses         Stack of rigid impulses
   * @param[in] costs            Stack of cost functions
   * @param[in] r_coeff          Restitution coefficient (default 0.)
   * @param[in] JMinvJt_damping  Damping term used in operational space inertia
   * matrix (default 0.)
   * @param[in] enable_force     Enable the computation of the contact force
   * derivatives (default false)
   */
  ActionModelImpulseFwdDynamicsTpl(
      std::shared_ptr<StateMultibody> state,
      std::shared_ptr<ImpulseModelMultiple> impulses,
      std::shared_ptr<CostModelSum> costs, const Scalar r_coeff = Scalar(0.),
      const Scalar JMinvJt_damping = Scalar(0.),
      const bool enable_force = false);

  /**
   * @brief Initialize the impulse forward-dynamics action model
   *
   * It describes the impulse dynamics of a multibody system under rigid-contact
   * constraints defined by `ImpulseModelMultipleTpl`. It computes the cost
   * described in `CostModelSumTpl`.
   *
   * @param[in] state            State of the multibody system
   * @param[in] actuation        Actuation model
   * @param[in] impulses         Stack of rigid impulses
   * @param[in] costs            Stack of cost functions
   * @param[in] constraints      Stack of constraints
   * @param[in] r_coeff          Restitution coefficient (default 0.)
   * @param[in] JMinvJt_damping  Damping term used in operational space inertia
   * matrix (default 0.)
   * @param[in] enable_force     Enable the computation of the contact force
   * derivatives (default false)
   */
  ActionModelImpulseFwdDynamicsTpl(
      std::shared_ptr<StateMultibody> state,
      std::shared_ptr<ImpulseModelMultiple> impulses,
      std::shared_ptr<CostModelSum> costs,
      std::shared_ptr<ConstraintModelManager> constraints,
      const Scalar r_coeff = Scalar(0.),
      const Scalar JMinvJt_damping = Scalar(0.),
      const bool enable_force = false);
  virtual ~ActionModelImpulseFwdDynamicsTpl() = default;

  /**
   * @brief Compute the system acceleration, and cost value
   *
   * It computes the system acceleration using the impulse dynamics.
   *
   * @param[in] data  Impulse forward-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Compute the total cost value for nodes that depends only on the
   * state
   *
   * It updates the total cost and the system acceleration is not updated as it
   * is expected to be zero. Additionally, it does not update the contact
   * forces. This function is used in the terminal nodes of an optimal control
   * problem.
   *
   * @param[in] data  Impulse forward-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calc(const std::shared_ptr<ActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Compute the derivatives of the impulse dynamics, and cost function
   *
   * @param[in] data  Impulse forward-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Compute the derivatives of the cost functions with respect to the
   * state only
   *
   * It updates the derivatives of the cost function with respect to the state
   * only. Additionally, it does not update the contact forces derivatives. This
   * function is used in the terminal nodes of an optimal control problem.
   *
   * @param[in] data  Impulse forward-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Create the impulse forward-dynamics data
   *
   * @return impulse forward-dynamics data
   */
  virtual std::shared_ptr<ActionDataAbstract> createData() override;

  /**
   * @brief Cast the impulse-fwddyn model to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return ActionModelImpulseFwdDynamicsTpl<NewScalar> An action model with
   * the new scalar type.
   */
  template <typename NewScalar>
  ActionModelImpulseFwdDynamicsTpl<NewScalar> cast() const;

  /**
   * @brief Check that the given data belongs to the impulse forward-dynamics
   * data
   */
  virtual bool checkData(
      const std::shared_ptr<ActionDataAbstract>& data) override;

  /**
   * @brief @copydoc Base::quasiStatic()
   */
  virtual void quasiStatic(const std::shared_ptr<ActionDataAbstract>& data,
                           Eigen::Ref<VectorXs> u,
                           const Eigen::Ref<const VectorXs>& x,
                           const std::size_t maxiter = 100,
                           const Scalar tol = Scalar(1e-9)) override;

  /**
   * @brief Return the number of inequality constraints
   */
  virtual std::size_t get_ng() const override;

  /**
   * @brief Return the number of equality constraints
   */
  virtual std::size_t get_nh() const override;

  /**
   * @brief Return the number of equality terminal constraints
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
   * @brief Return the impulse model
   */
  const std::shared_ptr<ImpulseModelMultiple>& get_impulses() const;

  /**
   * @brief Return the cost model
   */
  const std::shared_ptr<CostModelSum>& get_costs() const;

  /**
   * @brief Return the constraint model manager
   */
  const std::shared_ptr<ConstraintModelManager>& get_constraints() const;

  /**
   * @brief Return the Pinocchio model
   */
  pinocchio::ModelTpl<Scalar>& get_pinocchio() const;

  /**
   * @brief Return the armature vector
   */
  const VectorXs& get_armature() const;

  /**
   * @brief Return the restituion coefficient
   */
  const Scalar get_restitution_coefficient() const;

  /**
   * @brief Return the damping factor used in the operational space inertia
   * matrix
   */
  const Scalar get_damping_factor() const;

  /**
   * @brief Modify the armature vector
   */
  void set_armature(const VectorXs& armature);

  /**
   * @brief Modify the restituion coefficient
   */
  void set_restitution_coefficient(const Scalar r_coeff);

  /**
   * @brief Modify the damping factor used in the operational space inertia
   * matrix
   */
  void set_damping_factor(const Scalar damping);

  /**
   * @brief Print relevant information of the impulse forward-dynamics model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override;

 protected:
  using Base::g_lb_;   //!< Lower bound of the inequality constraints
  using Base::g_ub_;   //!< Upper bound of the inequality constraints
  using Base::state_;  //!< Model of the state

 private:
  void init();
  void initCalc(Data* data, const Eigen::Ref<const VectorXs>& x);
  void initCalcDiff(Data* data, const Eigen::Ref<const VectorXs>& x);
  std::shared_ptr<ImpulseModelMultiple> impulses_;       //!< Impulse model
  std::shared_ptr<CostModelSum> costs_;                  //!< Cost model
  std::shared_ptr<ConstraintModelManager> constraints_;  //!< Constraint model
  pinocchio::ModelTpl<Scalar>* pinocchio_;               //!< Pinocchio model
  bool with_armature_;      //!< Indicate if we have defined an armature
  VectorXs armature_;       //!< Armature vector
  Scalar r_coeff_;          //!< Restitution coefficient
  Scalar JMinvJt_damping_;  //!< Damping factor used in operational space
                            //!< inertia matrix
  bool enable_force_;  //!< Indicate if we have enabled the computation of the
                       //!< contact-forces derivatives
  pinocchio::MotionTpl<Scalar> gravity_;  //! Gravity acceleration
};

template <typename _Scalar>
struct ActionDataImpulseFwdDynamicsTpl : public ActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActionDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit ActionDataImpulseFwdDynamicsTpl(Model<Scalar>* const model)
      : Base(model),
        pinocchio(pinocchio::DataTpl<Scalar>(model->get_pinocchio())),
        multibody(&pinocchio, model->get_impulses()->createData(&pinocchio)),
        costs(model->get_costs()->createData(&multibody)),
        vnone(model->get_state()->get_nv()),
        Kinv(model->get_state()->get_nv() +
                 model->get_impulses()->get_nc_total(),
             model->get_state()->get_nv() +
                 model->get_impulses()->get_nc_total()),
        df_dx(model->get_impulses()->get_nc_total(),
              model->get_state()->get_ndx()),
        dgrav_dq(model->get_state()->get_nv(), model->get_state()->get_nv()) {
    costs->shareMemory(this);
    if (model->get_constraints() != nullptr) {
      constraints = model->get_constraints()->createData(&multibody);
      constraints->shareMemory(this);
    }
    vnone.setZero();
    Kinv.setZero();
    df_dx.setZero();
    dgrav_dq.setZero();
  }
  virtual ~ActionDataImpulseFwdDynamicsTpl() = default;

  pinocchio::DataTpl<Scalar> pinocchio;
  DataCollectorMultibodyInImpulseTpl<Scalar> multibody;
  std::shared_ptr<CostDataSumTpl<Scalar> > costs;
  std::shared_ptr<ConstraintDataManagerTpl<Scalar> > constraints;
  VectorXs vnone;
  MatrixXs Kinv;
  MatrixXs df_dx;
  MatrixXs dgrav_dq;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include <crocoddyl/multibody/actions/impulse-fwddyn.hxx>

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::ActionModelImpulseFwdDynamicsTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::ActionDataImpulseFwdDynamicsTpl)

#endif  // CROCODDYL_MULTIBODY_ACTIONS_IMPULSE_FWDDYN_HPP_
