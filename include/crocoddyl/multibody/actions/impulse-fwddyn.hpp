///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTIONS_IMPULSE_FWDDYN_HPP_
#define CROCODDYL_MULTIBODY_ACTIONS_IMPULSE_FWDDYN_HPP_

#include <stdexcept>

#include <pinocchio/algorithm/compute-all-terms.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/contact-dynamics.hpp>
#include <pinocchio/algorithm/centroidal.hpp>
#include <pinocchio/algorithm/rnea-derivatives.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/constraints/constraint-manager.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/actuations/floating-base.hpp"
#include "crocoddyl/multibody/impulses/multiple-impulses.hpp"
#include "crocoddyl/multibody/data/impulses.hpp"
#include "crocoddyl/multibody/actions/impulse-fwddyn.hpp"

namespace crocoddyl {

/**
 * @brief Action model for impulse forward dynamics in multibody systems.
 *
 * This class implements impulse forward dynamics given a stack of rigid-impulses described in
 * `ImpulseModelMultipleTpl`, i.e.,
 * \f[
 * \left[\begin{matrix}\mathbf{v}^+ \\ -\boldsymbol{\Lambda}\end{matrix}\right] =
 * \left[\begin{matrix}\mathbf{M} & \mathbf{J}^{\top}_c \\ {\mathbf{J}_{c}} & \mathbf{0} \end{matrix}\right]^{-1}
 * \left[\begin{matrix}\mathbf{M}\mathbf{v}^- \\ -e\mathbf{J}_c\mathbf{v}^- \\\end{matrix}\right],
 * \f]
 * where \f$\mathbf{q}\in Q\f$, \f$\mathbf{v}\in\mathbb{R}^{nv}\f$ are the configuration point and generalized velocity
 * (its tangent vector), respectively; \f$\mathbf{v}^+\f$, \f$\mathbf{v}^-\f$ are the discontinuous changes in the
 * generalized velocity (i.e., velocity before and after impact, respectively);
 * \f$\mathbf{J}_c\in\mathbb{R}^{nc\times nv}\f$ is the contact Jacobian expressed in the local frame; and
 * \f$\boldsymbol{\Lambda}\in\mathbb{R}^{nc}\f$ is the impulse vector.
 *
 * The derivatives of the next state and contact impulses are computed efficiently
 * based on the analytical derivatives of Recursive Newton Euler Algorithm (RNEA) as described in
 * \cite mastalli-icra20. Note that the algorithm for computing the RNEA derivatives is described in
 * \cite carpentier-rss18.
 *
 * The stack of cost and constraint functions are implemented in `CostModelSumTpl` and `ConstraintModelAbstractTpl`,
 * respectively. The computation of the impulse dynamics and its derivatives are carrying out inside `calc()` and
 * `calcDiff()` functions, respectively. It is also important to remark that `calcDiff()` computes the derivatives
 * using the latest stored values by `calc()`. Thus, we need to run `calc()` first.
 *
 * \sa `ActionModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ActionModelImpulseFwdDynamicsTpl : public ActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

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
   * It describes the impulse dynamics of a multibody system under rigid-contact constraints defined by
   * `ImpulseModelMultipleTpl`. It computes the cost described in `CostModelSumTpl`.
   *
   * @param[in] state            State of the multibody system
   * @param[in] actuation        Actuation model
   * @param[in] impulses         Stack of rigid impulses
   * @param[in] costs            Stack of cost functions
   * @param[in] r_coeff          Restitution coefficient (default 0.)
   * @param[in] JMinvJt_damping  Damping term used in operational space inertia matrix (default 0.)
   * @param[in] enable_force     Enable the computation of the contact force derivatives (default false)
   */
  ActionModelImpulseFwdDynamicsTpl(boost::shared_ptr<StateMultibody> state,
                                   boost::shared_ptr<ImpulseModelMultiple> impulses,
                                   boost::shared_ptr<CostModelSum> costs, const Scalar r_coeff = Scalar(0.),
                                   const Scalar JMinvJt_damping = Scalar(0.), const bool enable_force = false);

  ActionModelImpulseFwdDynamicsTpl(boost::shared_ptr<StateMultibody> state,
                                   boost::shared_ptr<ImpulseModelMultiple> impulses,
                                   boost::shared_ptr<CostModelSum> costs,
                                   boost::shared_ptr<ConstraintModelManager> constraints,
                                   const Scalar r_coeff = Scalar(0.), const Scalar JMinvJt_damping = Scalar(0.),
                                   const bool enable_force = false);
  virtual ~ActionModelImpulseFwdDynamicsTpl();

  /**
   * @brief Compute the system acceleration, and cost value
   *
   * It computes the system acceleration using the impulse dynamics.
   *
   * @param[in] data  Impulse forward-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the derivatives of the impulse dynamics, and cost function
   *
   * @param[in] data  Impulse forward-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Create the impulse forward-dynamics data
   *
   * @return impulse forward-dynamics data
   */
  virtual boost::shared_ptr<ActionDataAbstract> createData();

  /**
   * @brief Check that the given data belongs to the impulse forward-dynamics data
   */
  virtual bool checkData(const boost::shared_ptr<ActionDataAbstract>& data);

  /**
   * @brief Return the impulse model
   */
  const boost::shared_ptr<ImpulseModelMultiple>& get_impulses() const;

  /**
   * @brief Return the cost model
   */
  const boost::shared_ptr<CostModelSum>& get_costs() const;

  /**
   * @brief Return the constraint model
   */
  const boost::shared_ptr<ConstraintModelManager>& get_constraints() const;

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
   * @brief Return the damping factor used in the operational space inertia matrix
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
   * @brief Modify the damping factor used in the operational space inertia matrix
   */
  void set_damping_factor(const Scalar damping);

  /**
   * @brief Print relevant information of the impulse forward-dynamics model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 protected:
  using Base::state_;  //!< Model of the state

 private:
  boost::shared_ptr<ImpulseModelMultiple> impulses_;       //!< Impulse model
  boost::shared_ptr<CostModelSum> costs_;                  //!< Cost model
  boost::shared_ptr<ConstraintModelManager> constraints_;  //!< Constraint model
  pinocchio::ModelTpl<Scalar>& pinocchio_;                 //!< Pinocchio model
  bool with_armature_;                                     //!< Indicate if we have defined an armature
  VectorXs armature_;                                      //!< Armature vector
  Scalar r_coeff_;                                         //!< Restitution coefficient
  Scalar JMinvJt_damping_;                                 //!< Damping factor used in operational space inertia matrix
  bool enable_force_;  //!< Indicate if we have enabled the computation of the contact-forces derivatives
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
        Kinv(model->get_state()->get_nv() + model->get_impulses()->get_nc_total(),
             model->get_state()->get_nv() + model->get_impulses()->get_nc_total()),
        df_dx(model->get_impulses()->get_nc_total(), model->get_state()->get_ndx()),
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

  pinocchio::DataTpl<Scalar> pinocchio;
  DataCollectorMultibodyInImpulseTpl<Scalar> multibody;
  boost::shared_ptr<CostDataSumTpl<Scalar> > costs;
  boost::shared_ptr<ConstraintDataManagerTpl<Scalar> > constraints;
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

#endif  // CROCODDYL_MULTIBODY_ACTIONS_IMPULSE_FWDDYN_HPP_
