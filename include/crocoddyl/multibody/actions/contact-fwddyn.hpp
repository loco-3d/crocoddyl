///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, CTU, INRIA, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTIONS_CONTACT_FWDDYN_HPP_
#define CROCODDYL_MULTIBODY_ACTIONS_CONTACT_FWDDYN_HPP_

#include <stdexcept>

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/constraints/constraint-manager.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/multibody/data/contacts.hpp"

namespace crocoddyl {

/**
 * @brief Differential action model for contact forward dynamics in multibody systems.
 *
 * This class implements contact forward dynamics given a stack of rigid-contacts described in
 * `ContactModelMultipleTpl`, i.e.,
 * \f[
 * \left[\begin{matrix}\dot{\mathbf{v}} \\ -\boldsymbol{\lambda}\end{matrix}\right] =
 * \left[\begin{matrix}\mathbf{M} & \mathbf{J}^{\top}_c \\ {\mathbf{J}_{c}} & \mathbf{0} \end{matrix}\right]^{-1}
 * \left[\begin{matrix}\boldsymbol{\tau}_b \\ -\mathbf{a}_0 \\\end{matrix}\right],
 * \f]
 * where \f$\mathbf{q}\in Q\f$, \f$\mathbf{v}\in\mathbb{R}^{nv}\f$ are the configuration point and generalized velocity
 * (its tangent vector), respectively; \f$\boldsymbol{\tau}_b=\boldsymbol{\tau} - \mathbf{h}(\mathbf{q},\mathbf{v})\f$
 * is the bias forces that depends on the torque inputs \f$\boldsymbol{\tau}\f$ and the Coriolis effect and gravity
 * field \f$\mathbf{h}(\mathbf{q},\mathbf{v})\f$; \f$\mathbf{J}_c\in\mathbb{R}^{nc\times nv}\f$ is the contact Jacobian
 * expressed in the local frame; and \f$\mathbf{a}_0\in\mathbb{R}^{nc}\f$ is the desired acceleration in the constraint
 * space. To improve stability in the numerical integration, we define PD gains that are similar in spirit to Baumgarte
 * stabilization: \f[ \mathbf{a}_0 = \mathbf{a}_{\lambda(c)} - \alpha \,^oM^{ref}_{\lambda(c)}\ominus\,^oM_{\lambda(c)}
 * - \beta\mathbf{v}_{\lambda(c)}, \f] where \f$\mathbf{v}_{\lambda(c)}\f$, \f$\mathbf{a}_{\lambda(c)}\f$ are the
 * spatial velocity and acceleration at the parent body of the contact \f$\lambda(c)\f$, respectively; \f$\alpha\f$ and
 * \f$\beta\f$ are the stabilization gains; \f$\,^oM^{ref}_{\lambda(c)}\ominus\,^oM_{\lambda(c)}\f$ is the
 * \f$\mathbb{SE}(3)\f$ inverse composition between the reference contact placement and the current one.
 *
 * The derivatives of the system acceleration and contact forces are computed efficiently
 * based on the analytical derivatives of Recursive Newton Euler Algorithm (RNEA) as described in
 * \cite mastalli-icra20. Note that the algorithm for computing the RNEA derivatives is described in
 * \cite carpentier-rss18.
 *
 * The stack of cost and constraint functions are implemented in `CostModelSumTpl` and `ConstraintModelAbstractTpl`,
 * respectively. The computation of the contact dynamics and its derivatives are carrying out inside `calc()` and
 * `calcDiff()` functions, respectively. It is also important to remark that `calcDiff()` computes the derivatives
 * using the latest stored values by `calc()`. Thus, we need to run `calc()` first.
 *
 * \sa `DifferentialActionModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class DifferentialActionModelContactFwdDynamicsTpl : public DifferentialActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef DifferentialActionModelAbstractTpl<Scalar> Base;
  typedef DifferentialActionDataContactFwdDynamicsTpl<Scalar> Data;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef CostModelSumTpl<Scalar> CostModelSum;
  typedef ConstraintModelManagerTpl<Scalar> ConstraintModelManager;
  typedef ContactModelMultipleTpl<Scalar> ContactModelMultiple;
  typedef ActuationModelAbstractTpl<Scalar> ActuationModelAbstract;
  typedef DifferentialActionDataAbstractTpl<Scalar> DifferentialActionDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the contact forward-dynamics action model
   *
   * It describes the dynamics evolution of a multibody system under rigid-contact constraints defined by
   * `ContactModelMultipleTpl`. It computes the cost described in `CostModelSumTpl`.
   *
   * @param[in] state            State of the multibody system
   * @param[in] actuation        Actuation model
   * @param[in] contacts         Stack of rigid contact
   * @param[in] costs            Stack of cost functions
   * @param[in] JMinvJt_damping  Damping term used in operational space inertia matrix (default 0.)
   * @param[in] enable_force     Enable the computation of the contact force derivatives (default false)
   */
  DifferentialActionModelContactFwdDynamicsTpl(boost::shared_ptr<StateMultibody> state,
                                               boost::shared_ptr<ActuationModelAbstract> actuation,
                                               boost::shared_ptr<ContactModelMultiple> contacts,
                                               boost::shared_ptr<CostModelSum> costs,
                                               const Scalar JMinvJt_damping = Scalar(0.),
                                               const bool enable_force = false);

  /**
   * @brief Initialize the contact forward-dynamics action model
   *
   * It describes the dynamics evolution of a multibody system under rigid-contact constraints defined by
   * `ContactModelMultipleTpl`. It computes the cost described in `CostModelSumTpl`.
   *
   * @param[in] state            State of the multibody system
   * @param[in] actuation        Actuation model
   * @param[in] contacts         Stack of rigid contact
   * @param[in] costs            Stack of cost functions
   * @param[in] constraints      Stack of constraints
   * @param[in] JMinvJt_damping  Damping term used in operational space inertia matrix (default 0.)
   * @param[in] enable_force     Enable the computation of the contact force derivatives (default false)
   */
  DifferentialActionModelContactFwdDynamicsTpl(boost::shared_ptr<StateMultibody> state,
                                               boost::shared_ptr<ActuationModelAbstract> actuation,
                                               boost::shared_ptr<ContactModelMultiple> contacts,
                                               boost::shared_ptr<CostModelSum> costs,
                                               boost::shared_ptr<ConstraintModelManager> constraints,
                                               const Scalar JMinvJt_damping = Scalar(0.),
                                               const bool enable_force = false);

  DifferentialActionModelContactFwdDynamicsTpl(boost::shared_ptr<StateMultibody> state,
                                               boost::shared_ptr<ActuationModelAbstract> actuation,
                                               boost::shared_ptr<ContactModelMultiple> contacts,
                                               boost::shared_ptr<CostModelSum> costs,
                                               boost::shared_ptr<ConstraintModelManager> constraints,
                                               const Scalar JMinvJt_damping = Scalar(0.),
                                               const bool enable_force = false);
  virtual ~DifferentialActionModelContactFwdDynamicsTpl();

  /**
   * @brief Compute the system acceleration, and cost value
   *
   * It computes the system acceleration using the contact dynamics.
   *
   * @param[in] data  Contact forward-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the total cost value for nodes that depends only on the state
   *
   * It updates the total cost and the system acceleration is not updated as it is expected to be zero. Additionally,
   * it does not update the contact forces. This function is used in the terminal nodes of an optimal control
   * problem.
   *
   * @param[in] data  Contact forward-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the derivatives of the contact dynamics, and cost function
   *
   * @param[in] data  Contact forward-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the derivatives of the cost functions with respect to the state only
   *
   * It updates the derivatives of the cost function with respect to the state only. Additionally, it does not
   * update the contact forces derivatives. This function is used in the terminal nodes of an optimal control
   * problem.
   *
   * @param[in] data  Action data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Create the contact forward-dynamics data
   *
   * @return contact forward-dynamics data
   */
  virtual boost::shared_ptr<DifferentialActionDataAbstract> createData();

  /**
   * @brief Check that the given data belongs to the contact forward-dynamics data
   */
  virtual bool checkData(const boost::shared_ptr<DifferentialActionDataAbstract>& data);

  /**
   * @brief @copydoc Base::quasiStatic()
   */
  virtual void quasiStatic(const boost::shared_ptr<DifferentialActionDataAbstract>& data, Eigen::Ref<VectorXs> u,
                           const Eigen::Ref<const VectorXs>& x, const std::size_t maxiter = 100,
                           const Scalar tol = Scalar(1e-9));

  /**
   * @brief Return the actuation model
   */
  const boost::shared_ptr<ActuationModelAbstract>& get_actuation() const;

  /**
   * @brief Return the contact model
   */
  const boost::shared_ptr<ContactModelMultiple>& get_contacts() const;

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
   * @brief Return the damping factor used in operational space inertia matrix
   */
  const Scalar get_damping_factor() const;

  /**
   * @brief Modify the armature vector
   */
  void set_armature(const VectorXs& armature);

  /**
   * @brief Modify the damping factor used in operational space inertia matrix
   */
  void set_damping_factor(const Scalar damping);

  /**
   * @brief Print relevant information of the contact forward-dynamics model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 protected:
  using Base::nu_;     //!< Control dimension
  using Base::state_;  //!< Model of the state

 private:
  boost::shared_ptr<ActuationModelAbstract> actuation_;    //!< Actuation model
  boost::shared_ptr<ContactModelMultiple> contacts_;       //!< Contact model
  boost::shared_ptr<CostModelSum> costs_;                  //!< Cost model
  boost::shared_ptr<ConstraintModelManager> constraints_;  //!< Constraint model
  pinocchio::ModelTpl<Scalar>& pinocchio_;                 //!< Pinocchio model
  bool with_armature_;                                     //!< Indicate if we have defined an armature
  VectorXs armature_;                                      //!< Armature vector
  Scalar JMinvJt_damping_;                                 //!< Damping factor used in operational space inertia matrix
  bool enable_force_;  //!< Indicate if we have enabled the computation of the contact-forces derivatives
};

template <typename _Scalar>
struct DifferentialActionDataContactFwdDynamicsTpl : public DifferentialActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DifferentialActionDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit DifferentialActionDataContactFwdDynamicsTpl(Model<Scalar>* const model)
      : Base(model),
        pinocchio(pinocchio::DataTpl<Scalar>(model->get_pinocchio())),
        multibody(&pinocchio, model->get_actuation()->createData(), model->get_contacts()->createData(&pinocchio)),
        costs(model->get_costs()->createData(&multibody)),
        Kinv(model->get_state()->get_nv() + model->get_contacts()->get_nc_total(),
             model->get_state()->get_nv() + model->get_contacts()->get_nc_total()),
        df_dx(model->get_contacts()->get_nc_total(), model->get_state()->get_ndx()),
        df_du(model->get_contacts()->get_nc_total(), model->get_nu()),
        tmp_xstatic(model->get_state()->get_nx()),
        tmp_Jstatic(model->get_state()->get_nv(), model->get_nu() + model->get_contacts()->get_nc_total()) {
    costs->shareMemory(this);
    if (model->get_constraints() != nullptr) {
      constraints = model->get_constraints()->createData(&multibody);
      constraints->shareMemory(this);
    }
    Kinv.setZero();
    df_dx.setZero();
    df_du.setZero();
    tmp_xstatic.setZero();
    tmp_Jstatic.setZero();
    pinocchio.lambda_c.resize(model->get_contacts()->get_nc_total());
    pinocchio.lambda_c.setZero();
  }

  pinocchio::DataTpl<Scalar> pinocchio;
  DataCollectorActMultibodyInContactTpl<Scalar> multibody;
  boost::shared_ptr<CostDataSumTpl<Scalar> > costs;
  boost::shared_ptr<ConstraintDataManagerTpl<Scalar> > constraints;
  MatrixXs Kinv;
  MatrixXs df_dx;
  MatrixXs df_du;
  VectorXs tmp_xstatic;
  MatrixXs tmp_Jstatic;

  using Base::cost;
  using Base::Fu;
  using Base::Fx;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::r;
  using Base::xout;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include <crocoddyl/multibody/actions/contact-fwddyn.hxx>

#endif  // CROCODDYL_MULTIBODY_ACTIONS_CONTACT_FWDDYN_HPP_
