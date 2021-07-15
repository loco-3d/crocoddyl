///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_CONTACT_FORCE_HPP_
#define CROCODDYL_MULTIBODY_COSTS_CONTACT_FORCE_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/residuals/contact-force.hpp"
#include "crocoddyl/multibody/frames.hpp"

namespace crocoddyl {

#pragma GCC diagnostic push  // TODO: Remove once the deprecated FrameXX has been removed in a future release
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

/**
 * @brief Define a contact force cost function
 *
 * This cost function defines a residual vector \f$\mathbf{r}=\boldsymbol{\lambda}-\boldsymbol{\lambda}^*\f$,
 * where \f$\boldsymbol{\lambda}, \boldsymbol{\lambda}^*\f$ are the current and reference spatial forces, respectively.
 * The current spatial forces \f$\boldsymbol{\lambda}\in\mathbb{R}^{nc}\f$is computed by
 * `DifferentialActionModelContactFwdDynamicsTpl`, with `nc` as the dimension of the contact.
 *
 * Both cost and residual derivatives are computed analytically, where th force vector \f$\boldsymbol{\lambda}\f$ and
 * its derivatives \f$\left(\frac{\partial\boldsymbol{\lambda}}{\partial\mathbf{x}},
 * \frac{\partial\boldsymbol{\lambda}}{\partial\mathbf{u}}\right)\f$ are computed by
 * `DifferentialActionModelContactFwdDynamicsTpl`. These values are stored in a shared data (i.e.
 * `DataCollectorContactTpl`). Note that this cost function cannot be used with other action models.
 * For the computation of the cost Hessian, we use the Gauss-Newton approximation, e.g.
 * \f$\mathbf{l_{xu}} = \mathbf{l_{x}}^T \mathbf{l_{u}} \f$.
 *
 * \sa `DifferentialActionModelContactFwdDynamicsTpl`, `DataCollectorContactTpl`, `ActivationModelAbstractTpl`
 */
template <typename _Scalar>
class CostModelContactForceTpl : public CostModelResidualTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelResidualTpl<Scalar> Base;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef ResidualModelContactForceTpl<Scalar> ResidualModelContactForce;
  typedef FrameForceTpl<Scalar> FrameForce;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the contact force cost model
   *
   * Note that the `nr`, defined in the activation model, has to be lower / equals than 6.
   *
   * @param[in] state       Multibody state
   * @param[in] activation  Activation model
   * @param[in] fref        Reference spatial contact force \f$\boldsymbol{\lambda}^*\f$
   * @param[in] nu          Dimension of control vector
   */
  CostModelContactForceTpl(boost::shared_ptr<StateMultibody> state,
                           boost::shared_ptr<ActivationModelAbstract> activation, const FrameForce& fref,
                           const std::size_t nu);

  /**
   * @brief Initialize the contact force cost model
   *
   * The default `nu` is obtained from `StateAbstractTpl::get_nv()`. Note that the `nr`, defined in the activation
   * model, has to be lower / equals than 6.
   *
   * @param[in] state       Multibody state
   * @param[in] activation  Activation model
   * @param[in] fref        Reference spatial contact force \f$\boldsymbol{\lambda}^*\f$
   */
  CostModelContactForceTpl(boost::shared_ptr<StateMultibody> state,
                           boost::shared_ptr<ActivationModelAbstract> activation, const FrameForce& fref);

  /**
   * @brief Initialize the contact force cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$).
   * Note that the `nr`, defined in the activation model, has to be lower / equals than 6.
   *
   * @param[in] state  Multibody state
   * @param[in] fref   Reference spatial contact force \f$\boldsymbol{\lambda}^*\f$
   * @param[in] nr     Dimension of residual vector
   * @param[in] nu     Dimension of control vector
   */
  CostModelContactForceTpl(boost::shared_ptr<StateMultibody> state, const FrameForce& fref, const std::size_t nr,
                           const std::size_t nu);

  /**
   * @brief Initialize the contact force cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$), and `nu`
   * is obtained from `StateAbstractTpl::get_nv()`. Note that the `nr`, defined in the activation model, has to be
   * lower / equals than 6.
   *
   * @param[in] state  Multibody state
   * @param[in] fref   Reference spatial contact force \f$\boldsymbol{\lambda}^*\f$
   * @param[in] nr     Dimension of residual vector
   */
  CostModelContactForceTpl(boost::shared_ptr<StateMultibody> state, const FrameForce& fref, const std::size_t nr);

  /**
   * @brief Initialize the contact force cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$), and `nr`
   * and `nu` are equals to 6 and `StateAbstractTpl::get_nv()`, respectively.
   *
   * @param[in] state  Multibody state
   * @param[in] fref   Reference spatial contact force \f$\boldsymbol{\lambda}^*\f$
   */
  CostModelContactForceTpl(boost::shared_ptr<StateMultibody> state, const FrameForce& fref);
  virtual ~CostModelContactForceTpl();

 protected:
  /**
   * @brief Modify the reference spatial contact force \f$\boldsymbol{\lambda}^*\f$
   */
  virtual void get_referenceImpl(const std::type_info& ti, void* pv);

  /**
   * @brief Return the reference spatial contact force \f$\boldsymbol{\lambda}^*\f$
   */
  virtual void set_referenceImpl(const std::type_info& ti, const void* pv);

  using Base::activation_;
  using Base::nu_;
  using Base::residual_;
  using Base::state_;
  using Base::unone_;

 protected:
  FrameForce fref_;  //!< Reference spatial contact force \f$\boldsymbol{\lambda}^*\f$
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/costs/contact-force.hxx"

#pragma GCC diagnostic pop

#endif  // CROCODDYL_MULTIBODY_COSTS_CONTACT_FORCE_HPP_
