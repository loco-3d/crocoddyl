///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_COSTS_CONTROL_GRAVITY_CONTACT_HPP_
#define CROCODDYL_CORE_COSTS_CONTROL_GRAVITY_CONTACT_HPP_

#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/residuals/contact-control-gravity.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief Control gravity cost
 *
 * This cost function defines a residual vector as \f$\mathbf{r}=\mathbf{u}-(g(q) - \sum
 * \mathbf{J_c}(\mathbf{q})^{\top} f_{\text{ext}})\f$, where \f$\mathbf{u}\in~\mathbb{R}^{nu}\f$ is the current control
 * input, \f$\mathbf{J_c}(\mathbf{q})\f$ is the contact Jacobians, \f$\mathbf{f_c}}\f$ is the contact forces,
 * \f$\mathbf{g}(\mathbf{q})\f$ is the gravity torque corresponding to the* current configuration,
 * \f$\mathbf{q}\in~\mathbb{R}^{nq}\f$ the current position joints input. Note that the dimension of the residual
 * vector is obtained from `nu`.
 *
 * Both cost and residual derivatives are computed analytically. For the computation of the cost Hessian, we use the
 * Gauss-Newton approximation, e.g. \f$\mathbf{l_{xx}} = \mathbf{l_{x}}^T \mathbf{l_{x}} \f$.
 *
 * As described in `CostModelResidualTpl()`, the cost value and its derivatives
 * are calculated by `calc` and `calcDiff`, respectively.
 *
 * \sa `CostModelResidualTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class CostModelControlGravContactTpl : public CostModelResidualTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelResidualTpl<Scalar> Base;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef ActuationModelAbstractTpl<Scalar> ActuationModelAbstract;
  typedef ResidualModelContactControlGravTpl<Scalar> ResidualModelContactControlGrav;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the control gravity contact cost model
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   * @param[in] nu          Dimension of the control vector
   */
  CostModelControlGravContactTpl(boost::shared_ptr<StateMultibody> state,
                                 boost::shared_ptr<ActivationModelAbstract> activation, const std::size_t nu);

  /**
   * @brief Initialize the control gravity contact cost model
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   */
  CostModelControlGravContactTpl(boost::shared_ptr<StateMultibody> state,
                                 boost::shared_ptr<ActivationModelAbstract> activation);

  /**
   * @brief Initialize the control gravity contact cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e.
   * \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$).
   *
   * @param[in] state  State of the multibody system
   * @param[in] nu     Dimension of the control vector
   */
  CostModelControlGravContactTpl(boost::shared_ptr<StateMultibody> state, const std::size_t nu);

  /**
   * @brief Initialize the control gravity contact cost model
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   * We use `ActivationModelQuadTpl` as a default activation model (i.e.
   * \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$).
   *
   * @param[in] state  State of the multibody system
   */
  explicit CostModelControlGravContactTpl(boost::shared_ptr<StateMultibody> state);
  virtual ~CostModelControlGravContactTpl();

 protected:
  using Base::activation_;
  using Base::nu_;
  using Base::residual_;
  using Base::state_;
  using Base::unone_;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/costs/control-gravity-contact.hxx"

#endif  // CROCODDYL_MULTIBODY_COSTS_CONTROL_GRAVITY_CONTACT_HPP_
