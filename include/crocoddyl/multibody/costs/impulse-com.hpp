///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_IMPULSE_COM_HPP_
#define CROCODDYL_MULTIBODY_COSTS_IMPULSE_COM_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/residuals/impulse-com.hpp"
#include "crocoddyl/core/utils/exception.hpp"

#define CROCODDYL_IGNORE_DEPRECATED_HEADER  // TODO: Remove once the deprecated FrameXX has been removed in a future
                                            // release
#include "crocoddyl/multibody/frames.hpp"
#undef CROCODDYL_IGNORE_DEPRECATED_HEADER

namespace crocoddyl {

/**
 * @brief Impulse CoM cost
 *
 * This cost function defines a residual vector as \f$\mathbf{r}=\mathbf{J}_{com}*(\mathbf{v}_{next}-\mathbf{v})\f$,
 * \f$\mathbf{J}_{com}\in\mathbb{R}^{3\times nv}\f$ is the CoM Jacobian, and \f$\mathbf{v}_{next},\mathbf{v}\in
 * T_{\mathbf{q}}\mathcal{Q}\f$ are the generalized velocities after and before the impulse, respectively. Note that
 * the dimension of the residual vector is 3.
 *
 * Both cost and residual derivatives are computed analytically. For the computation of the cost Hessian, we use the
 * Gauss-Newton approximation, e.g. \f$\mathbf{l_{xu}} = \mathbf{l_{x}}^T \mathbf{l_{u}} \f$.
 *
 * As described in `CostModelResidualTpl()`, the cost value and its derivatives are calculated by `calc` and
 * `calcDiff`, respectively.
 *
 * \sa `CostModelResidualTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class CostModelImpulseCoMTpl : public CostModelResidualTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelResidualTpl<Scalar> Base;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef ResidualModelImpulseCoMTpl<Scalar> ResidualModelImpulseCoM;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize the impulse CoM cost model
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   */
  CostModelImpulseCoMTpl(boost::shared_ptr<StateMultibody> state,
                         boost::shared_ptr<ActivationModelAbstract> activation);

  /**
   * @brief Initialize the impulse CoM cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$).
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state  State of the multibody system
   */
  CostModelImpulseCoMTpl(boost::shared_ptr<StateMultibody> state);
  virtual ~CostModelImpulseCoMTpl();

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
#include "crocoddyl/multibody/costs/impulse-com.hxx"

#endif  // CROCODDYL_MULTIBODY_COSTS_IMPULSE_COM_HPP_
