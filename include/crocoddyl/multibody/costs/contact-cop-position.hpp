///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, University of Duisburg-Essen, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_CONTACT_COP_POSITION_HPP_
#define CROCODDYL_MULTIBODY_COSTS_CONTACT_COP_POSITION_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/residuals/contact-cop-position.hpp"
#include "crocoddyl/core/activations/quadratic-barrier.hpp"
#include "crocoddyl/core/utils/exception.hpp"

#include "crocoddyl/multibody/frames-deprecated.hpp"

namespace crocoddyl {

#pragma GCC diagnostic push  // TODO: Remove once the deprecated FrameXX has been removed in a future release
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

/**
 * @brief Define a center of pressure cost function
 *
 * It builds a cost function that bounds the center of pressure (CoP) for one contact surface to
 * lie inside a certain geometric area defined around the reference contact frame. The cost residual
 * vector is described as \f$\mathbf{r}=\mathbf{A}\boldsymbol{\lambda}\geq\mathbf{0}\f$, where
 * \f[
 *   \mathbf{A}=
 *     \begin{bmatrix} 0 & 0 & X/2 & 0 & -1 & 0 \\ 0 & 0 & X/2 & 0 & 1 & 0 \\ 0 & 0 & Y/2 & 1 & 0 & 0 \\
 *      0 & 0 & Y/2 & -1 & 0 & 0
 *     \end{bmatrix}
 * \f]
 * is the inequality matrix and \f$\boldsymbol{\lambda}\f$ is the reference spatial contact force in the frame
 * coordinate. The CoP lies inside the convex hull of the foot, see eq.(18-19) of
 * https://hal.archives-ouvertes.fr/hal-02108449/document is we have:
 * \f[
 *  \begin{align}\begin{split}\tau^x &\leq
 * Yf^z \\-\tau^x &\leq Yf^z \\\tau^y &\leq Yf^z \\-\tau^y &\leq Yf^z
 *  \end{split}\end{align}
 * \f]
 * with \f$\boldsymbol{\lambda}= \begin{bmatrix}f^x & f^y & f^z & \tau^x & \tau^y & \tau^z \end{bmatrix}^T\f$.
 *
 * The cost is computed, from the residual vector \f$\mathbf{r}\f$, through an user defined activation model.
 * Additionally, the contact frame id, the desired support region for the CoP and the inequality matrix
 * are handled within `FrameCoPSupportTpl`. The force vector \f$\boldsymbol{\lambda}\f$ and its derivatives are
 * computed by `DifferentialActionModelContactFwdDynamicsTpl`. These values are stored in a shared data (i.e.
 * `DataCollectorContactTpl`). Note that this cost function cannot be used with other action models.
 *
 * \sa `DifferentialActionModelContactFwdDynamicsTpl`, `DataCollectorContactTpl`, `ActivationModelAbstractTpl`
 */
template <typename _Scalar>
class CostModelContactCoPPositionTpl : public CostModelResidualTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelResidualTpl<Scalar> Base;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef ResidualModelContactCoPPositionTpl<Scalar> ResidualModelContactCoPPosition;
  typedef ActivationModelQuadraticBarrierTpl<Scalar> ActivationModelQuadraticBarrier;
  typedef ActivationBoundsTpl<Scalar> ActivationBounds;
  typedef CoPSupportTpl<Scalar> CoPSupport;
  typedef FrameCoPSupportTpl<Scalar> FrameCoPSupport;
  typedef typename MathBase::Matrix3s Matrix3s;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize the contact CoP cost model
   *
   * @param[in] state        State of the multibody system
   * @param[in] activation   Activation model
   * @param[in] cop_support  Id of contact frame and support region of the CoP
   * @param[in] nu           Dimension of control vector
   */
  CostModelContactCoPPositionTpl(boost::shared_ptr<StateMultibody> state,
                                 boost::shared_ptr<ActivationModelAbstract> activation,
                                 const FrameCoPSupport& cop_support, const std::size_t nu);

  /**
   * @brief Initialize the contact CoP cost model
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state        State of the multibody system
   * @param[in] activation   Activation model
   * @param[in] cop_support  Id of contact frame and support region of the CoP
   */
  CostModelContactCoPPositionTpl(boost::shared_ptr<StateMultibody> state,
                                 boost::shared_ptr<ActivationModelAbstract> activation,
                                 const FrameCoPSupport& cop_support);

  /**
   * @brief Initialize the contact CoP cost model
   *
   * We use as default activation model a quadratic barrier `ActivationModelQuadraticBarrierTpl`, with 0 and inf as
   * lower and upper bounds, respectively.
   *
   * @param[in] state        State of the multibody system
   * @param[in] cop_support  Id of contact frame and support region of the cop
   * @param[in] nu           Dimension of control vector
   */
  CostModelContactCoPPositionTpl(boost::shared_ptr<StateMultibody> state, const FrameCoPSupport& cop_support,
                                 const std::size_t nu);

  /**
   * @brief Initialize the contact CoP cost model
   *
   * We use as default activation model a quadratic barrier `ActivationModelQuadraticBarrierTpl`, with 0 and inf as
   * lower and upper bounds, respectively. Additionally, the default `nu` value is obtained from
   * `StateAbstractTpl::get_nv()`
   *
   * @param[in] state        State of the multibody system
   * @param[in] cop_support  Id of contact frame and support region of the cop
   */
  CostModelContactCoPPositionTpl(boost::shared_ptr<StateMultibody> state, const FrameCoPSupport& cop_support);
  virtual ~CostModelContactCoPPositionTpl();

 protected:
  /**
   * @brief Return the frame CoP support
   */
  virtual void set_referenceImpl(const std::type_info& ti, const void* pv);

  /**
   * @brief Modify the frame CoP support
   */
  virtual void get_referenceImpl(const std::type_info& ti, void* pv);

  using Base::activation_;
  using Base::nu_;
  using Base::residual_;
  using Base::state_;
  using Base::unone_;

 private:
  FrameCoPSupport cop_support_;  //!< Frame name of the contact foot and support region of the CoP
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/costs/contact-cop-position.hxx"

#pragma GCC diagnostic pop

#endif  // CROCODDYL_MULTIBODY_COSTS_CONTACT_COP_POSITION_HPP_
