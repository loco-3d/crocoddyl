///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_FRAME_VELOCITY_HPP_
#define CROCODDYL_MULTIBODY_COSTS_FRAME_VELOCITY_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/residuals/frame-velocity.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/frames.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief Frame velocity cost
 *
 * This cost function defines a residual vector as \f$\mathbf{r}=\mathbf{v}-\mathbf{v}^*\f$, where
 * \f$\mathbf{v},\mathbf{v}^*\in~T_{\mathbf{p}}~\mathbb{SE(3)}\f$ are the current and reference frame velocities,
 * respectively. Note that the tangent vector is described by the frame placement \f$\mathbf{p}\f$, and the dimension
 * of the residual vector is 6.
 *
 * Both cost and residual derivatives are computed analytically. For the computation of the cost Hessian, we use the
 * Gauss-Newton approximation, e.g. \f$\mathbf{l_{xu}} = \mathbf{l_{x}}^T \mathbf{l_{u}} \f$.
 *
 * As described in `CostModelResidualTpl`, the cost value and its derivatives are calculated by `calc` and `calcDiff`,
 * respectively.
 *
 * \sa `CostModelResidualTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class CostModelFrameVelocityTpl : public CostModelResidualTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelResidualTpl<Scalar> Base;
  typedef CostDataFrameVelocityTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef ResidualModelFrameVelocityTpl<Scalar> ResidualModelFrameVelocity;
  typedef FrameMotionTpl<Scalar> FrameMotion;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize the frame velocity cost model
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   * @param[in] Fref        Reference frame velocity
   * @param[in] nu          Dimension of the control vector
   */
  CostModelFrameVelocityTpl(boost::shared_ptr<StateMultibody> state,
                            boost::shared_ptr<ActivationModelAbstract> activation, const FrameMotion& Fref,
                            const std::size_t nu);

  /**
   * @brief Initialize the frame velocity cost model
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   * @param[in] Fref        Reference frame velocity
   */
  CostModelFrameVelocityTpl(boost::shared_ptr<StateMultibody> state,
                            boost::shared_ptr<ActivationModelAbstract> activation, const FrameMotion& Fref);

  /**
   * @brief Initialize the frame velocity cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$).
   *
   * @param[in] state  State of the multibody system
   * @param[in] Fref   Reference frame velocity
   * @param[in] nu     Dimension of the control vector
   */
  CostModelFrameVelocityTpl(boost::shared_ptr<StateMultibody> state, const FrameMotion& Fref, const std::size_t nu);

  /**
   * @brief Initialize the frame velocity cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$).
   * Furthermore, the default `nu` value is obtained from `StateAbstractTpl::get_nv()`
   *
   * @param[in] state  State of the multibody system
   * @param[in] Fref   Reference frame velocity
   */
  CostModelFrameVelocityTpl(boost::shared_ptr<StateMultibody> state, const FrameMotion& Fref);
  virtual ~CostModelFrameVelocityTpl();

 protected:
  /**
   * @brief Return the frame velocity reference
   */
  virtual void get_referenceImpl(const std::type_info& ti, void* pv);

  /**
   * @brief Modify the frame velocity reference
   */
  virtual void set_referenceImpl(const std::type_info& ti, const void* pv);

  using Base::activation_;
  using Base::nu_;
  using Base::residual_;
  using Base::state_;
  using Base::unone_;

 private:
  FrameMotion vref_;  //!< Reference frame velocity
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/costs/frame-velocity.hxx"

#endif  // CROCODDYL_MULTIBODY_COSTS_FRAME_VELOCITY_HPP_
