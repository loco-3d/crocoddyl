///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_COM_POSITION_HPP_
#define CROCODDYL_MULTIBODY_COSTS_COM_POSITION_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/residuals/com-position.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief CoM position cost
 *
 * This cost function defines a residual vector as \f$\mathbf{r}=\mathbf{c}-\mathbf{c}^*\f$, where
 * \f$\mathbf{c},\mathbf{c}^*\in~\mathbb{R}^3\f$ are the current and reference CoM position, respectively. Note that
 * the dimension of the residual vector is obtained from 3.
 *
 * Both cost and residual derivatives are computed analytically. For the computation of the cost Hessian, we use the
 * Gauss-Newton approximation, e.g. \f$\mathbf{l_{xx}} = \mathbf{l_{x}}^T \mathbf{l_{x}} \f$.
 *
 * As described in `CostModelResidualTpl()`, the cost value and its derivatives are calculated by `calc` and
 * `calcDiff`, respectively.
 *
 * \sa `CostModelResidualTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class CostModelCoMPositionTpl : public CostModelResidualTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelResidualTpl<Scalar> Base;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef ResidualModelCoMPositionTpl<Scalar> ResidualModelCoMPosition;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Matrix3xs Matrix3xs;

  /**
   * @brief Initialize the CoM position cost model
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   * @param[in] cref        Reference CoM position
   * @param[in] nu          Dimension of the control vector
   */
  CostModelCoMPositionTpl(boost::shared_ptr<StateMultibody> state,
                          boost::shared_ptr<ActivationModelAbstract> activation, const Vector3s& cref,
                          const std::size_t nu);

  /**
   * @brief Initialize the CoM position cost model
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   * @param[in] cref        Reference CoM position
   */
  CostModelCoMPositionTpl(boost::shared_ptr<StateMultibody> state,
                          boost::shared_ptr<ActivationModelAbstract> activation, const Vector3s& cref);

  /**
   * @brief Initialize the CoM position cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$).
   *
   * @param[in] state  State of the multibody system
   * @param[in] cref   Reference CoM position
   * @param[in] nu     Dimension of the control vector
   */
  CostModelCoMPositionTpl(boost::shared_ptr<StateMultibody> state, const Vector3s& cref, const std::size_t nu);

  /**
   * @brief Initialize the CoM position cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$).
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state  State of the multibody system
   * @param[in] cref   Reference CoM position
   */
  CostModelCoMPositionTpl(boost::shared_ptr<StateMultibody> state, const Vector3s& cref);
  virtual ~CostModelCoMPositionTpl();

 protected:
  /**
   * @brief Modify the CoM position reference
   */
  virtual void set_referenceImpl(const std::type_info& ti, const void* pv);

  /**
   * @brief Return the CoM position reference
   */
  virtual void get_referenceImpl(const std::type_info& ti, void* pv);

  using Base::activation_;
  using Base::nu_;
  using Base::residual_;
  using Base::state_;
  using Base::unone_;

 private:
  Vector3s cref_;  //!< Reference CoM position
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/costs/com-position.hxx"

#endif  // CROCODDYL_MULTIBODY_COSTS_COM_POSITION_HPP_
