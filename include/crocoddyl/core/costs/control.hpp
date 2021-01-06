///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_COSTS_CONTROL_HPP_
#define CROCODDYL_CORE_COSTS_CONTROL_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/cost-base.hpp"

namespace crocoddyl {

/**
 * @brief Control cost
 *
 * This cost function defines a residual vector as \f$\mathbf{r}=\mathbf{u}-\mathbf{u}^*\f$, where
 * \f$\mathbf{u},\mathbf{u}^*\in~\mathbb{R}^{nu}\f$ are the current and reference control inputs, respetively. Note
 * that the dimension of the residual vector is obtained from `nu`.
 *
 * Both cost and residual derivatives are computed analytically.
 * For the computation of the cost Hessian, we use the Gauss-Newton approximation, e.g.
 * \f$\mathbf{l_{xx}} = \mathbf{l_{x}}^T \mathbf{l_{x}} \f$.
 *
 * As described in CostModelAbstractTpl(), the cost value and its derivatives are calculated by `calc` and `calcDiff`,
 * respectively.
 *
 * \sa `CostModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class CostModelControlTpl : public CostModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelAbstractTpl<Scalar> Base;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize the control cost model
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   * @param[in] uref        Reference control input
   */
  CostModelControlTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                      boost::shared_ptr<ActivationModelAbstract> activation, const VectorXs& uref);
  /**
   * @brief Initialize the control cost model
   *
   * The default reference control is obtained from `MathBaseTpl<>::VectorXs::Zero(nu)` with `nu` obtained by
   * `ActivationAbstractTpl::get_nr()`.
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   */
  CostModelControlTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                      boost::shared_ptr<ActivationModelAbstract> activation);

  /**
   * @brief Initialize the control cost model
   *
   * The default reference control is obtained from `MathBaseTpl<>::VectorXs::Zero(nu)`.
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   * @param[in] nu          Dimension of the control vector
   */
  CostModelControlTpl(boost::shared_ptr<typename Base::StateAbstract> state,
                      boost::shared_ptr<ActivationModelAbstract> activation, const std::size_t& nu);

  /**
   * @brief Initialize the control cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$). The
   * default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state       State of the multibody system
   * @param[in] uref        Reference control input
   */
  CostModelControlTpl(boost::shared_ptr<typename Base::StateAbstract> state, const VectorXs& uref);

  /**
   * @brief Initialize the control cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$). The
   * default reference control is obtained from `MathBaseTpl<>::VectorXs::Zero(nu)` with `nu` defined by
   * `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model control vector
   */
  explicit CostModelControlTpl(boost::shared_ptr<typename Base::StateAbstract> state);

  /**
   * @brief Initialize the control cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$). The
   * default reference control is obtained from `MathBaseTpl<>::VectorXs::Zero(nu)`.
   *
   * @param[in] state       State of the multibody system
   * @param[in] nu          Dimension of the control vector
   */
  CostModelControlTpl(boost::shared_ptr<typename Base::StateAbstract> state, const std::size_t& nu);
  virtual ~CostModelControlTpl();

  /**
   * @brief Compute the control cost
   *
   * @param[in] data  Control cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the derivatives of the control cost
   *
   * @param[in] data  Control cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

 protected:
  /**
   * @brief Modify the control reference
   */
  virtual void set_referenceImpl(const std::type_info& ti, const void* pv);

  /**
   * @brief Return the state control
   */
  virtual void get_referenceImpl(const std::type_info& ti, void* pv) const;

  using Base::activation_;
  using Base::nu_;
  using Base::state_;
  using Base::unone_;

 private:
  VectorXs uref_;  //!< Reference control input
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/costs/control.hxx"

#endif  // CROCODDYL_CORE_COSTS_CONTROL_HPP_
