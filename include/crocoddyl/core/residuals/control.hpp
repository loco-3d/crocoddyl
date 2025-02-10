///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_RESIDUALS_CONTROL_HPP_
#define CROCODDYL_CORE_RESIDUALS_CONTROL_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/residual-base.hpp"

namespace crocoddyl {

/**
 * @brief Define a control residual
 *
 * This residual function is defined as
 * \f$\mathbf{r}=\mathbf{u}-\mathbf{u}^*\f$, where
 * \f$\mathbf{u},\mathbf{u}^*\in~\mathbb{R}^{nu}\f$ are the current and
 * reference control inputs, respectively. Note that the dimension of the
 * residual vector is obtained from `nu`.
 *
 * Both residual and residual Jacobians are computed analytically.
 *
 * As described in ResidualModelAbstractTpl(), the residual value and its
 * Jacobians are calculated by `calc` and `calcDiff`, respectively.
 *
 * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ResidualModelControlTpl : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the control residual model
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state  State of the multibody system
   * @param[in] uref   Reference control input
   */
  ResidualModelControlTpl(std::shared_ptr<typename Base::StateAbstract> state,
                          const VectorXs& uref);

  /**
   * @brief Initialize the control residual model
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state  State of the multibody system
   * @param[in] nu     Dimension of the control vector
   */
  ResidualModelControlTpl(std::shared_ptr<typename Base::StateAbstract> state,
                          const std::size_t nu);

  /**
   * @brief Initialize the control residual model
   *
   * The default reference control is obtained from
   * `MathBaseTpl<>::VectorXs::Zero(nu)`.
   *
   * @param[in] state  State of the multibody system
   */
  explicit ResidualModelControlTpl(
      std::shared_ptr<typename Base::StateAbstract> state);
  virtual ~ResidualModelControlTpl();

  /**
   * @brief Compute the control residual
   *
   * @param[in] data  Control residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<ResidualDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief @copydoc Base::calc(const std::shared_ptr<ResidualDataAbstract>&
   * data, const Eigen::Ref<const VectorXs>& x)
   */
  virtual void calc(const std::shared_ptr<ResidualDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the derivatives of the control residual
   *
   * @param[in] data  Control residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Create the control residual data
   */
  virtual std::shared_ptr<ResidualDataAbstract> createData(
      DataCollectorAbstract* const data);

  /**
   * @brief Compute the derivative of the control-cost function
   *
   * This function assumes that the derivatives of the activation and residual
   * are computed via calcDiff functions.
   *
   * @param cdata     Cost data
   * @param rdata     Residual data
   * @param adata     Activation data
   * @param update_u  Update the derivative of the cost function w.r.t. to the
   * control if True.
   */
  virtual void calcCostDiff(
      const std::shared_ptr<CostDataAbstract>& cdata,
      const std::shared_ptr<ResidualDataAbstract>& rdata,
      const std::shared_ptr<ActivationDataAbstract>& adata,
      const bool update_u = true);

  /**
   * @brief Return the reference control vector
   */
  const VectorXs& get_reference() const;

  /**
   * @brief Modify the reference control vector
   */
  void set_reference(const VectorXs& reference);

  /**
   * @brief Print relevant information of the control residual
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 protected:
  using Base::nu_;
  using Base::state_;

 private:
  VectorXs uref_;  //!< Reference control input
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/residuals/control.hxx"

#endif  // CROCODDYL_CORE_RESIDUALS_CONTROL_HPP_
