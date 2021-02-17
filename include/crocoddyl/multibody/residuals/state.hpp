///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_RESIDUALS_STATE_HPP_
#define CROCODDYL_MULTIBODY_RESIDUALS_STATE_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

/**
 * @brief State residual
 *
 * This residual function defines the state tracking as \f$\mathbf{r}=\mathbf{x}\ominus\mathbf{x}^*\f$, where
 * \f$\mathbf{x},\mathbf{x}^*\in~\mathcal{X}\f$ are the current and reference states, respectively, which belong to the
 * state manifold \f$\mathcal{X}\f$. Note that the dimension of the residual vector is obtained from
 * `StateAbstract::get_ndx()`. Furthermore, the Jacobians of the residual function are
 * computed analytically.
 *
 * As described in `ResidualModelAbstractTpl()`, the residual value and its derivatives are calculated by `calc` and
 * `calcDiff`, respectively.
 *
 * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ResidualModelStateTpl : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the state residual model
   *
   * @param[in] state       State of the multibody system
   * @param[in] xref        Reference state
   * @param[in] nu          Dimension of the control vector
   */
  ResidualModelStateTpl(boost::shared_ptr<typename Base::StateAbstract> state, const VectorXs& xref,
                        const std::size_t nu);

  /**
   * @brief Initialize the state residual model
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state       State of the multibody system
   * @param[in] xref        Reference state
   */
  ResidualModelStateTpl(boost::shared_ptr<typename Base::StateAbstract> state, const VectorXs& xref);

  /**
   * @brief Initialize the state residual model
   *
   * The default reference state is obtained from `StateAbstractTpl::zero()`.
   *
   * @param[in] state  State of the multibody system
   * @param[in] nu     Dimension of the control vector
   */
  ResidualModelStateTpl(boost::shared_ptr<typename Base::StateAbstract> state, const std::size_t nu);

  /**
   * @brief Initialize the state residual model
   *
   * The default state reference is obtained from `StateAbstractTpl::zero()`, and `nu` from
   * `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   */
  ResidualModelStateTpl(boost::shared_ptr<typename Base::StateAbstract> state);
  virtual ~ResidualModelStateTpl();

  /**
   * @brief Compute the state residual
   *
   * @param[in] data  State residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the Jacobians of the state residual
   *
   * @param[in] data  State residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Return the reference state
   */
  const VectorXs& get_reference() const;

  /**
   * @brief Modify the reference state
   */
  void set_reference(const VectorXs& reference);

 protected:
  using Base::nu_;
  using Base::state_;
  using Base::unone_;

 private:
  VectorXs xref_;  //!< Reference state
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/residuals/state.hxx"

#endif  // CROCODDYL_MULTIBODY_RESIDUALS_STATE_HPP_
