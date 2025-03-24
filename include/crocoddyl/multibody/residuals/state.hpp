///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2022-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_RESIDUALS_STATE_HPP_
#define CROCODDYL_MULTIBODY_RESIDUALS_STATE_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

/**
 * @brief State residual
 *
 * This residual function defines the state tracking as
 * \f$\mathbf{r}=\mathbf{x}\ominus\mathbf{x}^*\f$, where
 * \f$\mathbf{x},\mathbf{x}^*\in~\mathcal{X}\f$ are the current and reference
 * states, respectively, which belong to the state manifold \f$\mathcal{X}\f$.
 * Note that the dimension of the residual vector is obtained from
 * `StateAbstract::get_ndx()`. Furthermore, the Jacobians of the residual
 * function are computed analytically.
 *
 * As described in `ResidualModelAbstractTpl()`, the residual value and its
 * derivatives are calculated by `calc` and `calcDiff`, respectively.
 *
 * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ResidualModelStateTpl : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ResidualModelBase, ResidualModelStateTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef ActivationDataAbstractTpl<Scalar> ActivationDataAbstract;
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
  ResidualModelStateTpl(std::shared_ptr<typename Base::StateAbstract> state,
                        const VectorXs& xref, const std::size_t nu);

  /**
   * @brief Initialize the state residual model
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state       State of the multibody system
   * @param[in] xref        Reference state
   */
  ResidualModelStateTpl(std::shared_ptr<typename Base::StateAbstract> state,
                        const VectorXs& xref);

  /**
   * @brief Initialize the state residual model
   *
   * The default reference state is obtained from `StateAbstractTpl::zero()`.
   *
   * @param[in] state  State of the multibody system
   * @param[in] nu     Dimension of the control vector
   */
  ResidualModelStateTpl(std::shared_ptr<typename Base::StateAbstract> state,
                        const std::size_t nu);

  /**
   * @brief Initialize the state residual model
   *
   * The default state reference is obtained from `StateAbstractTpl::zero()`,
   * and `nu` from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   */
  ResidualModelStateTpl(std::shared_ptr<typename Base::StateAbstract> state);
  virtual ~ResidualModelStateTpl() = default;

  /**
   * @brief Compute the state residual
   *
   * @param[in] data  State residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<ResidualDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Compute the Jacobians of the state residual
   *
   * @param[in] data  State residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Compute the derivative of the state-cost function and store it in
   * cost data
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
      const bool update_u = true) override;

  /**
   * @brief Cast the state residual model to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return ResidualModelStateTpl<NewScalar> A residual model with the
   * new scalar type.
   */
  template <typename NewScalar>
  ResidualModelStateTpl<NewScalar> cast() const;

  /**
   * @brief Return the reference state
   */
  const VectorXs& get_reference() const;

  /**
   * @brief Modify the reference state
   */
  void set_reference(const VectorXs& reference);

  /**
   * @brief Print relevant information of the state residual
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override;

 protected:
  using Base::nr_;
  using Base::nu_;
  using Base::state_;
  using Base::u_dependent_;

 private:
  VectorXs xref_;  //!< Reference state
  std::shared_ptr<typename StateMultibody::PinocchioModel>
      pin_model_;  //!< Pinocchio model
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/residuals/state.hxx"

extern template class CROCODDYL_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI
    crocoddyl::ResidualModelStateTpl<double>;
extern template class CROCODDYL_EXPLICIT_INSTANTIATION_DECLARATION_DLLAPI
    crocoddyl::ResidualModelStateTpl<float>;

#endif  // CROCODDYL_MULTIBODY_RESIDUALS_STATE_HPP_
