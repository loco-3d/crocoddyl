///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_RESIDUALS_PARAMETERS_HPP_
#define CROCODDYL_CORE_RESIDUALS_PARAMETERS_HPP_

#include "crocoddyl/core/data/params.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/residual-base.hpp"

namespace crocoddyl {

/**
 * @brief Parameter regularization residual
 *
 * This residual function is defined as \f$\mathbf{r} = \mathbf{p} -
 * \mathbf{p}^*\f$, where \f$\mathbf{p}\f$ is the current parameter vector
 * (read from shared data) and \f$\mathbf{p}^*\f$ is the reference. The
 * residual is purely parameter-dependent: \f$\mathbf{R_x} = 0\f$,
 * \f$\mathbf{R_u} = 0\f$, \f$\mathbf{R_p} = \mathbf{I}\f$.
 *
 * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ResidualModelParametersTpl : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ResidualModelBase, ResidualModelParametersTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef DataCollectorParamsTpl<Scalar> DataCollectorParams;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize the parameter residual model
   *
   * @param[in] state  State of the multibody system
   * @param[in] pref   Reference parameter vector
   * @param[in] nu     Dimension of the control vector
   */
  ResidualModelParametersTpl(
      std::shared_ptr<typename Base::StateAbstract> state, const VectorXs& pref,
      const std::size_t nu);

  virtual ~ResidualModelParametersTpl() = default;

  /**
   * @brief Compute the parameter residual \f$\mathbf{r} = \mathbf{p} -
   * \mathbf{p}^*\f$
   *
   * @param[in] data  Residual data
   * @param[in] x     State point (unused)
   * @param[in] u     Control input (unused)
   */
  virtual void calc(const std::shared_ptr<ResidualDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief @copydoc Base::calc(const std::shared_ptr<ResidualDataAbstract>&,
   * const Eigen::Ref<const VectorXs>&)
   *
   * The terminal overload evaluates the same parameter residual without a
   * control argument.
   */
  virtual void calc(const std::shared_ptr<ResidualDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Compute the Jacobians of the parameter residual
   *
   * Rp = I is constant and set in createData; nothing to recompute.
   *
   * @param[in] data  Residual data
   * @param[in] x     State point (unused)
   * @param[in] u     Control input (unused)
   */
  virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Create the parameter residual data
   *
   * Sets Rp = I.
   */
  virtual std::shared_ptr<ResidualDataAbstract> createData(
      DataCollectorAbstract* const data) override;

  /**
   * @brief Cast the parameter residual model to a different scalar type.
   */
  template <typename NewScalar>
  ResidualModelParametersTpl<NewScalar> cast() const;

  /**
   * @brief Return the reference parameter vector
   */
  const VectorXs& get_reference() const;

  /**
   * @brief Modify the reference parameter vector
   */
  void set_reference(const VectorXs& reference);

  /**
   * @brief Print relevant information of the parameter residual
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override;

 protected:
  using Base::np_;
  using Base::nr_;
  using Base::nu_;
  using Base::state_;
  using Base::unone_;

 private:
  VectorXs pref_;  //!< Reference parameter vector
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/residuals/parameters.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ResidualModelParametersTpl)

#endif  // CROCODDYL_CORE_RESIDUALS_PARAMETERS_HPP_
