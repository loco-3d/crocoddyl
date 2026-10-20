///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_RESIDUALS_ACTUATION_PARAMETERS_HPP_
#define CROCODDYL_MULTIBODY_RESIDUALS_ACTUATION_PARAMETERS_HPP_

#include <string>

#include "crocoddyl/core/params/parameter-manager.hpp"
#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/params/actuation.hpp"

namespace crocoddyl {

/**
 * @brief Actuation-parameter residual
 *
 * This residual is defined as \f$\mathbf{r} = \boldsymbol{\gamma} -
 * \boldsymbol{\gamma}^*\f$, where \f$\boldsymbol{\gamma}\f$ is the decoded
 * actuation parameter vector obtained by applying the actuation parametrization
 * to the current decision variables, and \f$\boldsymbol{\gamma}^*\f$ is the
 * reference.
 *
 * The residual reads the decoded parameters \f$\boldsymbol{\gamma}\f$ and
 * their Jacobian \f$\partial\boldsymbol{\gamma}/\partial\mathbf{p}\f$ directly
 * from a named \c ActuationMultibodyParamsData entry in the shared
 * \c ParameterDataManager.
 *
 * The residual Jacobian \f$\mathbf{R_p}\f$ is a sparse \f$nr \times np\f$
 * matrix: the \f$nr \times nr\f$ block
 * \f$\partial\boldsymbol{\gamma}/\partial\mathbf{p}_\text{actuation}\f$ is
 * placed at the column offset corresponding to the named active parameter's
 * slice within the global parameter vector. The offset follows the manager's
 * action-then-dynamics, lexicographic active layout.
 *
 * @note The constructor parameter \c np must equal the total active parameter
 *       dimension of the \c ParameterDataManager (action + dynamics combined).
 *
 * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ResidualModelActuationParametersTpl
    : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ResidualModelBase, ResidualModelActuationParametersTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef ResidualDataActuationParametersTpl<Scalar>
      ResidualDataActuationParameters;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef ActuationMultibodyParamsDataTpl<Scalar> ActuationMultibodyParamsData;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize the actuation-parameter residual model
   *
   * @param[in] state       State of the multibody system
   * @param[in] gamma_ref   Reference actuation-parameter vector
   * @param[in] nu          Dimension of the control vector
   * @param[in] np          Total active parameter dimension of the
   *                        ParameterDataManager (action + dynamics combined)
   * @param[in] param_name  Name of the ActuationMultibodyParams entry in the
   *                        ParameterDataManager
   */
  ResidualModelActuationParametersTpl(
      std::shared_ptr<typename Base::StateAbstract> state,
      const VectorXs& gamma_ref, const std::size_t nu, const std::size_t np,
      const std::string& param_name);

  virtual ~ResidualModelActuationParametersTpl() = default;

  /**
   * @brief Compute \f$\mathbf{r} = \boldsymbol{\gamma} -
   * \boldsymbol{\gamma}^*\f$
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
   * The terminal overload evaluates the same parameter residual.
   */
  virtual void calc(const std::shared_ptr<ResidualDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Compute the Jacobians of the actuation-parameter residual
   *
   * Sets the \f$nr \times nr\f$ block of \f$\mathbf{R_p}\f$ at column
   * \c np_offset to
   * \f$\partial\boldsymbol{\gamma}/\partial\mathbf{p}_\text{actuation}\f$.
   *
   * @param[in] data  Residual data
   * @param[in] x     State point (unused)
   * @param[in] u     Control input (unused)
   */
  virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Create the actuation-parameter residual data
   *
   * Validates that the shared data is a \c ParameterDataManager containing the
   * named \c ActuationMultibodyParamsData entry, and pre-computes the column
   * offset into the global parameter vector. Inactive references are rejected.
   */
  virtual std::shared_ptr<ResidualDataAbstract> createData(
      DataCollectorAbstract* const data) override;

  /**
   * @brief Cast the actuation-parameter residual model to a different scalar
   * type
   */
  template <typename NewScalar>
  ResidualModelActuationParametersTpl<NewScalar> cast() const;

  /**
   * @brief Return the reference actuation-parameter vector
   */
  const VectorXs& get_reference() const;

  /**
   * @brief Modify the reference actuation-parameter vector
   */
  void set_reference(const VectorXs& reference);

  /**
   * @brief Return the name of the ActuationMultibodyParams entry
   */
  const std::string& get_param_name() const;

  /**
   * @brief Print relevant information of the actuation-parameter residual
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
  VectorXs gamma_ref_;      //!< Reference actuation-parameter vector
  std::string param_name_;  //!< Name of the ActuationMultibodyParams entry
};

/**
 * @brief Data for ResidualModelActuationParametersTpl
 *
 * Extends ResidualDataAbstractTpl with the current column offset
 * np_offset that locates the named parameter's slice within the global
 * parameter vector.
 */
template <typename _Scalar>
struct ResidualDataActuationParametersTpl
    : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ResidualDataAbstractTpl<Scalar> Base;

  template <template <typename S> class Model>
  ResidualDataActuationParametersTpl(
      Model<Scalar>* const model, DataCollectorAbstractTpl<Scalar>* const data)
      : Base(model, data), np_offset(0) {}

  virtual ~ResidualDataActuationParametersTpl() = default;

  using Base::r;
  using Base::Rp;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;

  std::size_t np_offset;  //!< Column offset of the named param in global p
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/residuals/actuation-parameters.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::ResidualModelActuationParametersTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::ResidualDataActuationParametersTpl)

#endif  // CROCODDYL_MULTIBODY_RESIDUALS_ACTUATION_PARAMETERS_HPP_
