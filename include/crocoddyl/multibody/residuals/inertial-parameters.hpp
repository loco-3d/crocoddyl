///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_RESIDUALS_INERTIAL_PARAMETERS_HPP_
#define CROCODDYL_MULTIBODY_RESIDUALS_INERTIAL_PARAMETERS_HPP_

#include <string>

#include "crocoddyl/core/params/parameter-manager.hpp"
#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/params/inertial.hpp"

namespace crocoddyl {

/**
 * @brief Inertial-parameter residual
 *
 * This residual is defined as \f$\mathbf{r} = \boldsymbol{\psi} -
 * \boldsymbol{\psi}^*\f$, where \f$\boldsymbol{\psi}\f$ is the stacked vector
 * of decoded inertial parameters (mass, first moments, and inertia tensor
 * components) for all tracked bodies, obtained by applying the parametrization
 * to the current decision variables, and \f$\boldsymbol{\psi}^*\f$ is the
 * reference.
 *
 * The residual reads the per-body decoded parameters \f$\boldsymbol{\psi}_i\f$
 * and their Jacobians
 * \f$\partial\boldsymbol{\psi}_i/\partial\mathbf{p}\f$ directly from a named
 * \c MultibodyInertialParamsData entry in the shared \c ParameterDataManager.
 *
 * The residual Jacobian \f$\mathbf{R_p}\f$ contains the block-diagonal
 * parametrization Jacobian at the D075 active offset of the named parameter.
 *
 * @note This residual requires the shared data to be a
 *       \c ParameterDataManagerTpl and the named entry to be a
 *       \c MultibodyInertialParamsDataTpl.  The parameter dimension \c np is
 *       derived from \c psi_ref and must equal 10 * nbodies. Inactive named
 *       parameters are rejected.
 *
 * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ResidualModelInertialParametersTpl
    : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ResidualModelBase, ResidualModelInertialParametersTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef ResidualDataInertialParametersTpl<Scalar>
      ResidualDataInertialParameters;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef MultibodyInertialParamsDataTpl<Scalar> MultibodyInertialParamsData;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize the inertial-parameter residual model
   *
   * @param[in] state       State of the multibody system
   * @param[in] psi_ref     Reference inertial-parameter vector of size
   *                        10 * nbodies (m, h, I per body in stacked order)
   * @param[in] nu          Dimension of the control vector
   * @param[in] param_name  Name of the MultibodyInertialParams entry in the
   *                        ParameterDataManager
   */
  ResidualModelInertialParametersTpl(
      std::shared_ptr<typename Base::StateAbstract> state,
      const VectorXs& psi_ref, const std::size_t nu,
      const std::string& param_name);

  /**
   * @brief Initialize the inertial-parameter residual model
   *
   * @param[in] state       State of the multibody system
   * @param[in] psi_ref     Reference inertial-parameter vector of size
   *                        10 * nbodies (m, h, I per body in stacked order)
   * @param[in] nu          Dimension of the control vector
   * @param[in] np          Dimension of the full parameter vector
   * @param[in] param_name  Name of the MultibodyInertialParams entry in the
   *                        ParameterDataManager
   */
  ResidualModelInertialParametersTpl(
      std::shared_ptr<typename Base::StateAbstract> state,
      const VectorXs& psi_ref, const std::size_t nu, const std::size_t np,
      const std::string& param_name)
      : Base(state, static_cast<std::size_t>(psi_ref.size()), nu,
             /*q_dependent=*/false, /*v_dependent=*/false,
             /*u_dependent=*/false, np),
        psi_ref_(psi_ref),
        param_name_(param_name) {}

  virtual ~ResidualModelInertialParametersTpl() = default;

  /**
   * @brief Compute \f$\mathbf{r} = \boldsymbol{\psi} - \boldsymbol{\psi}^*\f$
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
   * @brief Compute the Jacobians of the inertial-parameter residual
   *
   * Sets \f$\mathbf{R_p}\f$ to the block-diagonal matrix of per-body
   * parametrization Jacobians
   * \f$\partial\boldsymbol{\psi}_i/\partial\mathbf{p}_i\f$.
   *
   * @param[in] data  Residual data
   * @param[in] x     State point (unused)
   * @param[in] u     Control input (unused)
   */
  virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Create the inertial-parameter residual data
   *
   * Validates that the shared data is a \c ParameterDataManager containing the
   * named \c MultibodyInertialParamsData entry with matching dimension.
   */
  virtual std::shared_ptr<ResidualDataAbstract> createData(
      DataCollectorAbstract* const data) override;

  /**
   * @brief Cast the inertial-parameter residual model to a different scalar
   * type
   */
  template <typename NewScalar>
  ResidualModelInertialParametersTpl<NewScalar> cast() const;

  /**
   * @brief Return the reference inertial-parameter vector
   */
  const VectorXs& get_reference() const;

  /**
   * @brief Modify the reference inertial-parameter vector
   */
  void set_reference(const VectorXs& reference);

  /**
   * @brief Return the name of the MultibodyInertialParams entry
   */
  const std::string& get_param_name() const;

  /**
   * @brief Print relevant information of the inertial-parameter residual
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
  VectorXs psi_ref_;        //!< Reference inertial-parameter vector
  std::string param_name_;  //!< Name of the MultibodyInertialParams entry
};

/**
 * @brief Data for ResidualModelInertialParametersTpl
 *
 * The data retains the current D075 active offset of the named parameter. The
 * shared collector remains non-owning through ResidualDataAbstractTpl.
 */
template <typename _Scalar>
struct ResidualDataInertialParametersTpl
    : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ResidualDataAbstractTpl<Scalar> Base;

  template <template <typename S> class Model>
  ResidualDataInertialParametersTpl(
      Model<Scalar>* const model, DataCollectorAbstractTpl<Scalar>* const data)
      : Base(model, data), np_offset(0) {}

  virtual ~ResidualDataInertialParametersTpl() = default;

  std::size_t np_offset;  //!< D075 active offset in the global parameter vector
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/residuals/inertial-parameters.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::ResidualModelInertialParametersTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::ResidualDataInertialParametersTpl)

#endif  // CROCODDYL_MULTIBODY_RESIDUALS_INERTIAL_PARAMETERS_HPP_
