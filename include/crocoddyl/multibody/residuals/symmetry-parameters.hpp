///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_RESIDUALS_SYMMETRY_PARAMETERS_HPP_
#define CROCODDYL_MULTIBODY_RESIDUALS_SYMMETRY_PARAMETERS_HPP_

#include <string>

#include "crocoddyl/core/params/parameter-manager.hpp"
#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/params/actuation.hpp"
#include "crocoddyl/multibody/params/inertial.hpp"

namespace crocoddyl {

/**
 * @brief Parameter symmetry residual
 *
 * This residual encodes a symmetry constraint of the form
 * \f$\mathbf{r} = \mathbf{S}\,\boldsymbol{\theta}\f$, where \f$\mathbf{S}\f$
 * is a user-supplied square symmetry matrix. In generic mode,
 * \f$\boldsymbol{\theta}=\mathbf{p}\f$ is the complete active parameter
 * vector. In named mode, \f$\boldsymbol{\theta}\f$ is the decoded parameter
 * vector read from a named entry in the shared \c ParameterDataManager.
 *
 * The residual automatically detects the type of the named dynamics parameter:
 *
 * - **\c ActuationMultibodyParamsData**: \f$\boldsymbol{\theta} =
 *   \boldsymbol{\gamma}\f$ (decoded actuation parameters).
 * - **\c MultibodyInertialParamsData**: \f$\boldsymbol{\theta} =
 *   [\boldsymbol{\psi}_0^\top, \ldots, \boldsymbol{\psi}_{N-1}^\top]^\top\f$
 *   (stacked decoded inertial parameters for all bodies).
 *
 * If the named entry is neither of these types, \c createData throws. Generic
 * mode requires \f$\mathbf{S}\f$ to have side length \c np and has the
 * constant Jacobian \f$\mathbf{R_p}=\mathbf{S}\f$.
 *
 * The Jacobian \f$\mathbf{R_p}\f$ is an \f$nr \times np\f$ sparse matrix:
 * the \f$nr \times nr\f$ block \f$\mathbf{S}\,\partial\boldsymbol{\theta}/
 * \partial\mathbf{p}_\text{local}\f$ is placed at the column offset
 * corresponding to the named parameter's D075 active slice within the global
 * parameter vector. Inactive named parameters are rejected.
 *
 * @note \c S must be square with side length equal to the local parameter
 *       dimension of the named entry. In generic mode, \c np must equal the
 *       aggregate dimension of \c DataCollectorParams. In named mode it must
 *       equal the total active dimension of \c ParameterDataManager.
 *
 * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ResidualModelSymmetryParametersTpl
    : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ResidualModelBase, ResidualModelSymmetryParametersTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef ResidualDataSymmetryParametersTpl<Scalar>
      ResidualDataSymmetryParameters;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef DataCollectorParamsTpl<Scalar> DataCollectorParams;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef ActuationMultibodyParamsDataTpl<Scalar> ActuationMultibodyParamsData;
  typedef MultibodyInertialParamsDataTpl<Scalar> MultibodyInertialParamsData;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize a generic symmetry residual over the active parameter
   * vector
   *
   * @param[in] state  State of the system
   * @param[in] S      Square symmetry matrix of size np x np
   * @param[in] nu     Dimension of the control vector
   * @param[in] np     Aggregate parameter dimension
   */
  ResidualModelSymmetryParametersTpl(
      std::shared_ptr<typename Base::StateAbstract> state, const MatrixXs& S,
      const std::size_t nu, const std::size_t np);

  /**
   * @brief Initialize the parameter symmetry residual model
   *
   * @param[in] state       State of the multibody system
   * @param[in] S           Square symmetry matrix (nr x nr)
   * @param[in] nu          Dimension of the control vector
   * @param[in] np          Total active parameter dimension of the
   *                        ParameterDataManager (action + dynamics combined)
   * @param[in] param_name  Name of the dynamics parameter entry in the
   *                        ParameterDataManager (must be either
   *                        ActuationMultibodyParams or MultibodyInertialParams)
   */
  ResidualModelSymmetryParametersTpl(
      std::shared_ptr<typename Base::StateAbstract> state, const MatrixXs& S,
      const std::size_t nu, const std::size_t np,
      const std::string& param_name);

  virtual ~ResidualModelSymmetryParametersTpl() = default;

  /**
   * @brief Compute \f$\mathbf{r} = \mathbf{S}\,\boldsymbol{\theta}\f$
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
   * @brief Compute the Jacobians of the parameter symmetry residual
   *
   * Generic mode has the constant \f$\mathbf{R_p}=\mathbf{S}\f$ initialized
   * by createData() and this method does nothing. Named mode sets the
   * \f$nr \times nr\f$ block at the current D075 \c np_offset to
   * \f$\mathbf{S}\,\partial\boldsymbol{\theta}/\partial\mathbf{p}_\text{local}
   * \f$.
   *
   * @param[in] data  Residual data
   * @param[in] x     State point (unused)
   * @param[in] u     Control input (unused)
   */
  virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Create the parameter symmetry residual data
   *
   * Generic mode requires only a valid \c DataCollectorParams aggregate and
   * initializes the constant \f$\mathbf{R_p}=\mathbf{S}\f$. Named mode
   * additionally checks the manager, active decoded entry and D075 offset.
   */
  virtual std::shared_ptr<ResidualDataAbstract> createData(
      DataCollectorAbstract* const data) override;

  /**
   * @brief Cast the parameter symmetry residual model to a different scalar
   * type
   */
  template <typename NewScalar>
  ResidualModelSymmetryParametersTpl<NewScalar> cast() const;

  /**
   * @brief Return the symmetry matrix S
   */
  const MatrixXs& get_S() const;

  /**
   * @brief Modify the symmetry matrix S
   *
   * The new matrix must have the same shape as the current one.
   */
  void set_S(const MatrixXs& S);

  /**
   * @brief Return the name of the dynamics parameter entry
   */
  const std::string& get_param_name() const;

  /**
   * @brief Print relevant information of the parameter symmetry residual
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
  MatrixXs S_;              //!< Symmetry matrix
  std::string param_name_;  //!< Name of the dynamics parameter entry
};

/**
 * @brief Data for ResidualModelSymmetryParametersTpl
 *
 * Stores the D075 column offset used by named mode and non-owning pointers to
 * the detected parameter data objects.
 */
template <typename _Scalar>
struct ResidualDataSymmetryParametersTpl
    : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef ActuationMultibodyParamsDataTpl<Scalar> ActuationMultibodyParamsData;
  typedef MultibodyInertialParamsDataTpl<Scalar> MultibodyInertialParamsData;

  template <template <typename S> class Model>
  ResidualDataSymmetryParametersTpl(
      Model<Scalar>* const model, DataCollectorAbstractTpl<Scalar>* const data)
      : Base(model, data),
        np_offset(0),
        actuation_data(nullptr),
        inertial_data(nullptr) {}

  virtual ~ResidualDataSymmetryParametersTpl() = default;

  using Base::r;
  using Base::Rp;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;

  std::size_t np_offset;  //!< Column offset of the named param in global p

  //! Non-owning pointer to actuation param data (null if inertial)
  ActuationMultibodyParamsData* actuation_data;
  //! Non-owning pointer to inertial param data (null if actuation)
  MultibodyInertialParamsData* inertial_data;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/residuals/symmetry-parameters.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::ResidualModelSymmetryParametersTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::ResidualDataSymmetryParametersTpl)

#endif  // CROCODDYL_MULTIBODY_RESIDUALS_SYMMETRY_PARAMETERS_HPP_
