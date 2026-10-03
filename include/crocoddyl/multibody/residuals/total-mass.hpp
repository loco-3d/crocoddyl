///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_RESIDUALS_TOTAL_MASS_HPP_
#define CROCODDYL_MULTIBODY_RESIDUALS_TOTAL_MASS_HPP_

#include <pinocchio/algorithm/center-of-mass.hpp>
#include <string>

#include "crocoddyl/core/params/parameter-manager.hpp"
#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/core/utils/conversions.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/params/inertial.hpp"

namespace crocoddyl {

/**
 * @brief Total-mass residual
 *
 * This residual is defined as \f$r = m - m^*\f$, where \f$m\f$ is the total
 * mass of the complete Pinocchio model, including bodies that are not selected
 * for parameterization, and \f$m^*\f$ is the reference mass.
 *
 * Parameter derivatives are read from the named
 * \c MultibodyInertialParamsData entry in the shared \c ParameterDataManager
 * and therefore cover only its selected bodies.
 *
 * The residual Jacobian \f$\mathbf{R_p}\f$ (1 × np) is assembled by placing
 * the mass row \f$\partial m_i / \partial \mathbf{p}_{\text{local},i}\f$
 * (row 0 of
 * \f$\partial\boldsymbol{\psi}_i/\partial\mathbf{p}_{\text{local},i}\f$) at
 * columns \f$[\texttt{np\_offset} + 10 i,\; \texttt{np\_offset} + 10 i +
 * 10)\f$.
 *
 * @note The constructor parameter \c np must equal the total active parameter
 *       dimension of the \c ParameterDataManager (action + dynamics combined).
 *
 * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ResidualModelTotalMassTpl : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ResidualModelBase, ResidualModelTotalMassTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef ResidualDataTotalMassTpl<Scalar> ResidualDataTotalMass;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef DataCollectorParamsTpl<Scalar> DataCollectorParams;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef MultibodyInertialParamsDataTpl<Scalar> MultibodyInertialParamsData;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize the total-mass residual model
   *
   * @param[in] state       State of the multibody system
   * @param[in] mass_ref    Reference total mass
   * @param[in] nu          Dimension of the control vector
   * @param[in] np          Total active parameter dimension of the
   *                        ParameterDataManager (action + dynamics combined)
   * @param[in] param_name  Name of the MultibodyInertialParams entry in the
   *                        ParameterDataManager
   */
  ResidualModelTotalMassTpl(std::shared_ptr<typename Base::StateAbstract> state,
                            const Scalar mass_ref, const std::size_t nu,
                            const std::size_t np,
                            const std::string& param_name);

  virtual ~ResidualModelTotalMassTpl() = default;

  /**
   * @brief Compute the complete-model total-mass residual
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
   * @brief Compute the Jacobian of the total-mass residual
   *
   * Sets the mass-row block of \f$\mathbf{R_p}\f$ for each body at its
   * corresponding column offset.
   *
   * @param[in] data  Residual data
   * @param[in] x     State point (unused)
   * @param[in] u     Control input (unused)
   */
  virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Create the total-mass residual data
   *
   * Validates that the shared data is a \c ParameterDataManager containing the
   * named active \c MultibodyInertialParamsData entry and its D075 offset.
   */
  virtual std::shared_ptr<ResidualDataAbstract> createData(
      DataCollectorAbstract* const data) override;

  /**
   * @brief Cast the total-mass residual model to a different scalar type
   */
  template <typename NewScalar>
  ResidualModelTotalMassTpl<NewScalar> cast() const;

  /**
   * @brief Return the reference total mass
   */
  Scalar get_reference() const;

  /**
   * @brief Modify the reference total mass
   */
  void set_reference(const Scalar reference);

  /**
   * @brief Return the name of the MultibodyInertialParams entry
   */
  const std::string& get_param_name() const;

  /**
   * @brief Print relevant information of the total-mass residual
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
  Scalar mass_ref_;         //!< Reference total mass
  std::string param_name_;  //!< Name of the MultibodyInertialParams entry
  std::shared_ptr<typename StateMultibody::PinocchioModel> pin_model_;
};

/**
 * @brief Data for ResidualModelTotalMassTpl
 *
 * Extends ResidualDataAbstractTpl with the pre-computed column offset
 * np_offset that locates the named parameter's slice within the global
 * parameter vector.
 */
template <typename _Scalar>
struct ResidualDataTotalMassTpl : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ResidualDataAbstractTpl<Scalar> Base;

  template <template <typename S> class Model>
  ResidualDataTotalMassTpl(Model<Scalar>* const model,
                           DataCollectorAbstractTpl<Scalar>* const data)
      : Base(model, data), np_offset(0) {}

  virtual ~ResidualDataTotalMassTpl() = default;

  using Base::r;
  using Base::Rp;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;

  std::size_t np_offset;  //!< Column offset of the named param in global p
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/residuals/total-mass.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ResidualModelTotalMassTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ResidualDataTotalMassTpl)

#endif  // CROCODDYL_MULTIBODY_RESIDUALS_TOTAL_MASS_HPP_
