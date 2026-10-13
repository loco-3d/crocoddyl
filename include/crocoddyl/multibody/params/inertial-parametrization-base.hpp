///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_PARAMS_INERTIAL_PARAMETRIZATION_BASE_HPP_
#define CROCODDYL_MULTIBODY_PARAMS_INERTIAL_PARAMETRIZATION_BASE_HPP_

#include "crocoddyl/multibody/fwd.hpp"

namespace crocoddyl {

/** @brief Type-erased base used by scalar clone/cast dispatch. */
class InertialParametrizationBase {
 public:
  virtual ~InertialParametrizationBase() = default;

  CROCODDYL_BASE_CAST(InertialParametrizationBase,
                      InertialParametrizationAbstractTpl)
};

/**
 * @brief Caller-owned workspace for an inertial parametrization
 *
 * Derived parametrizations extend this payload with reusable intermediate
 * quantities. Model calls require data created by the same concrete family.
 */
template <typename _Scalar>
struct InertialParametrizationDataAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;

  InertialParametrizationDataAbstractTpl() = default;
  virtual ~InertialParametrizationDataAbstractTpl() = default;
};

/**
 * @brief Abstract physically consistent inertial parametrization
 *
 * The model maps a ten-dimensional smooth vector \f$p\f$ to Pinocchio's
 * dynamic parameters
 * \f$\psi=(m,h_x,h_y,h_z,I_{xx},I_{xy},I_{yy},I_{xz},I_{yz},I_{zz})\f$,
 * and computes \f$\partial\psi/\partial p\f$. Output vectors and matrices are
 * caller-owned and must have dimension 10 and 10-by-10, respectively.
 */
template <typename _Scalar>
class InertialParametrizationAbstractTpl : public InertialParametrizationBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef InertialParametrizationDataAbstractTpl<Scalar>
      InertialParametrizationDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  static const std::size_t kDimension = 10;

  /** @brief Initialize a ten-parameter inertial parametrization. */
  InertialParametrizationAbstractTpl() = default;
  virtual ~InertialParametrizationAbstractTpl() = default;

  /** @brief Convert smooth parameters \f$p\f$ to dynamic parameters \f$\psi\f$.
   */
  virtual void fromParametrization(
      const std::shared_ptr<InertialParametrizationDataAbstract>& data,
      Eigen::Ref<VectorXs> psi, const Eigen::Ref<const VectorXs>& p) = 0;

  /** @brief Convert physical dynamic parameters \f$\psi\f$ to smooth
   * parameters.
   */
  virtual void toParametrization(Eigen::Ref<VectorXs> p,
                                 const Eigen::Ref<const VectorXs>& psi) = 0;

  /** @brief Compute the 10-by-10 Jacobian \f$\partial\psi/\partial p\f$. */
  virtual void updateParametrizationDerivative(
      const std::shared_ptr<InertialParametrizationDataAbstract>& data,
      Eigen::Ref<MatrixXs> dpsi_dp, const Eigen::Ref<const VectorXs>& p,
      const Eigen::Ref<const VectorXs>& psi) = 0;

  /** @brief Allocate caller-owned workspace for this parametrization. */
  virtual std::shared_ptr<InertialParametrizationDataAbstract> createData();

  /** @brief Check whether a data object is compatible with this model. */
  virtual bool checkData(
      const std::shared_ptr<InertialParametrizationDataAbstract>& data) const;

  /** @brief Return the fixed parameter dimension (10). */
  std::size_t get_np() const;

  /** @brief Print the concrete parametrization type. */
  virtual void print(std::ostream& os) const;

  /** @brief Clone this model using double precision. */
  virtual std::shared_ptr<InertialParametrizationBase> cloneAsDouble()
      const override;
  /** @brief Clone this model using single precision. */
  virtual std::shared_ptr<InertialParametrizationBase> cloneAsFloat()
      const override;
#ifdef CROCODDYL_WITH_CODEGEN
  virtual std::shared_ptr<InertialParametrizationBase> cloneAsADDouble()
      const override;
  virtual std::shared_ptr<InertialParametrizationBase> cloneAsADFloat()
      const override;
#endif

  template <class Scalar>
  friend std::ostream& operator<<(
      std::ostream& os, const InertialParametrizationAbstractTpl<Scalar>& obj);
};

}  // namespace crocoddyl

#include "crocoddyl/multibody/params/inertial-parametrization-base.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::InertialParametrizationDataAbstractTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::InertialParametrizationAbstractTpl)

#endif  // CROCODDYL_MULTIBODY_PARAMS_INERTIAL_PARAMETRIZATION_BASE_HPP_
