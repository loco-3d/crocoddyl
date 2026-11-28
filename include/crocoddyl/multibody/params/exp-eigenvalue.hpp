///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_PARAMS_EXP_EIGENVALUE_HPP_
#define CROCODDYL_MULTIBODY_PARAMS_EXP_EIGENVALUE_HPP_

#include <cmath>
#include <pinocchio/spatial/explog.hpp>

#include "crocoddyl/multibody/params/inertial-parametrization-base.hpp"

namespace crocoddyl {

/**
 * @brief Reusable matrix workspace for the exponential-eigenvalue mapping
 *
 * It stores the principal-axis rotation, positive principal moments and
 * translated inertia terms used by conversion and analytical derivatives.
 */
template <typename _Scalar>
struct ExpEigenValueParametrizationDataTpl
    : public InertialParametrizationDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef Eigen::Matrix<Scalar, 3, 3> Matrix3s;
  typedef Eigen::Matrix<Scalar, 3, 1> Vector3s;

  /** @brief Initialize rotations to identity and all other storage to zero. */
  ExpEigenValueParametrizationDataTpl();
  virtual ~ExpEigenValueParametrizationDataTpl() = default;

  Matrix3s R;   //!< Principal-axis rotation.
  Matrix3s RD;  //!< Rotation times diagonal inertia.
  Matrix3s I;   //!< Spatial inertia matrix about the body origin.
  Matrix3s Sh;  //!< Skew matrix of the first moment.
  Vector3s D;   //!< Principal inertia values.
  Scalar Lx;    //!< Positive first pseudo-inertia eigenvalue.
  Scalar Ly;    //!< Positive second pseudo-inertia eigenvalue.
  Scalar Lz;    //!< Positive third pseudo-inertia eigenvalue.
};

/**
 * @brief Physically consistent exponential-eigenvalue parametrization
 *
 * The smooth vector contains log mass, first moment, a rotation vector and
 * three log principal pseudo-inertia eigenvalues. Exponentiation guarantees
 * positive mass and principal values; the rotation uses Pinocchio's SO(3)
 * exponential and Jacobian.
 */
template <typename _Scalar>
class ExpEigenValueParametrizationTpl
    : public InertialParametrizationAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef InertialParametrizationAbstractTpl<Scalar> Base;
  typedef InertialParametrizationDataAbstractTpl<Scalar>
      InertialParametrizationDataAbstract;
  typedef ExpEigenValueParametrizationDataTpl<Scalar>
      ExpEigenValueParametrizationData;
  typedef typename Base::VectorXs VectorXs;
  typedef typename Base::MatrixXs MatrixXs;
  typedef Eigen::Matrix<Scalar, 3, 3> Matrix3s;
  typedef Eigen::Matrix<Scalar, 3, 1> Vector3s;

  CROCODDYL_DERIVED_CAST(InertialParametrizationBase,
                         ExpEigenValueParametrizationTpl)

  /** @brief Initialize the stateless exponential-eigenvalue mapping. */
  ExpEigenValueParametrizationTpl() = default;
  virtual ~ExpEigenValueParametrizationTpl() = default;

  /** @copydoc InertialParametrizationAbstractTpl::fromParametrization */
  virtual void fromParametrization(
      const std::shared_ptr<InertialParametrizationDataAbstract>& data,
      Eigen::Ref<VectorXs> psi, const Eigen::Ref<const VectorXs>& p) override;

  /** @copydoc InertialParametrizationAbstractTpl::toParametrization */
  virtual void toParametrization(
      Eigen::Ref<VectorXs> p, const Eigen::Ref<const VectorXs>& psi) override;

  /** @copydoc
   * InertialParametrizationAbstractTpl::updateParametrizationDerivative */
  virtual void updateParametrizationDerivative(
      const std::shared_ptr<InertialParametrizationDataAbstract>& data,
      Eigen::Ref<MatrixXs> dpsi_dp, const Eigen::Ref<const VectorXs>& p,
      const Eigen::Ref<const VectorXs>& psi) override;

  /** @brief Allocate specialized exponential-eigenvalue workspace. */
  virtual std::shared_ptr<InertialParametrizationDataAbstract> createData()
      override;

  /** @brief Check for specialized exponential-eigenvalue data. */
  virtual bool checkData(
      const std::shared_ptr<InertialParametrizationDataAbstract>& data)
      const override;

  /** @brief Cast the stateless mapping to another scalar type. */
  template <typename NewScalar>
  ExpEigenValueParametrizationTpl<NewScalar> cast() const;

  /** @brief Print the parametrization name. */
  virtual void print(std::ostream& os) const override;

 protected:
  std::shared_ptr<ExpEigenValueParametrizationData> castData(
      const std::shared_ptr<InertialParametrizationDataAbstract>& data) const;
};

}  // namespace crocoddyl

#include "crocoddyl/multibody/params/exp-eigenvalue.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::ExpEigenValueParametrizationTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::ExpEigenValueParametrizationDataTpl)

#endif  // CROCODDYL_MULTIBODY_PARAMS_EXP_EIGENVALUE_HPP_
