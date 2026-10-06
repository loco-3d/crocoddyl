///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_PARAMS_LOG_CHOLESKY_HPP_
#define CROCODDYL_MULTIBODY_PARAMS_LOG_CHOLESKY_HPP_

#include <cmath>

#include "crocoddyl/multibody/params/inertial-parametrization-base.hpp"

namespace crocoddyl {

/**
 * @brief Reusable scalar workspace for the log-Cholesky mapping
 *
 * The fields cache exponentials and products needed by conversion and its
 * analytical Jacobian. The data owns these values and can be copied safely.
 */
template <typename _Scalar>
struct LogCholeskyParametrizationDataTpl
    : public InertialParametrizationDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;

  /** @brief Initialize every cached scalar to zero. */
  LogCholeskyParametrizationDataTpl();
  virtual ~LogCholeskyParametrizationDataTpl() = default;

  Scalar alpha;  //!< Log mass scale.
  Scalar d1;     //!< First log diagonal entry.
  Scalar d2;     //!< Second log diagonal entry.
  Scalar d3;     //!< Third log diagonal entry.
  Scalar s1;     //!< First off-diagonal entry.
  Scalar s2;     //!< Second off-diagonal entry.
  Scalar s3;     //!< Third off-diagonal entry.
  Scalar t1;     //!< First first-moment entry.
  Scalar t2;     //!< Second first-moment entry.
  Scalar t3;     //!< Third first-moment entry.
  Scalar exp2alpha;
  Scalar exp2alpha2;
  Scalar expd1;
  Scalar expd2;
  Scalar expd3;
  Scalar exp2d1;
  Scalar exp2d2;
  Scalar exp2d3;
  Scalar s1pow2;
  Scalar s2pow2;
  Scalar s3pow2;
  Scalar t1pow2;
  Scalar t2pow2;
  Scalar t3pow2;
};

/**
 * @brief Physically consistent pseudo-inertia log-Cholesky parametrization
 *
 * The smooth vector is \f$(\alpha,d_1,d_2,d_3,s_1,s_2,s_3,t_1,t_2,t_3)\f$.
 * Exponentiated diagonal entries make the pseudo-inertia positive definite,
 * yielding positive mass and a physically valid spatial inertia.
 */
template <typename _Scalar>
class LogCholeskyParametrizationTpl
    : public InertialParametrizationAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef InertialParametrizationAbstractTpl<Scalar> Base;
  typedef InertialParametrizationDataAbstractTpl<Scalar>
      InertialParametrizationDataAbstract;
  typedef LogCholeskyParametrizationDataTpl<Scalar>
      LogCholeskyParametrizationData;
  typedef typename Base::VectorXs VectorXs;
  typedef typename Base::MatrixXs MatrixXs;

  CROCODDYL_DERIVED_CAST(InertialParametrizationBase,
                         LogCholeskyParametrizationTpl)

  /** @brief Initialize the stateless log-Cholesky mapping. */
  LogCholeskyParametrizationTpl() = default;
  virtual ~LogCholeskyParametrizationTpl() = default;

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

  /** @brief Allocate specialized log-Cholesky workspace. */
  virtual std::shared_ptr<InertialParametrizationDataAbstract> createData()
      override;

  /** @brief Check for specialized log-Cholesky data. */
  virtual bool checkData(
      const std::shared_ptr<InertialParametrizationDataAbstract>& data)
      const override;

  /** @brief Cast the stateless mapping to another scalar type. */
  template <typename NewScalar>
  LogCholeskyParametrizationTpl<NewScalar> cast() const;

  /** @brief Print the parametrization name. */
  virtual void print(std::ostream& os) const override;

 protected:
  std::shared_ptr<LogCholeskyParametrizationData> castData(
      const std::shared_ptr<InertialParametrizationDataAbstract>& data) const;
};

}  // namespace crocoddyl

#include "crocoddyl/multibody/params/log-cholesky.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::LogCholeskyParametrizationTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::LogCholeskyParametrizationDataTpl)

#endif  // CROCODDYL_MULTIBODY_PARAMS_LOG_CHOLESKY_HPP_
