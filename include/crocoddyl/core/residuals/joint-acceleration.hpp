///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2022-2025, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_RESIDUALS_JOINT_ACCELERATION_HPP_
#define CROCODDYL_CORE_RESIDUALS_JOINT_ACCELERATION_HPP_

#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/data/joint.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/residual-base.hpp"

namespace crocoddyl {

/**
 * @brief Define a joint-acceleration residual
 *
 * This residual function is defined as
 * \f$\mathbf{r}=\mathbf{u}-\mathbf{u}^*\f$, where
 * \f$\mathbf{u},\mathbf{u}^*\in~\mathbb{R}^{nu}\f$ are the current and
 * reference joint acceleration, respectively. Note that the dimension of the
 * residual vector is obtained from `StateAbstract::nv`, as it represents the
 * generalized acceleration.
 *
 * Both residual and residual Jacobians are computed analytically.
 *
 * As described in ResidualModelAbstractTpl(), the residual value and its
 * Jacobians are calculated by `calc` and `calcDiff`, respectively.
 *
 * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ResidualModelJointAccelerationTpl
    : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ResidualModelBase, ResidualModelJointAccelerationTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataJointAccelerationTpl<Scalar> Data;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef StateAbstractTpl<Scalar> StateAbstract;
  typedef ActuationModelAbstractTpl<Scalar> ActuationModelAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the joint-acceleration residual model
   *
   * @param[in] state       State description
   * @param[in] aref        Reference joint acceleration
   * @param[in] nu          Dimension of the control vector
   */
  ResidualModelJointAccelerationTpl(std::shared_ptr<StateAbstract> state,
                                    const VectorXs& aref, const std::size_t nu);

  /**
   * @brief Initialize the joint-acceleration residual model
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state       State description
   * @param[in] aref        Reference joint acceleration
   */
  ResidualModelJointAccelerationTpl(std::shared_ptr<StateAbstract> state,
                                    const VectorXs& aref);

  /**
   * @brief Initialize the joint-acceleration residual model
   *
   * The default reference joint acceleration is obtained from
   * `MathBaseTpl<>::VectorXs::Zero(state->get_nv())`.
   *
   * @param[in] state       State description
   * @param[in] nu          Dimension of the control vector
   */
  ResidualModelJointAccelerationTpl(std::shared_ptr<StateAbstract> state,
                                    const std::size_t nu);

  /**
   * @brief Initialize the joint-acceleration residual model
   *
   * The default reference joint acceleration is obtained from
   * `MathBaseTpl<>::VectorXs::Zero(state->get_nv())`. The default `nu` value is
   * obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state       State description
   */
  ResidualModelJointAccelerationTpl(std::shared_ptr<StateAbstract> state);

  virtual ~ResidualModelJointAccelerationTpl() = default;

  /**
   * @brief Compute the joint-acceleration residual
   *
   * @param[in] data  Joint-acceleration residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<ResidualDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief @copydoc Base::calc(const std::shared_ptr<ResidualDataAbstract>&
   * data, const Eigen::Ref<const VectorXs>& x)
   */
  virtual void calc(const std::shared_ptr<ResidualDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Compute the derivatives of the joint-acceleration residual
   *
   * @param[in] data  Joint-acceleration residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Create the joint-acceleration residual data
   */
  virtual std::shared_ptr<ResidualDataAbstract> createData(
      DataCollectorAbstract* const data) override;

  /**
   * @brief Cast the joint-acceleration residual model to a different scalar
   * type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return ResidualModelJointAccelerationTpl<NewScalar> A residual model with
   * the new scalar type.
   */
  template <typename NewScalar>
  ResidualModelJointAccelerationTpl<NewScalar> cast() const;

  /**
   * @brief Return the reference joint-acceleration vector
   */
  const VectorXs& get_reference() const;

  /**
   * @brief Modify the reference joint-acceleration vector
   */
  void set_reference(const VectorXs& reference);

  /**
   * @brief Print relevant information of the joint-acceleration residual
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override;

 protected:
  using Base::nr_;
  using Base::nu_;
  using Base::state_;

 private:
  VectorXs aref_;  //!< Reference joint-acceleration input
};

template <typename _Scalar>
struct ResidualDataJointAccelerationTpl
    : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;

  template <template <typename Scalar> class Model>
  ResidualDataJointAccelerationTpl(Model<Scalar>* const model,
                                   DataCollectorAbstract* const data)
      : Base(model, data) {
    // Check that proper shared data has been passed
    DataCollectorJointTpl<Scalar>* d =
        dynamic_cast<DataCollectorJointTpl<Scalar>*>(shared);
    if (d == nullptr) {
      throw_pretty(
          "Invalid argument: the shared data should be derived from "
          "DataCollectorJoint");
    }
    joint = d->joint;
  }
  virtual ~ResidualDataJointAccelerationTpl() = default;

  std::shared_ptr<JointDataAbstractTpl<Scalar> > joint;  //!< Joint data
  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/residuals/joint-acceleration.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::ResidualModelJointAccelerationTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::ResidualDataJointAccelerationTpl)

#endif  // CROCODDYL_CORE_RESIDUALS_JOINT_ACCELERATION_HPP_
