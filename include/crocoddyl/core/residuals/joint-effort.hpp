///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2022-2025, Heriot-Watt University, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_RESIDUALS_JOINT_TORQUE_HPP_
#define CROCODDYL_CORE_RESIDUALS_JOINT_TORQUE_HPP_

#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/data/joint.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/residual-base.hpp"

namespace crocoddyl {

/**
 * @brief Define a joint-effort residual
 *
 * This residual function is defined as
 * \f$\mathbf{r}=\mathbf{u}-\mathbf{u}^*\f$, where
 * \f$\mathbf{u},\mathbf{u}^*\in~\mathbb{R}^{nu}\f$ are the current and
 * reference joint effort inputs, respectively. Note that the dimension of the
 * residual vector is obtained from `ActuationModelAbstract::nu`.
 *
 * Both residual and residual Jacobians are computed analytically.
 *
 * As described in ResidualModelAbstractTpl(), the residual value and its
 * Jacobians are calculated by `calc` and `calcDiff`, respectively.
 *
 * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ResidualModelJointEffortTpl : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ResidualModelBase, ResidualModelJointEffortTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataJointEffortTpl<Scalar> Data;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef StateAbstractTpl<Scalar> StateAbstract;
  typedef ActuationModelAbstractTpl<Scalar> ActuationModelAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the joint-effort residual model
   *
   * @param[in] state      State description
   * @param[in] actuation  Actuation model
   * @param[in] uref       Reference joint effort
   * @param[in] nu         Dimension of the control vector
   * @param[in] fwddyn     Indicates that we have a forward dynamics problem
   * (true) or inverse dynamics (false) (default false)
   */
  ResidualModelJointEffortTpl(std::shared_ptr<StateAbstract> state,
                              std::shared_ptr<ActuationModelAbstract> actuation,
                              const VectorXs& uref, const std::size_t nu,
                              const bool fwddyn = false);

  /**
   * @brief Initialize the joint-effort residual model
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state      State description
   * @param[in] actuation  Actuation model
   * @param[in] uref       Reference joint effort
   */
  ResidualModelJointEffortTpl(std::shared_ptr<StateAbstract> state,
                              std::shared_ptr<ActuationModelAbstract> actuation,
                              const VectorXs& uref);

  /**
   * @brief Initialize the joint-effort residual model
   *
   * The default reference joint effort is obtained from
   * `MathBaseTpl<>::VectorXs::Zero(actuation->get_nu())`.
   *
   * @param[in] state      State description
   * @param[in] actuation  Actuation model
   * @param[in] nu         Dimension of the control vector
   */
  ResidualModelJointEffortTpl(std::shared_ptr<StateAbstract> state,
                              std::shared_ptr<ActuationModelAbstract> actuation,
                              const std::size_t nu);

  /**
   * @brief Initialize the joint-effort residual model
   *
   * The default reference joint effort is obtained from
   * `MathBaseTpl<>::VectorXs::Zero(actuation->get_nu())`. The default `nu`
   * value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state      State description
   * @param[in] actuation  Actuation model
   */
  ResidualModelJointEffortTpl(
      std::shared_ptr<StateAbstract> state,
      std::shared_ptr<ActuationModelAbstract> actuation);

  virtual ~ResidualModelJointEffortTpl() = default;

  /**
   * @brief Compute the joint-effort residual
   *
   * @param[in] data  Joint-effort residual data
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
   * @brief Compute the derivatives of the joint-effort residual
   *
   * @param[in] data  Joint-effort residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief @copydoc Base::calcDiff(const
   * std::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const
   * VectorXs>& x)
   */
  virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Create the joint-effort residual data
   */
  virtual std::shared_ptr<ResidualDataAbstract> createData(
      DataCollectorAbstract* const data) override;

  /**
   * @brief Cast the joint-effort residual model to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return ResidualModelJointEffortTpl<NewScalar> A residual model with the
   * new scalar type.
   */
  template <typename NewScalar>
  ResidualModelJointEffortTpl<NewScalar> cast() const;

  /**
   * @brief Return the reference joint-effort vector
   */
  const VectorXs& get_reference() const;

  /**
   * @brief Modify the reference joint-effort vector
   */
  void set_reference(const VectorXs& reference);

  /**
   * @brief Print relevant information of the joint-effort residual
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override;

 protected:
  using Base::nr_;
  using Base::nu_;
  using Base::q_dependent_;
  using Base::state_;
  using Base::v_dependent_;

 private:
  std::shared_ptr<ActuationModelAbstract> actuation_;  //!< Actuation model
  VectorXs uref_;  //!< Reference joint-effort input
  bool fwddyn_;    //!< True for forward dynamics, False for inverse dynamics
};

template <typename _Scalar>
struct ResidualDataJointEffortTpl : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;

  template <template <typename Scalar> class Model>
  ResidualDataJointEffortTpl(Model<Scalar>* const model,
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
  virtual ~ResidualDataJointEffortTpl() = default;

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
#include "crocoddyl/core/residuals/joint-effort.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ResidualModelJointEffortTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ResidualDataJointEffortTpl)

#endif  // CROCODDYL_CORE_RESIDUALS_JOINT_TORQUE_HPP_
