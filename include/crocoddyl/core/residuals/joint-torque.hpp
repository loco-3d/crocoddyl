///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2022, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_RESIDUALS_JOINT_TORQUE_HPP_
#define CROCODDYL_CORE_RESIDUALS_JOINT_TORQUE_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/data/joint.hpp"

namespace crocoddyl {

/**
 * @brief Define a joint-torque residual
 *
 * This residual function is defined as \f$\mathbf{r}=\mathbf{u}-\mathbf{u}^*\f$, where
 * \f$\mathbf{u},\mathbf{u}^*\in~\mathbb{R}^{nu}\f$ are the current and reference joint torque inputs, respectively.
 * Note that the dimension of the residual vector is obtained from `ActuationModelAbstract::nu`.
 *
 * Both residual and residual Jacobians are computed analytically.
 *
 * As described in ResidualModelAbstractTpl(), the residual value and its Jacobians are calculated by `calc` and
 * `calcDiff`, respectively.
 *
 * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ResidualModelJointTorqueTpl : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataJointTorqueTpl<Scalar> Data;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef StateAbstractTpl<Scalar> StateAbstract;
  typedef ActuationModelAbstractTpl<Scalar> ActuationModelAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the joint-torque residual model
   *
   * @param[in] state       State description
   * @param[in] actuation   Actuation model
   * @param[in] uref        Reference joint torque
   * @param[in] nu          Dimension of the control vector
   */
  ResidualModelJointTorqueTpl(boost::shared_ptr<StateAbstract> state,
                              boost::shared_ptr<ActuationModelAbstract> actuation, const VectorXs& uref,
                              const std::size_t nu);

  /**
   * @brief Initialize the joint-torque residual model
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state       State description
   * @param[in] actuation   Actuation model
   * @param[in] uref        Reference joint torque
   */
  ResidualModelJointTorqueTpl(boost::shared_ptr<StateAbstract> state,
                              boost::shared_ptr<ActuationModelAbstract> actuation, const VectorXs& uref);

  /**
   * @brief Initialize the joint-torque residual model
   *
   * The default reference joint torque is obtained from `MathBaseTpl<>::VectorXs::Zero(actuation->get_nu())`.
   *
   * @param[in] state       State description
   * @param[in] actuation   Actuation model
   * @param[in] nu          Dimension of the control vector
   */
  ResidualModelJointTorqueTpl(boost::shared_ptr<StateAbstract> state,
                              boost::shared_ptr<ActuationModelAbstract> actuation, const std::size_t nu);

  /**
   * @brief Initialize the joint-torque residual model
   *
   * The default reference joint torque is obtained from `MathBaseTpl<>::VectorXs::Zero(actuation->get_nu())`.
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state       State description
   * @param[in] actuation   Actuation model
   */
  ResidualModelJointTorqueTpl(boost::shared_ptr<StateAbstract> state,
                              boost::shared_ptr<ActuationModelAbstract> actuation);

  virtual ~ResidualModelJointTorqueTpl();

  /**
   * @brief Compute the joint-torque residual
   *
   * @param[in] data  Joint-torque residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief @copydoc Base::calc(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>&
   * x)
   */
  virtual void calc(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the derivatives of the joint-torque residual
   *
   * @param[in] data  Joint-torque residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Create the joint-torque residual data
   */
  virtual boost::shared_ptr<ResidualDataAbstract> createData(DataCollectorAbstract* const data);

  /**
   * @brief Return the reference joint-torque vector
   */
  const VectorXs& get_reference() const;

  /**
   * @brief Modify the reference joint-torque vector
   */
  void set_reference(const VectorXs& reference);

  /**
   * @brief Print relevant information of the joint-torque residual
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 protected:
  using Base::nu_;
  using Base::state_;
  using Base::unone_;

 private:
  VectorXs uref_;  //!< Reference joint-torque input
};

template <typename _Scalar>
struct ResidualDataJointTorqueTpl : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;

  template <template <typename Scalar> class Model>
  ResidualDataJointTorqueTpl(Model<Scalar>* const model, DataCollectorAbstract* const data) : Base(model, data) {
    // Check that proper shared data has been passed
    DataCollectorJointTpl<Scalar>* d = dynamic_cast<DataCollectorJointTpl<Scalar>*>(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorJoint");
    }

    joint = d->joint;
  }

  boost::shared_ptr<JointDataAbstractTpl<Scalar> > joint;  //!< Joint data
  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/residuals/joint-torque.hxx"

#endif  // CROCODDYL_CORE_RESIDUALS_JOINT_TORQUE_HPP_
