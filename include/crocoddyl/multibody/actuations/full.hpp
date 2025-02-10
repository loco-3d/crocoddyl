///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2022, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTUATIONS_FULL_HPP_
#define CROCODDYL_MULTIBODY_ACTUATIONS_FULL_HPP_

#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/multibody/fwd.hpp"

namespace crocoddyl {

/**
 * @brief Full actuation model
 *
 * This actuation model applies input controls for all the `nv` dimensions of
 * the system.
 *
 * Both actuation and Jacobians are computed analytically by `calc` and
 * `calcDiff`, respectively.
 *
 * \sa `ActuationModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ActuationModelFullTpl : public ActuationModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActuationModelAbstractTpl<Scalar> Base;
  typedef ActuationDataAbstractTpl<Scalar> Data;
  typedef StateAbstractTpl<Scalar> StateAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the full actuation model
   *
   * @param[in] state  State of the dynamical system
   */
  explicit ActuationModelFullTpl(std::shared_ptr<StateAbstract> state)
      : Base(state, state->get_nv()) {};
  virtual ~ActuationModelFullTpl() {};

  /**
   * @brief Compute the full actuation
   *
   * @param[in] data  Full actuation data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Joint torque input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<Data>& data,
                    const Eigen::Ref<const VectorXs>& /*x*/,
                    const Eigen::Ref<const VectorXs>& u) {
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw_pretty(
          "Invalid argument: " << "u has wrong dimension (it should be " +
                                      std::to_string(nu_) + ")");
    }
    data->tau = u;
  };

  /**
   * @brief Compute the Jacobians of the full actuation model
   *
   * @param[in] data  Full actuation data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Joint torque input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
#ifndef NDEBUG
  virtual void calcDiff(const std::shared_ptr<Data>& data,
                        const Eigen::Ref<const VectorXs>& /*x*/,
                        const Eigen::Ref<const VectorXs>&) {
#else
  virtual void calcDiff(const std::shared_ptr<Data>&,
                        const Eigen::Ref<const VectorXs>& /*x*/,
                        const Eigen::Ref<const VectorXs>&) {
#endif
    // The derivatives has constant values which were set in createData.
    assert_pretty(data->dtau_dx.isZero(), "dtau_dx has wrong value");
    assert_pretty(MatrixXs(data->dtau_du)
                      .isApprox(MatrixXs::Identity(state_->get_nv(), nu_)),
                  "dtau_du has wrong value");
  };

  virtual void commands(const std::shared_ptr<Data>& data,
                        const Eigen::Ref<const VectorXs>&,
                        const Eigen::Ref<const VectorXs>& tau) {
    if (static_cast<std::size_t>(tau.size()) != nu_) {
      throw_pretty(
          "Invalid argument: " << "tau has wrong dimension (it should be " +
                                      std::to_string(nu_) + ")");
    }
    data->u = tau;
  }

#ifndef NDEBUG
  virtual void torqueTransform(const std::shared_ptr<Data>& data,
                               const Eigen::Ref<const VectorXs>&,
                               const Eigen::Ref<const VectorXs>&) {
#else
  virtual void torqueTransform(const std::shared_ptr<Data>&,
                               const Eigen::Ref<const VectorXs>&,
                               const Eigen::Ref<const VectorXs>&) {
#endif
    // The torque transform has constant values which were set in createData.
    assert_pretty(MatrixXs(data->Mtau).isApprox(MatrixXs::Identity(nu_, nu_)),
                  "Mtau has wrong value");
  }

  /**
   * @brief Create the full actuation data
   *
   * @param[in] data  shared data (it should be of type DataCollectorContactTpl)
   * @return the cost data.
   */
  virtual std::shared_ptr<Data> createData() {
    std::shared_ptr<Data> data =
        std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
    data->dtau_du.diagonal().setOnes();
    data->Mtau.setIdentity();
    return data;
  };

 protected:
  using Base::nu_;
  using Base::state_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_ACTUATIONS_FULL_HPP_
