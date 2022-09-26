///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2022, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTUATIONS_FLOATING_BASE_HPP_
#define CROCODDYL_MULTIBODY_ACTUATIONS_FLOATING_BASE_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

/**
 * @brief Floating-base actuation model
 *
 * It considers the first joint, defined in the Pinocchio model, as the floating-base joints.
 * Then, this joint (that might have various DoFs) is unactuated.
 *
 * The main computations are carrying out in `calc`, and `calcDiff`, where the former computes actuation signal
 * \f$\mathbf{a}\f$ from a given control input \f$\mathbf{u}\f$ and state point \f$\mathbf{x}\f$, and the latter
 * computes the Jacobians of the actuation-mapping function. Note that `calcDiff` requires to run `calc` first.
 *
 * \sa `ActuationModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ActuationModelFloatingBaseTpl : public ActuationModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActuationModelAbstractTpl<Scalar> Base;
  typedef ActuationDataAbstractTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the floating-base actuation model
   *
   * @param[in] state  State of a multibody system
   * @param[in] nu     Dimension of control vector
   */
  explicit ActuationModelFloatingBaseTpl(boost::shared_ptr<StateMultibody> state)
      : Base(state, state->get_nv() - state->get_pinocchio()->joints[1].nv()){};
  virtual ~ActuationModelFloatingBaseTpl(){};

  /**
   * @brief Compute the floating-base actuation signal from the control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   *
   * @param[in] data  Actuation data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Joint torque input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<Data>& data, const Eigen::Ref<const VectorXs>& /*x*/,
                    const Eigen::Ref<const VectorXs>& u) {
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw_pretty("Invalid argument: "
                   << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
    }
    data->tau.tail(nu_) = u;
  };

    /**
     * @brief Compute the Jacobians of the floating-base actuation function
     *
     * @param[in] data  Actuation data
     * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
     * @param[in] u     Joint torque input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
     */
#ifndef NDEBUG
  virtual void calcDiff(const boost::shared_ptr<Data>& data, const Eigen::Ref<const VectorXs>& /*x*/,
                        const Eigen::Ref<const VectorXs>& /*u*/) {
#else
  virtual void calcDiff(const boost::shared_ptr<Data>&, const Eigen::Ref<const VectorXs>& /*x*/,
                        const Eigen::Ref<const VectorXs>& /*u*/) {
#endif
    // The derivatives has constant values which were set in createData.
    assert_pretty(data->dtau_dx.isZero(), "dtau_dx has wrong value");
    assert_pretty(MatrixXs(data->dtau_du).isApprox(dtau_du_), "dtau_du has wrong value");
  };

  virtual void commands(const boost::shared_ptr<Data>& data, const Eigen::Ref<const VectorXs>&,
                        const Eigen::Ref<const VectorXs>& tau) {
    if (static_cast<std::size_t>(tau.size()) != state_->get_nv()) {
      throw_pretty("Invalid argument: "
                   << "tau has wrong dimension (it should be " + std::to_string(state_->get_nv()) + ")");
    }
    data->u = tau.tail(nu_);
  }

#ifndef NDEBUG
  virtual void torqueTransform(const boost::shared_ptr<Data>& data, const Eigen::Ref<const VectorXs>&,
                               const Eigen::Ref<const VectorXs>&) {
#else
  virtual void torqueTransform(const boost::shared_ptr<Data>&, const Eigen::Ref<const VectorXs>&,
                               const Eigen::Ref<const VectorXs>&) {
#endif
    // The torque transform has constant values which were set in createData.
    assert_pretty(MatrixXs(data->Mtau).isApprox(Mtau_), "Mtau has wrong value");
  }

  /**
   * @brief Create the floating-base actuation data
   *
   * @return the actuation data
   */
  virtual boost::shared_ptr<Data> createData() {
    typedef StateMultibodyTpl<Scalar> StateMultibody;
    boost::shared_ptr<StateMultibody> state = boost::static_pointer_cast<StateMultibody>(state_);
    boost::shared_ptr<Data> data = boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
    const std::size_t nfb = state->get_pinocchio()->joints[1].nv();
    data->dtau_du.diagonal(-nfb).setOnes();
    data->Mtau.diagonal(nfb).setOnes();
    for (std::size_t i = 0; i < nfb; ++i) {
      data->tau_set[i] = false;
    }
#ifndef NDEBUG
    dtau_du_ = data->dtau_du;
    Mtau_ = data->Mtau;
#endif
    return data;
  };

 protected:
  using Base::nu_;
  using Base::state_;

#ifndef NDEBUG
 private:
  MatrixXs dtau_du_;
  MatrixXs Mtau_;
#endif
};

}  // namespace crocoddyl

#endif  // CROCODDYL_MULTIBODY_ACTUATIONS_FLOATING_BASE_HPP_
