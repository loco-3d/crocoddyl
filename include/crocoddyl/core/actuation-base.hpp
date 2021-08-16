///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTUATION_BASE_HPP_
#define CROCODDYL_CORE_ACTUATION_BASE_HPP_

#include <stdexcept>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/mathbase.hpp"
#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/core/utils/to-string.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief Abstract class for the actuation-mapping model
 *
 * In Crocoddyl, the generalized torques \f$\mathbf{a}\in\mathbb{R}^{nv}\f$ can by any nonlinear function of the
 * control inputs \f$\mathbf{u}\in\mathbb{R}^{nu}\f$, and state point \f$\mathbf{x}\in\mathbb{R}^{nx}\f$, where
 * `nv`, `nu`, and `ndx` are the number of joints, dimension of the control input and state manifold,
 * respectively. Additionally, the generalized torques are also named as the actuation signals of our system.
 *
 * The main computations are carrying out in `calc`, and `calcDiff`, where the former computes actuation signal
 * \f$\mathbf{a}\f$ from a given control input \f$\mathbf{u}\f$ and state point \f$\mathbf{x}\f$, and the latter
 * computes the Jacobians of the actuation-mapping function. Note that `caldDiff` requires to run `calc` first.
 *
 * \sa `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ActuationModelAbstractTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef StateAbstractTpl<Scalar> StateAbstract;
  typedef ActuationDataAbstractTpl<Scalar> ActuationDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the actuation model
   *
   * @param[in] state  State description
   * @param[in] nu     Dimension of control vector
   */
  ActuationModelAbstractTpl(boost::shared_ptr<StateAbstract> state, const std::size_t nu) : nu_(nu), state_(state) {
    if (nu_ == 0) {
      throw_pretty("Invalid argument: "
                   << "nu cannot be zero");
    }
  };
  virtual ~ActuationModelAbstractTpl(){};

  /**
   * @brief Compute the actuation signal from the control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   *
   * @param[in] data  Actuation data
   * @param[in] x     State point
   * @param[in] u     Control input
   */
  virtual void calc(const boost::shared_ptr<ActuationDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) = 0;

  /**
   * @brief Compute the Jacobians of the actuation function
   *
   * @param[in] data  Actuation data
   * @param[in] x     State point
   * @param[in] u     Control input
   */
  virtual void calcDiff(const boost::shared_ptr<ActuationDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) = 0;

  /**
   * @brief Create the actuation data
   *
   * @return the actuation data
   */
  virtual boost::shared_ptr<ActuationDataAbstract> createData() {
    return boost::allocate_shared<ActuationDataAbstract>(Eigen::aligned_allocator<ActuationDataAbstract>(), this);
  };

  /**
   * @brief Return the dimension of the control input
   */
  std::size_t get_nu() const { return nu_; };

  /**
   * @brief Return the state
   */
  const boost::shared_ptr<StateAbstract>& get_state() const { return state_; };

 protected:
  std::size_t nu_;                          //!< Control dimension
  boost::shared_ptr<StateAbstract> state_;  //!< Model of the state
};

template <typename _Scalar>
struct ActuationDataAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit ActuationDataAbstractTpl(Model<Scalar>* const model)
      : tau(model->get_state()->get_nv()),
        dtau_dx(model->get_state()->get_nv(), model->get_state()->get_ndx()),
        dtau_du(model->get_state()->get_nv(), model->get_nu()) {
    tau.setZero();
    dtau_dx.setZero();
    dtau_du.setZero();
  }
  virtual ~ActuationDataAbstractTpl() {}

  VectorXs tau;      //!< Actuation (generalized force) signal
  MatrixXs dtau_dx;  //!< Partial derivatives of the actuation model w.r.t. the state point
  MatrixXs dtau_du;  //!< Partial derivatives of the actuation model w.r.t. the control input
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTUATION_BASE_HPP_
