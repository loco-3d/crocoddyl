///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh, INRIA, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_CONTROL_BASE_HPP_
#define CROCODDYL_CORE_CONTROL_BASE_HPP_

#include <vector>
#include <string>
#include <stdexcept>

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/mathbase.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief Abstract class for the control trajectory discretization
 * 
 * The control trajectory is a function of the (normalized) time.
 * Normalized time is between 0 and 1, where 0 represents the beginning of the time step, 
 * and 1 represents its end.
 * The trajectory depends on the control parameters p, whose size may be larger than the
 * size of the control inputs u.
 *
 */
template <typename _Scalar>
class ControlAbstractTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the control dimensions
   *
   * @param[in] nx   Dimension of control
   * @param[in] np   Dimension of control parameters
   */
  ControlAbstractTpl(const std::size_t nu, const std::size_t np);
  ControlAbstractTpl();
  virtual ~ControlAbstractTpl();

  virtual void resize(const std::size_t nu) = 0;

  /**
   * @brief Generate a zero control
   */
//   virtual VectorXs zero() const = 0;

  /**
   * @brief Generate a random control
   */
//   virtual VectorXs rand() const = 0;

  /**
   * @brief Get the value of the control at the specified time
   *
   * @param[in]  t      Time
   * @param[in]  p      Control parameters
   * @param[out] u_out  Control value
   */
  virtual void value(double t, const Eigen::Ref<const VectorXs>& p, Eigen::Ref<VectorXs> u_out) const = 0;

  /**
   * @brief Get the value of the Jacobian of the control with respect to the parameters
   *
   * @param[in]  t      Time
   * @param[in]  p      Control parameters
   * @param[out] J_out  Jacobian of the control value with respect to the parameters
   */
  virtual void dValue(double t, const Eigen::Ref<const VectorXs>& p, Eigen::Ref<MatrixXs> J_out) const = 0;

  /**
   * @brief Compute the product between a specified matrix and the Jacobian of the control (with respect to the parameters)
   *
   * @param[in]  t      Time
   * @param[in]  p      Control parameters
   * @param[in]  A      A matrix to multiply times the Jacobian
   * @param[out] out    Product between the matrix A and the Jacobian of the control with respect to the parameters
   */
  virtual void multiplyByDValue(double t, const Eigen::Ref<const VectorXs>& p, 
        const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXs> out) const = 0;

  /**
   * @brief Return the dimension of the control value
   */
  std::size_t get_nu() const;

  /**
   * @brief Return the dimension of control parameters
   */
  std::size_t get_np() const;

 protected:

  std::size_t nu_;  //!< Control dimension
  std::size_t np_;  //!< Control parameters dimension
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/control-base.hxx"

#endif  // CROCODDYL_CORE_CONTROL_BASE_HPP_
