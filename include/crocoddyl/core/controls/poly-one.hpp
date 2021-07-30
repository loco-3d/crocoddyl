///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_CONTROLS_POLY_ONE_HPP_
#define CROCODDYL_CORE_CONTROLS_POLY_ONE_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/control-base.hpp"

namespace crocoddyl {

/**
 * @brief A polynomial function of time of degree one, that is a linear function
 *
 * The size of the parameters p is twice the size of the control input u.
 * The first half of p represents the value of u at time 0.
 * The second half of p represents the value of u at time 0.5.
 */
template <typename _Scalar>
class ControlParametrizationModelPolyOneTpl : public ControlParametrizationModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef ControlParametrizationDataAbstractTpl<Scalar> ControlParametrizationDataAbstract;
  typedef ControlParametrizationModelAbstractTpl<Scalar> Base;

  explicit ControlParametrizationModelPolyOneTpl(const std::size_t nu);
  virtual ~ControlParametrizationModelPolyOneTpl();

  /**
   * @brief Get the value of the control at the specified time
   *
   * @param[in]  data   Data structure containing the control vector to write
   * @param[in]  t      Time in [0,1]
   * @param[in]  p      Control parameters
   */
  virtual void calc(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double t,
                    const Eigen::Ref<const VectorXs>& p) const;

  /**
   * @brief Get a value of the control parameters such that the control at the specified time
   * t is equal to the specified value u
   *
   * @param[in]  data   Data structure containing the control parameters vector to write
   * @param[in]  t      Time in [0,1]
   * @param[in]  u      Control values
   */
  virtual void params(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double t,
                      const Eigen::Ref<const VectorXs>& u) const;

  /**
   * @brief Map the specified bounds from the control space to the parameter space
   *
   * @param[in]  u_lb   Control lower bound
   * @param[in]  u_ub   Control lower bound
   * @param[out] p_lb   Control parameters lower bound
   * @param[out] p_ub   Control parameters upper bound
   */
  virtual void convertBounds(const Eigen::Ref<const VectorXs>& u_lb, const Eigen::Ref<const VectorXs>& u_ub,
                              Eigen::Ref<VectorXs> p_lb, Eigen::Ref<VectorXs> p_ub) const;

  /**
   * @brief Get the value of the Jacobian of the control with respect to the parameters
   *
   * @param[in]  data   Data structure containing the Jacobian matrix to write
   * @param[in]  t      Time in [0,1]
   * @param[in]  p      Control parameters
   */
  virtual void calcDiff(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double t,
                        const Eigen::Ref<const VectorXs>& p) const;

  /**
   * @brief Compute the product between a specified matrix and the Jacobian of the control (with respect to the
   * parameters)
   *
   * @param[in]  t      Time
   * @param[in]  p      Control parameters
   * @param[in]  A      A matrix to multiply times the Jacobian
   * @param[out] out    Product between the matrix A and the Jacobian of the control with respect to the parameters
   */
  virtual void multiplyByJacobian(double t, const Eigen::Ref<const VectorXs>& p, const Eigen::Ref<const MatrixXs>& A,
                                  Eigen::Ref<MatrixXs> out) const;

  /**
   * @brief Compute the product between the transposed Jacobian of the control (with respect to the parameters) and
   * a specified matrix
   *
   * @param[in]  t      Time
   * @param[in]  p      Control parameters
   * @param[in]  A      A matrix to multiply times the Jacobian
   * @param[out] out    Product between the transposed Jacobian of the control with respect to the parameters and the
   * matrix A
   */
  virtual void multiplyJacobianTransposeBy(double t, const Eigen::Ref<const VectorXs>& p,
                                           const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXs> out) const;

 protected:
  using Base::nw_;
  using Base::np_;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/controls/poly-one.hxx"

#endif  // CROCODDYL_CORE_CONTROLS_POLY_ONE_HPP_
