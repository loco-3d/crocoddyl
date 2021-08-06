///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_CONTROLS_POLY_ZERO_HPP_
#define CROCODDYL_CORE_CONTROLS_POLY_ZERO_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/control-base.hpp"

namespace crocoddyl {

/**
 * @brief A polynomial function of time of degree zero, that is a constant
 *
 * The main computations are carrying out in `calc`, `multiplyByJacobian` and `multiplyJacobianTransposeBy`,
 * where the former computes control input \f$\mathbf{w}\f from a set of control parameters \f$\mathbf{u}\f,
 * and the latters defines useful operations across the Jacobian of the control-parametrization model.
 * Finally, `params` allows us to obtain the control parameters from a the control input, i.e., it is the
 * dual of `calc`.
 * Note that `multiplyByJacobian` and `multiplyJacobianTransposeBy` requires to run `calc` first.
 *
 * \sa `calc()`, `calcDiff()`, `createData()`, `params`, `multiplyByJacobian`, `multiplyJacobianTransposeBy`
 */
template <typename _Scalar>
class ControlParametrizationModelPolyZeroTpl : public ControlParametrizationModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef ControlParametrizationDataAbstractTpl<Scalar> ControlParametrizationDataAbstract;
  typedef ControlParametrizationModelAbstractTpl<Scalar> Base;

  explicit ControlParametrizationModelPolyZeroTpl(const std::size_t nw);
  virtual ~ControlParametrizationModelPolyZeroTpl();

  /**
   * @brief Get the value of the control at the specified time
   *
   * @param[in]  data   Control-parametrization data
   * @param[in]  t      Time in [0,1]
   * @param[in]  u      Control parameters
   */
  virtual void calc(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, const Scalar t,
                    const Eigen::Ref<const VectorXs>& u) const;

  /**
   * @brief Get the value of the Jacobian of the control with respect to the parameters
   *
   * @param[in]  data   Control-parametrization data
   * @param[in]  t      Time in [0,1]
   * @param[in]  u      Control parameters
   */
  virtual void calcDiff(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, const Scalar t,
                        const Eigen::Ref<const VectorXs>& u) const;

  /**
   * @brief Get a value of the control parameters such that the control at the specified time
   * t is equal to the specified value u
   *
   * @param[in]  data   Control-parametrization data
   * @param[in]  t      Time in [0,1]
   * @param[in]  w      Control values
   */
  virtual void params(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, const Scalar t,
                      const Eigen::Ref<const VectorXs>& w) const;

  /**
   * @brief Map the specified bounds from the control space to the parameter space
   *
   * @param[in]  w_lb   Control lower bound
   * @param[in]  w_ub   Control lower bound
   * @param[out] u_lb   Control parameters lower bound
   * @param[out] u_ub   Control parameters upper bound
   */
  virtual void convertBounds(const Eigen::Ref<const VectorXs>& w_lb, const Eigen::Ref<const VectorXs>& w_ub,
                             Eigen::Ref<VectorXs> u_lb, Eigen::Ref<VectorXs> u_ub) const;

  /**
   * @brief Compute the product between a specified matrix and the Jacobian of the control (with respect to the
   * parameters)
   *
   * @param[in]  t      Time
   * @param[in]  u      Control parameters
   * @param[in]  A      A matrix to multiply times the Jacobian
   * @param[out] out    Product between the matrix A and the Jacobian of the control with respect to the parameters
   */
  virtual void multiplyByJacobian(const Scalar t, const Eigen::Ref<const VectorXs>& u,
                                  const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXs> out) const;

  /**
   * @brief Compute the product between the transposed Jacobian of the control (with respect to the parameters) and
   * a specified matrix
   *
   * @param[in]  t      Time
   * @param[in]  u      Control parameters
   * @param[in]  A      A matrix to multiply times the Jacobian
   * @param[out] out    Product between the transposed Jacobian of the control with respect to the parameters and the
   * matrix A
   */
  virtual void multiplyJacobianTransposeBy(const Scalar t, const Eigen::Ref<const VectorXs>& u,
                                           const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXs> out) const;

 protected:
  using Base::nu_;
  using Base::nw_;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/controls/poly-zero.hxx"

#endif  // CROCODDYL_CORE_CONTROLS_POLY_ZERO_HPP_
