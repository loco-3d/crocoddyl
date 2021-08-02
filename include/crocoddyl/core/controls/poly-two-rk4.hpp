///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_CONTROLS_POLY_TWO_RK4_HPP_
#define CROCODDYL_CORE_CONTROLS_POLY_TWO_RK4_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/control-base.hpp"

namespace crocoddyl {

/**
 * @brief A polynomial function of time of degree two, that is a quadratic function
 *
 * The size of the parameters p is 3 times the size of the control input u.
 * The first third of p represents the value of u at time 0.
 * The second third of p represents the value of u at time 0.5.
 * The last third of p represents the value of u at time 1.
 * This parametrization is suitable to be used with the RK-4 integration scheme,
 * because it requires the value of u exactly at 0, 0.5 and 1.
 */
template <typename _Scalar>
class ControlParametrizationModelPolyTwoRK4Tpl : public ControlParametrizationModelAbstractTpl<_Scalar> {
 public:
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef ControlParametrizationDataAbstractTpl<Scalar> ControlParametrizationDataAbstract;
  typedef ControlParametrizationModelAbstractTpl<Scalar> Base;
  typedef ControlParametrizationDataPolyTwoRK4Tpl<Scalar> Data;

  explicit ControlParametrizationModelPolyTwoRK4Tpl(const std::size_t nw);
  virtual ~ControlParametrizationModelPolyTwoRK4Tpl();

  virtual boost::shared_ptr<ControlParametrizationDataAbstract> createData();

  /**
   * @brief Get the value of the control at the specified time
   *
   * @param[in]  data   Data structure containing the control vector to write
   * @param[in]  t      Time in [0,1]
   * @param[in]  u      Control parameters
   */
  virtual void calc(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double t,
                    const Eigen::Ref<const VectorXs>& u) const;

  /**
   * @brief Get a value of the control parameters such that the control at the specified time
   * t is equal to the specified value u
   *
   * @param[in]  data   Data structure containing the control parameters vector to write
   * @param[in]  t      Time in [0,1]
   * @param[in]  w      Control values
   */
  virtual void params(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double t,
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
   * @brief Get the value of the Jacobian of the control with respect to the parameters
   *
   * @param[in]  data   Data structure containing the Jacobian matrix to write
   * @param[in]  t      Time in [0,1]
   * @param[in]  u      Control parameters
   */
  virtual void calcDiff(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double t,
                        const Eigen::Ref<const VectorXs>& u) const;

  /**
   * @brief Compute the product between a specified matrix and the Jacobian of the control (with respect to the
   * parameters)
   *
   * @param[in]  t      Time
   * @param[in]  u      Control parameters
   * @param[in]  A      A matrix to multiply times the Jacobian
   * @param[out] out    Product between the matrix A and the Jacobian of the control with respect to the parameters
   */
  virtual void multiplyByJacobian(double t, const Eigen::Ref<const VectorXs>& u, const Eigen::Ref<const MatrixXs>& A,
                                  Eigen::Ref<MatrixXs> out) const;

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
  virtual void multiplyJacobianTransposeBy(double t, const Eigen::Ref<const VectorXs>& u,
                                           const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXs> out) const;

 protected:
  using Base::nu_;
  using Base::nw_;
};

template <typename _Scalar>
struct ControlParametrizationDataPolyTwoRK4Tpl : public ControlParametrizationDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef ControlParametrizationDataAbstractTpl<Scalar> Base;

  template <template <typename Scalar> class Model>
  explicit ControlParametrizationDataPolyTwoRK4Tpl(Model<Scalar>* const model) : Base(model) {}

  virtual ~ControlParametrizationDataPolyTwoRK4Tpl() {}

  Scalar tmp_t2;  //!< Temporary variable to store the square of the time
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/controls/poly-two-rk4.hxx"

#endif  // CROCODDYL_CORE_CONTROLS_POLY_TWO_RK4_HPP_
