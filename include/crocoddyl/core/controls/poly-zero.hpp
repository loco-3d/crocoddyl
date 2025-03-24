///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, University of Edinburgh, University of Trento,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_CONTROLS_POLY_ZERO_HPP_
#define CROCODDYL_CORE_CONTROLS_POLY_ZERO_HPP_

#include "crocoddyl/core/control-base.hpp"
#include "crocoddyl/core/fwd.hpp"

namespace crocoddyl {

/**
 * @brief A polynomial function of time of degree zero, that is a constant
 *
 * The main computations are carrying out in `calc`, `multiplyByJacobian` and
 * `multiplyJacobianTransposeBy`, where the former computes control input
 * \f$\mathbf{w}\in\mathbb{R}^{nw}\f$ from a set of control parameters
 * \f$\mathbf{u}\in\mathbb{R}^{nu}\f$ where `nw` and `nu` represent the
 * dimension of the control inputs and parameters, respectively, and the latter
 * defines useful operations across the Jacobian of the control-parametrization
 * model. Finally, `params` allows us to obtain the control parameters from a
 * the control input, i.e., it is the dual of `calc`. Note that
 * `multiplyByJacobian` and `multiplyJacobianTransposeBy` requires to run `calc`
 * first.
 *
 * \sa `ControlParametrizationAbstractTpl`, `calc()`, `calcDiff()`,
 * `createData()`, `params`, `multiplyByJacobian`, `multiplyJacobianTransposeBy`
 */
template <typename _Scalar>
class ControlParametrizationModelPolyZeroTpl
    : public ControlParametrizationModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ControlParametrizationModelBase,
                         ControlParametrizationModelPolyZeroTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ControlParametrizationDataAbstractTpl<Scalar>
      ControlParametrizationDataAbstract;
  typedef ControlParametrizationModelAbstractTpl<Scalar> Base;
  typedef ControlParametrizationDataAbstractTpl<Scalar> Data;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the poly-zero control parametrization
   *
   * @param[in] nw  Dimension of control vector
   */
  explicit ControlParametrizationModelPolyZeroTpl(const std::size_t nw);
  virtual ~ControlParametrizationModelPolyZeroTpl() = default;

  /**
   * @brief Get the value of the control at the specified time
   *
   * @param[in]  data   Control-parametrization data
   * @param[in]  t      Time in [0,1]
   * @param[in]  u      Control parameters
   */
  virtual void calc(
      const std::shared_ptr<ControlParametrizationDataAbstract>& data,
      const Scalar t, const Eigen::Ref<const VectorXs>& u) const override;

  /**
   * @brief Get the value of the Jacobian of the control with respect to the
   * parameters
   *
   * It assumes that `calc()` has been run first
   *
   * @param[in]  data   Control-parametrization data
   * @param[in]  t      Time in [0,1]
   * @param[in]  u      Control parameters
   */
  virtual void calcDiff(
      const std::shared_ptr<ControlParametrizationDataAbstract>& data,
      const Scalar t, const Eigen::Ref<const VectorXs>& u) const override;

  /**
   * @brief Create the control-parametrization data
   *
   * @return the control-parametrization data
   */
  virtual std::shared_ptr<ControlParametrizationDataAbstract> createData()
      override;

  /**
   * @brief Get a value of the control parameters such that the control at the
   * specified time t is equal to the specified value u
   *
   * @param[in]  data   Control-parametrization data
   * @param[in]  t      Time in [0,1]
   * @param[in]  w      Control values
   */
  virtual void params(
      const std::shared_ptr<ControlParametrizationDataAbstract>& data,
      const Scalar t, const Eigen::Ref<const VectorXs>& w) const override;

  /**
   * @brief Map the specified bounds from the control space to the parameter
   * space
   *
   * @param[in]  w_lb   Control lower bound
   * @param[in]  w_ub   Control lower bound
   * @param[out] u_lb   Control parameters lower bound
   * @param[out] u_ub   Control parameters upper bound
   */
  virtual void convertBounds(const Eigen::Ref<const VectorXs>& w_lb,
                             const Eigen::Ref<const VectorXs>& w_ub,
                             Eigen::Ref<VectorXs> u_lb,
                             Eigen::Ref<VectorXs> u_ub) const override;

  /**
   * @brief Compute the product between a specified matrix and the Jacobian of
   * the control (with respect to the parameters)
   *
   * It assumes that `calc()` has been run first
   *
   * @param[in]  data   Control-parametrization data
   * @param[in]  A      A matrix to multiply times the Jacobian
   * @param[out] out    Product between the matrix A and the Jacobian of the
   * control with respect to the parameters
   * @param[in] op      Assignment operator which sets, adds, or removes the
   * given results
   */
  virtual void multiplyByJacobian(
      const std::shared_ptr<ControlParametrizationDataAbstract>& data,
      const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXs> out,
      const AssignmentOp = setto) const override;

  /**
   * @brief Compute the product between the transposed Jacobian of the control
   * (with respect to the parameters) and a specified matrix
   *
   * It assumes that `calc()` has been run first
   *
   * @param[in]  data   Control-parametrization data
   * @param[in]  A      A matrix to multiply times the Jacobian
   * @param[out] out    Product between the transposed Jacobian of the control
   * with respect to the parameters and the matrix A
   * @param[in] op      Assignment operator which sets, adds, or removes the
   * given results
   */
  virtual void multiplyJacobianTransposeBy(
      const std::shared_ptr<ControlParametrizationDataAbstract>& data,
      const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXs> out,
      const AssignmentOp = setto) const override;

  template <typename NewScalar>
  ControlParametrizationModelPolyZeroTpl<NewScalar> cast() const;

  virtual void print(std::ostream& os) const override;

 protected:
  using Base::nu_;
  using Base::nw_;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/controls/poly-zero.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::ControlParametrizationModelPolyZeroTpl)

#endif  // CROCODDYL_CORE_CONTROLS_POLY_ZERO_HPP_
