///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, University of Edinburgh, University of Trento,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_CONTROLS_POLY_TWO_RK_HPP_
#define CROCODDYL_CORE_CONTROLS_POLY_TWO_RK_HPP_

#include "crocoddyl/core/control-base.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/integrator/rk.hpp"

namespace crocoddyl {

/**
 * @brief A polynomial function of time of degree two, that is a quadratic
 * function
 *
 * The size of the parameters \f$\mathbf{u}\f$ is 3 times the size of the
 * control input \f$\mathbf{w}\f$. It defines a polynomial of degree two,
 * customized for the RK4 and RK4 integrators (even though it can be used with
 * whatever integration scheme). The first third of \f$\mathbf{u}\f$ represents
 * the value of \f$\mathbf{w}\f$ at time 0. The second third of \f$\mathbf{u}\f$
 * represents the value of \f$\mathbf{w}\f$ at time 0.5 or 1/3 for RK4 and RK3
 * parametrization, respectively. The last third of \f$\mathbf{u}\f$ represents
 * the value of \f$\mathbf{w}\f$ at time 1 or 2/3 for the RK4 and RK3
 * parametrization, respectively. This parametrization is suitable to be used
 * with the RK-4 or RK-3 integration schemes, because they require the value of
 * \f$\mathbf{w}\f$ exactly at 0, 0.5, 1 (for RK4) or 0, 1/3, 2/3 (for RK3).
 *
 * The main computations are carried out in `calc`, `multiplyByJacobian` and
 * `multiplyJacobianTransposeBy`, where the former computes control input
 * \f$\mathbf{w}\in\mathbb{R}^{nw}\f$ from a set of control parameters
 * \f$\mathbf{u}\in\mathbb{R}^{nu}\f$ where `nw` and `nu` represent the
 * dimension of the control inputs and parameters, respectively, and the latter
 * defines useful operations across the Jacobian of the control-parametrization
 * model. Finally, `params` allows us to obtain the control parameters from a
 * the control input, i.e., it is the inverse of `calc`. Note that
 * `multiplyByJacobian` and `multiplyJacobianTransposeBy` requires to run `calc`
 * first.
 *
 * \sa `ControlParametrizationAbstractTpl`, `calc()`, `calcDiff()`,
 * `createData()`, `params`, `multiplyByJacobian`, `multiplyJacobianTransposeBy`
 */
template <typename _Scalar>
class ControlParametrizationModelPolyTwoRKTpl
    : public ControlParametrizationModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ControlParametrizationModelBase,
                         ControlParametrizationModelPolyTwoRKTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ControlParametrizationDataAbstractTpl<Scalar>
      ControlParametrizationDataAbstract;
  typedef ControlParametrizationModelAbstractTpl<Scalar> Base;
  typedef ControlParametrizationDataPolyTwoRKTpl<Scalar> Data;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the poly-two RK control parametrization
   *
   * @param[in] nw      Dimension of control vector
   * @param[in] rktype  Type of RK parametrization
   */
  explicit ControlParametrizationModelPolyTwoRKTpl(const std::size_t nw,
                                                   const RKType rktype);
  virtual ~ControlParametrizationModelPolyTwoRKTpl() = default;

  /**
   * @brief Get the value of the control at the specified time
   *
   * @param[in]  data   Poly-two-RK data
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
   * @param[in]  data   Poly-two-RK data
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
   * @brief Get a value of the control parameters u such that the control at the
   * specified time t is equal to the specified value w
   *
   * @param[in]  data   Poly-two-RK data
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
   * @param[in]  data   Poly-two-RK data
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
   * @param[in]  data   Poly-two-RK data
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
  ControlParametrizationModelPolyTwoRKTpl<NewScalar> cast() const;

  virtual void print(std::ostream& os) const override;

 protected:
  using Base::nu_;
  using Base::nw_;

 private:
  RKType rktype_;
};

template <typename _Scalar>
struct ControlParametrizationDataPolyTwoRKTpl
    : public ControlParametrizationDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ControlParametrizationDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::Vector3s Vector3s;

  template <template <typename Scalar> class Model>
  explicit ControlParametrizationDataPolyTwoRKTpl(Model<Scalar>* const model)
      : Base(model), tmp_t2(0.) {
    c.setZero();
  }
  virtual ~ControlParametrizationDataPolyTwoRKTpl() = default;

  Vector3s c;     //!< Polynomial coefficients of the second-order control model
                  //!< that depends on time
  Scalar tmp_t2;  //!< Temporary variable to store the square of the time
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/controls/poly-two-rk.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::ControlParametrizationModelPolyTwoRKTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::ControlParametrizationDataPolyTwoRKTpl)

#endif  // CROCODDYL_CORE_CONTROLS_POLY_TWO_RK_HPP_
