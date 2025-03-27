///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, University of Edinburgh, University of Trento,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_CONTROL_BASE_HPP_
#define CROCODDYL_CORE_CONTROL_BASE_HPP_

#include "crocoddyl/core/fwd.hpp"

namespace crocoddyl {

class ControlParametrizationModelBase {
 public:
  virtual ~ControlParametrizationModelBase() = default;

  CROCODDYL_BASE_CAST(ControlParametrizationModelBase,
                      ControlParametrizationModelAbstractTpl)
};

/**
 * @brief Abstract class for the control trajectory parametrization
 *
 * The control trajectory is a function of the (normalized) time.
 * Normalized time is between 0 and 1, where 0 represents the beginning of the
 * time step, and 1 represents its end. The trajectory depends on the control
 * parameters u, whose size may be larger than the size of the control inputs w.
 *
 * The main computations are carried out in `calc`, `multiplyByJacobian` and
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
 * \sa `calc()`, `calcDiff()`, `createData()`, `params`, `multiplyByJacobian`,
 * `multiplyJacobianTransposeBy`
 */
template <typename _Scalar>
class ControlParametrizationModelAbstractTpl
    : public ControlParametrizationModelBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ControlParametrizationDataAbstractTpl<Scalar>
      ControlParametrizationDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the control dimensions
   *
   * @param[in] nw   Dimension of control inputs
   * @param[in] nu   Dimension of control parameters
   */
  ControlParametrizationModelAbstractTpl(const std::size_t nw,
                                         const std::size_t nu);
  virtual ~ControlParametrizationModelAbstractTpl() = default;

  /**
   * @brief Get the value of the control at the specified time
   *
   * @param[in]  data   Data structure containing the control vector to write
   * @param[in]  t      Time in [0,1]
   * @param[in]  u      Control parameters
   */
  virtual void calc(
      const std::shared_ptr<ControlParametrizationDataAbstract>& data,
      const Scalar t, const Eigen::Ref<const VectorXs>& u) const = 0;

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
      const Scalar t, const Eigen::Ref<const VectorXs>& u) const = 0;

  /**
   * @brief Create the control-parametrization data
   *
   * @return the control-parametrization data
   */
  virtual std::shared_ptr<ControlParametrizationDataAbstract> createData();

  /**
   * @brief Update the control parameters u for a specified time t given the
   * control input w
   *
   * @param[in]  data   Control-parametrization data
   * @param[in]  t      Time in [0,1]
   * @param[in]  w      Control inputs
   */
  virtual void params(
      const std::shared_ptr<ControlParametrizationDataAbstract>& data,
      const Scalar t, const Eigen::Ref<const VectorXs>& w) const = 0;

  /**
   * @brief Convert the bounds on the control inputs w to bounds on the control
   * parameters u
   *
   * @param[in]  w_lb   Control lower bound
   * @param[in]  w_ub   Control lower bound
   * @param[out] u_lb   Control parameters lower bound
   * @param[out] u_ub   Control parameters upper bound
   */
  virtual void convertBounds(const Eigen::Ref<const VectorXs>& w_lb,
                             const Eigen::Ref<const VectorXs>& w_ub,
                             Eigen::Ref<VectorXs> u_lb,
                             Eigen::Ref<VectorXs> u_ub) const = 0;

  /**
   * @brief Compute the product between the given matrix A and the derivative of
   * the control input with respect to the control parameters (i.e., A*dw_du).
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
      const AssignmentOp op = setto) const = 0;

  virtual MatrixXs multiplyByJacobian_J(
      const std::shared_ptr<ControlParametrizationDataAbstract>& data,
      const Eigen::Ref<const MatrixXs>& A, const AssignmentOp op = setto) const;

  /**
   * @brief Compute the product between the transpose of the derivative of the
   * control input with respect to the control parameters and a given matrix A
   * (i.e., dw_du^T*A)
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
      const AssignmentOp op = setto) const = 0;

  virtual MatrixXs multiplyJacobianTransposeBy_J(
      const std::shared_ptr<ControlParametrizationDataAbstract>& data,
      const Eigen::Ref<const MatrixXs>& A, const AssignmentOp op = setto) const;

  /**
   * @brief Checks that a specific data belongs to this model
   */
  virtual bool checkData(
      const std::shared_ptr<ControlParametrizationDataAbstract>& data);

  /**
   * @brief Print information on the control model
   */
  template <class Scalar>
  friend std::ostream& operator<<(
      std::ostream& os,
      const ControlParametrizationModelAbstractTpl<Scalar>& model);

  /**
   * @brief Print relevant information of the control model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

  /**
   * @brief Return the dimension of the control inputs
   */
  std::size_t get_nw() const;

  /**
   * @brief Return the dimension of control parameters
   */
  std::size_t get_nu() const;

 protected:
  std::size_t nw_;  //!< Control dimension
  std::size_t nu_;  //!< Control parameters dimension
  ControlParametrizationModelAbstractTpl() : nw_(0), nu_(0) {};
};

template <typename _Scalar>
struct ControlParametrizationDataAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit ControlParametrizationDataAbstractTpl(Model<Scalar>* const model)
      : w(model->get_nw()),
        u(model->get_nu()),
        dw_du(model->get_nw(), model->get_nu()) {
    w.setZero();
    u.setZero();
    dw_du.setZero();
  }
  virtual ~ControlParametrizationDataAbstractTpl() = default;

  VectorXs w;      //!< value of the differential control
  VectorXs u;      //!< value of the control parameters
  MatrixXs dw_du;  //!< Jacobian of the differential control with respect to the
                   //!< parameters
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/control-base.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::ControlParametrizationModelAbstractTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::ControlParametrizationDataAbstractTpl)

#endif  // CROCODDYL_CORE_CONTROL_BASE_HPP_
