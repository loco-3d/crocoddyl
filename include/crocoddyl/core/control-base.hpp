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
#include <boost/shared_ptr.hpp>

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/mathbase.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief Abstract class for the control trajectory parametrization
 *
 * The control trajectory is a function of the (normalized) time.
 * Normalized time is between 0 and 1, where 0 represents the beginning of the time step,
 * and 1 represents its end.
 * The trajectory depends on the control parameters p, whose size may be larger than the
 * size of the control inputs u.
 */
template <typename _Scalar>
class ControlParametrizationModelAbstractTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ControlParametrizationDataAbstractTpl<Scalar> ControlParametrizationDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the control dimensions
   *
   * @param[in] nw   Dimension of differential control
   * @param[in] nu   Dimension of control parameters
   */
  ControlParametrizationModelAbstractTpl(const std::size_t nw, const std::size_t nu);
  virtual ~ControlParametrizationModelAbstractTpl();

  /**
   * @brief Get the value of the control at the specified time
   *
   * @param[in]  data   Data structure containing the control vector to write
   * @param[in]  t      Time in [0,1]
   * @param[in]  u      Control parameters
   */
  virtual void calc(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double t,
                    const Eigen::Ref<const VectorXs>& u) const = 0;

  /**
   * @brief Get the value of the Jacobian of the control with respect to the parameters
   *
   * @param[in]  data   Data structure containing the Jacobian matrix to write
   * @param[in]  t      Time in [0,1]
   * @param[in]  u      Control parameters
   */
  virtual void calcDiff(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double t,
                        const Eigen::Ref<const VectorXs>& u) const = 0;

  /**
   * @brief Create the action data
   *
   * @return the action data
   */
  virtual boost::shared_ptr<ControlParametrizationDataAbstract> createData();

  /**
   * @brief Get a value of the control parameters such that the control at the specified time
   * t is equal to the specified value u
   *
   * @param[in]  data   Data structure containing the control parameters vector to write
   * @param[in]  t      Time in [0,1]
   * @param[in]  w      Control values
   */
  virtual void params(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double t,
                      const Eigen::Ref<const VectorXs>& w) const = 0;

  /**
   * @brief Map the specified bounds from the control space to the parameter space
   *
   * @param[in]  w_lb   Control lower bound
   * @param[in]  w_ub   Control lower bound
   * @param[out] u_lb   Control parameters lower bound
   * @param[out] u_ub   Control parameters upper bound
   */
  virtual void convertBounds(const Eigen::Ref<const VectorXs>& w_lb, const Eigen::Ref<const VectorXs>& w_ub,
                             Eigen::Ref<VectorXs> u_lb, Eigen::Ref<VectorXs> u_ub) const = 0;


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
                                  Eigen::Ref<MatrixXs> out) const = 0;

  virtual MatrixXs multiplyByJacobian_J(double t, const Eigen::Ref<const VectorXs>& u,
                                        const Eigen::Ref<const MatrixXs>& A) const;

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
                                           const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXs> out) const = 0;

  virtual MatrixXs multiplyJacobianTransposeBy_J(double t, const Eigen::Ref<const VectorXs>& p,
                                                 const Eigen::Ref<const MatrixXs>& A) const;

  /**
   * @brief Checks that a specific data belongs to this model
   */
  virtual bool checkData(const boost::shared_ptr<ControlParametrizationDataAbstract>& data);

  /**
   * @brief Return the dimension of the control value
   */
  std::size_t get_nw() const;

  /**
   * @brief Return the dimension of control parameters
   */
  std::size_t get_nu() const;

 protected:
  std::size_t nw_;  //!< Control dimension
  std::size_t nu_;  //!< Control parameters dimension
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
      : w(model->get_nw()), u(model->get_nu()), dw_du(model->get_nw(), model->get_nu()) {
    w.setZero();
    u.setZero();
    dw_du.setZero();
  }
  virtual ~ControlParametrizationDataAbstractTpl() {}

  VectorXs w;      //!< value of the differential control
  VectorXs u;      //!< value of the control parameters
  MatrixXs dw_du;  //!< Jacobian of the differential control with respect to the parameters
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/control-base.hxx"

#endif  // CROCODDYL_CORE_CONTROL_BASE_HPP_
