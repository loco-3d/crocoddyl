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
 * @brief Abstract class for the control trajectory parametrization
 * 
 * The control trajectory is a function of the (normalized) time.
 * Normalized time is between 0 and 1, where 0 represents the beginning of the time step, 
 * and 1 represents its end.
 * The trajectory depends on the control parameters p, whose size may be larger than the
 * size of the control inputs u.
 *
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
   * @param[in] nx   Dimension of control
   * @param[in] np   Dimension of control parameters
   */
  ControlParametrizationModelAbstractTpl(const std::size_t nu, const std::size_t np);
  virtual ~ControlParametrizationModelAbstractTpl();

  /**
   * @brief Create the action data
   *
   * @return the action data
   */
  virtual boost::shared_ptr<ControlParametrizationDataAbstract> createData();

  /**
   * @brief Checks that a specific data belongs to this model
   */
  virtual bool checkData(const boost::shared_ptr<ControlParametrizationDataAbstract>& data);

  /**
   * @brief Get the value of the control at the specified time
   *
   * @param[in]  t      Time
   * @param[in]  p      Control parameters
   * @param[out] u_out  Control value
   */
  virtual void calc(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double t, 
                    const Eigen::Ref<const VectorXs>& p) const = 0;

  // virtual VectorXs calc_u(double t, const Eigen::Ref<const VectorXs>& p) const;

  /**
   * @brief Get a value of the control parameters that results in the specified control value 
   * at the specified time
   *
   * @param[in]  t      Time
   * @param[in]  u      Control value
   * @param[out] p_out  Control parameters
   */
  virtual void params(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double t, 
                      const Eigen::Ref<const VectorXs>& u) const = 0;

  // virtual VectorXs params_p(double t, const Eigen::Ref<const VectorXs>& u) const;

  /**
   * @brief Map the specified bounds from the control space to the parameter space
   *
   * @param[in]  u_lb   Control lower bound
   * @param[in]  u_ub   Control lower bound
   * @param[out] p_lb   Control parameters lower bound
   * @param[out] p_ub   Control parameters upper bound
   */
  virtual void convert_bounds(const Eigen::Ref<const VectorXs>& u_lb, const Eigen::Ref<const VectorXs>& u_ub,
                              Eigen::Ref<VectorXs> p_lb, Eigen::Ref<VectorXs> p_ub) const = 0;

  /**
   * @brief Get the value of the Jacobian of the control with respect to the parameters
   *
   * @param[in]  t      Time
   * @param[in]  p      Control parameters
   * @param[out] J_out  Jacobian of the control value with respect to the parameters
   */
  virtual void calcDiff(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double t, 
                        const Eigen::Ref<const VectorXs>& p) const = 0;

  // virtual MatrixXs calcDiff_J(double t, const Eigen::Ref<const VectorXs>& p) const;

  /**
   * @brief Compute the product between a specified matrix and the Jacobian of the control (with respect to the parameters)
   *
   * @param[in]  t      Time
   * @param[in]  p      Control parameters
   * @param[in]  A      A matrix to multiply times the Jacobian
   * @param[out] out    Product between the matrix A and the Jacobian of the control with respect to the parameters
   */
  virtual void multiplyByJacobian(double t, const Eigen::Ref<const VectorXs>& p, 
        const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXs> out) const = 0;

  virtual MatrixXs multiplyByJacobian_J(double t, const Eigen::Ref<const VectorXs>& p, 
        const Eigen::Ref<const MatrixXs>& A) const;

  /**
   * @brief Compute the product between the transposed Jacobian of the control (with respect to the parameters) and
   * a specified matrix
   *
   * @param[in]  t      Time
   * @param[in]  p      Control parameters
   * @param[in]  A      A matrix to multiply times the Jacobian
   * @param[out] out    Product between the transposed Jacobian of the control with respect to the parameters and the matrix A
   */
  virtual void multiplyJacobianTransposeBy(double t, const Eigen::Ref<const VectorXs>& p, 
        const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXs> out) const = 0;

  virtual MatrixXs multiplyJacobianTransposeBy_J(double t, const Eigen::Ref<const VectorXs>& p, 
        const Eigen::Ref<const MatrixXs>& A) const;
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


template <typename _Scalar>
struct ControlParametrizationDataAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit ControlParametrizationDataAbstractTpl(Model<Scalar>* const model)
      : u_diff(model->get_nu()),
        u_params(model->get_np()),
        J(model->get_nu(), model->get_np()) 
  {
    u_diff.setZero();
    u_params.setZero();
    J.setZero();
  }
  virtual ~ControlParametrizationDataAbstractTpl() {}

  VectorXs u_diff;       //!< value of the control
  VectorXs u_params;     //!< value of the control parameters
  MatrixXs J;            //!< Jacobian of the control
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/control-base.hxx"

#endif  // CROCODDYL_CORE_CONTROL_BASE_HPP_
