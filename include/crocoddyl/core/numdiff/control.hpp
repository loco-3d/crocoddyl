///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, New York University, Max Planck Gesellschaft,
//                     University of Edinburgh, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_NUMDIFF_CONTROL_HPP_
#define CROCODDYL_CORE_NUMDIFF_CONTROL_HPP_

#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/control-base.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ControlParametrizationModelNumDiffTpl : public ControlParametrizationModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ControlParametrizationModelAbstractTpl<_Scalar> Base;
  typedef ControlParametrizationDataAbstractTpl<_Scalar> ControlParametrizationDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  explicit ControlParametrizationModelNumDiffTpl(boost::shared_ptr<Base> state);
  virtual ~ControlParametrizationModelNumDiffTpl();

  void resize(const std::size_t nu);

  /**
   * @brief Get the value of the control at the specified time
   *
   * @param[in]  t      Time
   * @param[in]  p      Control parameters
   * @param[out] u_out  Control value
   */
  void calc(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double t,
            const Eigen::Ref<const VectorXs>& p) const;

  void params(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double t,
              const Eigen::Ref<const VectorXs>& u) const;

  void convert_bounds(const Eigen::Ref<const VectorXs>& u_lb, const Eigen::Ref<const VectorXs>& u_ub,
                      Eigen::Ref<VectorXs> p_lb, Eigen::Ref<VectorXs> p_ub) const;

  /**
   * @brief Get the value of the Jacobian of the control with respect to the parameters
   *
   * @param[in]  t      Time
   * @param[in]  p      Control parameters
   * @param[out] J_out  Jacobian of the control value with respect to the parameters
   */
  void calcDiff(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double t,
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
  void multiplyByJacobian(double t, const Eigen::Ref<const VectorXs>& p, const Eigen::Ref<const MatrixXs>& A,
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
  void multiplyJacobianTransposeBy(double t, const Eigen::Ref<const VectorXs>& p, const Eigen::Ref<const MatrixXs>& A,
                                   Eigen::Ref<MatrixXs> out) const;

  const Scalar get_disturbance() const;
  void set_disturbance(const Scalar disturbance);

 private:
  /**
   * @brief This is the control we need to compute the numerical differentiation
   * from.
   */
  boost::shared_ptr<Base> control_;

  boost::shared_ptr<ControlParametrizationDataAbstract> data_;

  /**
   * @brief This the increment used in the finite differentiation and integration.
   */
  Scalar disturbance_;

 protected:
  using Base::np_;
  using Base::nu_;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/numdiff/control.hxx"

#endif  // CROCODDYL_CORE_NUMDIFF_CONTROL_HPP_
