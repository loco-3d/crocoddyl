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
  typedef ControlParametrizationDataNumDiffTpl<_Scalar> Data;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  explicit ControlParametrizationModelNumDiffTpl(boost::shared_ptr<Base> state);
  virtual ~ControlParametrizationModelNumDiffTpl();

  virtual boost::shared_ptr<ControlParametrizationDataAbstract> createData();

  /**
   * @brief Get the value of the control at the specified time
   *
   * @param[in]  data   Data structure containing the control vector to write
   * @param[in]  t      Time in [0,1]
   * @param[in]  p      Control parameters
   */
  void calc(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double t,
            const Eigen::Ref<const VectorXs>& p) const;

  /**
   * @brief Get a value of the control parameters such that the control at the specified time
   * t is equal to the specified value u
   *
   * @param[in]  data   Data structure containing the control parameters vector to write
   * @param[in]  t      Time in [0,1]
   * @param[in]  u      Control values
   */
  void params(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double t,
              const Eigen::Ref<const VectorXs>& u) const;

  /**
   * @brief Convert the bounds on the control to bounds on the control parameters
   *
   * @param[in]  u_lb   Control lower bound
   * @param[in]  u_ub   Control upper bound
   * @param[in]  p_lb   Control parameter lower bound
   * @param[in]  p_ub   Control parameter upper bound
   */
  void convertBounds(const Eigen::Ref<const VectorXs>& u_lb, const Eigen::Ref<const VectorXs>& u_ub,
                      Eigen::Ref<VectorXs> p_lb, Eigen::Ref<VectorXs> p_ub) const;

  /**
   * @brief Get the value of the Jacobian of the control with respect to the parameters
   *
   * @param[in]  data   Data structure containing the Jacobian matrix to write
   * @param[in]  t      Time in [0,1]
   * @param[in]  p      Control parameters
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

  boost::shared_ptr<ControlParametrizationDataAbstract> data0_;         //!< Data used in methods multiplyByJacobian
  boost::shared_ptr<ControlParametrizationDataAbstract> dataCalcDiff_;  //!< Data used for finite differences in calcDiff
  boost::shared_ptr<ControlParametrizationDataNumDiff>  dataNumDiff_;   //!< Data used for storing dp

  /**
   * @brief This the increment used in the finite differentiation and integration.
   */
  Scalar disturbance_;

 protected:
  using Base::np_;
  using Base::nw_;
};

template <typename _Scalar>
struct ControlParametrizationDataNumDiffTpl: public ControlParametrizationDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef ControlParametrizationDataAbstractTpl<Scalar> Base;

  template <template <typename Scalar> class Model>
  explicit ControlParametrizationDataNumDiffTpl(Model<Scalar>* const model)
      : Base(model) {
    dp = VectorXs::Zero(model->get_np());
  }

  virtual ~ControlParametrizationDataNumDiffTpl() {}

  VectorXs dp;        //!< temporary variable used for finite differencing
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/numdiff/control.hxx"

#endif  // CROCODDYL_CORE_NUMDIFF_CONTROL_HPP_
