///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, University of Edinburgh, LAAS-CNRS,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_NUMDIFF_CONTROL_HPP_
#define CROCODDYL_CORE_NUMDIFF_CONTROL_HPP_

#include "crocoddyl/core/control-base.hpp"
#include "crocoddyl/core/fwd.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ControlParametrizationModelNumDiffTpl
    : public ControlParametrizationModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ControlParametrizationModelBase,
                         ControlParametrizationModelNumDiffTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ControlParametrizationModelAbstractTpl<_Scalar> Base;
  typedef ControlParametrizationDataNumDiffTpl<_Scalar> Data;
  typedef ControlParametrizationDataAbstractTpl<_Scalar>
      ControlParametrizationDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Construct a new ControlParametrizationModelNumDiff object
   *
   * @param model
   */
  explicit ControlParametrizationModelNumDiffTpl(std::shared_ptr<Base> model);
  virtual ~ControlParametrizationModelNumDiffTpl() = default;

  /**
   * @brief Get the value of the control at the specified time
   *
   * @param[in]  data   Control-parametrization numdiff data
   * @param[in]  t      Time in [0,1]
   * @param[in]  u      Control parameters
   */
  void calc(const std::shared_ptr<ControlParametrizationDataAbstract>& data,
            const Scalar t, const Eigen::Ref<const VectorXs>& u) const override;

  /**
   * @brief Get the value of the Jacobian of the control with respect to the
   * parameters
   *
   * @param[in]  data   Control-parametrization numdiff data
   * @param[in]  t      Time in [0,1]
   * @param[in]  u      Control parameters
   */
  void calcDiff(const std::shared_ptr<ControlParametrizationDataAbstract>& data,
                const Scalar t,
                const Eigen::Ref<const VectorXs>& u) const override;

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
   * @param[in]  data   Control-parametrization numdiff data
   * @param[in]  t      Time in [0,1]
   * @param[in]  w      Control values
   */
  void params(const std::shared_ptr<ControlParametrizationDataAbstract>& data,
              const Scalar t,
              const Eigen::Ref<const VectorXs>& w) const override;

  /**
   * @brief Convert the bounds on the control to bounds on the control
   * parameters
   *
   * @param[in]  w_lb   Control lower bound
   * @param[in]  w_ub   Control upper bound
   * @param[in]  u_lb   Control parameter lower bound
   * @param[in]  u_ub   Control parameter upper bound
   */
  void convertBounds(const Eigen::Ref<const VectorXs>& w_lb,
                     const Eigen::Ref<const VectorXs>& w_ub,
                     Eigen::Ref<VectorXs> u_lb,
                     Eigen::Ref<VectorXs> u_ub) const override;

  /**
   * @brief Compute the product between a specified matrix and the Jacobian of
   * the control (with respect to the parameters)
   *
   * @param[in]  data   Control-parametrization numdiff data
   * @param[in]  A      A matrix to multiply times the Jacobian
   * @param[out] out    Product between the matrix A and the Jacobian of the
   * control with respect to the parameters
   * @param[in] op      Assignment operator which sets, adds, or removes the
   * given results
   */
  void multiplyByJacobian(
      const std::shared_ptr<ControlParametrizationDataAbstract>& data,
      const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXs> out,
      const AssignmentOp = setto) const override;

  /**
   * @brief Compute the product between the transposed Jacobian of the control
   * (with respect to the parameters) and a specified matrix
   *
   * @param[in]  data   Control-parametrization numdiff data
   * @param[in]  A      A matrix to multiply times the Jacobian
   * @param[out] out    Product between the transposed Jacobian of the control
   * with respect to the parameters and the matrix A
   * @param[in] op      Assignment operator which sets, adds, or removes the
   * given results
   */
  void multiplyJacobianTransposeBy(
      const std::shared_ptr<ControlParametrizationDataAbstract>& data,
      const Eigen::Ref<const MatrixXs>& A, Eigen::Ref<MatrixXs> out,
      const AssignmentOp = setto) const override;

  template <typename NewScalar>
  ControlParametrizationModelNumDiffTpl<NewScalar> cast() const;
  /**
   * @brief Get the model_ object
   *
   * @return Base&
   */
  const std::shared_ptr<Base>& get_model() const;

  /**
   * @brief Return the disturbance constant used in the numerical
   * differentiation routine
   */
  const Scalar get_disturbance() const;

  /**
   * @brief Modify the disturbance constant used in the numerical
   * differentiation routine
   */
  void set_disturbance(const Scalar disturbance);

 private:
  std::shared_ptr<Base>
      model_;     //!< model we need to compute the numerical differentiation
  Scalar e_jac_;  //!< Constant used for computing disturbances in Jacobian
                  //!< calculation

 protected:
  using Base::nu_;
  using Base::nw_;
};

template <typename _Scalar>
struct ControlParametrizationDataNumDiffTpl
    : public ControlParametrizationDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef ControlParametrizationDataAbstractTpl<Scalar> Base;

  template <template <typename Scalar> class Model>
  explicit ControlParametrizationDataNumDiffTpl(Model<Scalar>* const model)
      : Base(model), du(model->get_nu()) {
    du.setZero();

    const std::size_t nu = model->get_model()->get_nu();
    data_0 = model->get_model()->createData();
    for (std::size_t i = 0; i < nu; ++i) {
      data_u.push_back(model->get_model()->createData());
    }
  }

  virtual ~ControlParametrizationDataNumDiffTpl() {}

  VectorXs du;  //!< temporary variable used for finite differencing
  std::shared_ptr<Base> data_0;  //!< The data that contains the final results
  std::vector<std::shared_ptr<Base> >
      data_u;  //!< The temporary data associated with the control variation
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/numdiff/control.hxx"

#endif  // CROCODDYL_CORE_NUMDIFF_CONTROL_HPP_
