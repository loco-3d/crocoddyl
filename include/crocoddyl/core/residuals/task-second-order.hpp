///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_RESIDUALS_TASK_SECOND_ORDER_HPP_
#define CROCODDYL_CORE_RESIDUALS_TASK_SECOND_ORDER_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/guidance-base.hpp"
#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/core/task-base.hpp"

namespace crocoddyl {

/**
 * @brief Second-order task residual for a generic task and guidance model.
 *
 * The running residual is
 * \f[
 *   r = a(x, u) + K\big(v(x, u) - g(y(x, u))\big),
 * \f]
 * where \f$a\f$ is the task acceleration, \f$v\f$ is the task rate,
 * \f$g=\phi(y)\f$ is a guidance model, and \f$K\f$ is a task gain.
 *
 * The gain can be supplied as a full matrix, a diagonal vector, or a scalar
 * isotropic gain.
 *
 * The residual Jacobians are
 * \f[
 *   R_x = A_x + K\big(V_x - G_e Y_x\big), \qquad
 *   R_u = A_u + K\big(V_u - G_e Y_u\big),
 * \f]
 * where \f$G_e = \partial g / \partial y\f$.
 */
template <typename _Scalar>
class ResidualModelTaskSecondOrderTpl
    : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ResidualModelBase, ResidualModelTaskSecondOrderTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataTaskSecondOrderTpl<Scalar> Data;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef GuidanceModelAbstractTpl<Scalar> GuidanceModelAbstract;
  typedef TaskModelAbstractTpl<Scalar> TaskModelAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the second-order task residual with a full gain.
   *
   * The residual is
   * \f[
   *   r = a + K(v - g(y)),
   * \f]
   * where \f$a\f$ is the task acceleration, \f$v\f$ is the task rate, and
   * \f$g(y)\f$ is the guidance output.
   *
   * The residual Jacobians are
   * \f[
   *   R_x = A_x + K\big(V_x - G_e Y_x\big), \qquad
   *   R_u = A_u + K\big(V_u - G_e Y_u\big),
   * \f]
   * where \f$G_e = \partial g / \partial y\f$.
   *
   * @param[in] task      Task model
   * @param[in] guidance  Guidance model
   * @param[in] gain      Full task gain matrix
   */
  ResidualModelTaskSecondOrderTpl(
      std::shared_ptr<TaskModelAbstract> task,
      std::shared_ptr<GuidanceModelAbstract> guidance, const MatrixXs& gain);
  /**
   * @brief Initialize the second-order task residual with a diagonal gain.
   *
   * The diagonal entries are used to build the full gain matrix.
   *
   * @param[in] task            Task model
   * @param[in] guidance        Guidance model
   * @param[in] diagonal_gain   Diagonal task gain
   */
  ResidualModelTaskSecondOrderTpl(
      std::shared_ptr<TaskModelAbstract> task,
      std::shared_ptr<GuidanceModelAbstract> guidance,
      const VectorXs& diagonal_gain);
  /**
   * @brief Initialize the second-order task residual with a scalar gain.
   *
   * The gain is expanded to a scalar multiple of the identity matrix.
   *
   * @param[in] task     Task model
   * @param[in] guidance Guidance model
   * @param[in] gain     Isotropic task gain
   */
  ResidualModelTaskSecondOrderTpl(
      std::shared_ptr<TaskModelAbstract> task,
      std::shared_ptr<GuidanceModelAbstract> guidance, const Scalar& gain);
  virtual ~ResidualModelTaskSecondOrderTpl() = default;

  /**
   * @brief Compute the second-order task residual.
   *
   * This evaluates
   * \f[
   *   r = a + K(v - g(y)).
   * \f]
   */
  virtual void calc(const std::shared_ptr<ResidualDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Compute the residual Jacobians.
   *
   * This evaluates
   * \f[
   *   R_x = A_x + K\big(V_x - G_e Y_x\big), \qquad
   *   R_u = A_u + K\big(V_u - G_e Y_u\big).
   * \f]
   */
  virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Allocate data for the residual.
   */
  virtual std::shared_ptr<ResidualDataAbstract> createData(
      DataCollectorAbstract* const data) override;

  /**
   * @brief Cast the residual model to a different scalar type.
   */
  template <typename NewScalar>
  ResidualModelTaskSecondOrderTpl<NewScalar> cast() const;

  const std::shared_ptr<TaskModelAbstract>& get_task() const;
  const std::shared_ptr<GuidanceModelAbstract>& get_guidance() const;
  const MatrixXs& get_gain() const;

  void set_guidance(std::shared_ptr<GuidanceModelAbstract> guidance);
  void set_gain(const MatrixXs& gain);

  virtual void print(std::ostream& os) const override;

 protected:
  using Base::nr_;
  using Base::nu_;
  using Base::q_dependent_;
  using Base::state_;
  using Base::u_dependent_;
  using Base::v_dependent_;

  void checkDimensions() const;

  std::shared_ptr<TaskModelAbstract> task_;
  std::shared_ptr<GuidanceModelAbstract> guidance_;
  MatrixXs gain_;
};

/**
 * @brief Data for the second-order task residual.
 *
 * The task data stores \f$y\f$, \f$v\f$, and \f$a\f$. The guidance data
 * stores \f$g\f$ and \f$G_e\f$. The temporary matrices cache
 * \f$V_x - G_e Y_x\f$ and \f$V_u - G_e Y_u\f$ for the Jacobian computation.
 */
template <typename _Scalar>
struct ResidualDataTaskSecondOrderTpl
    : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef GuidanceDataAbstractTpl<Scalar> GuidanceDataAbstract;
  typedef TaskDataAbstractTpl<Scalar> TaskDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Construct the residual data.
   *
   * The task, guidance, and temporary buffers are allocated from the model
   * and shared data.
   */
  template <template <typename Scalar> class Model>
  ResidualDataTaskSecondOrderTpl(Model<Scalar>* const model,
                                 DataCollectorAbstract* const data)
      : Base(model, data),
        task(model->get_task()->createData(data)),
        guidance(model->get_guidance()->createData()),
        v_error(model->get_nr()),
        v_error_x(model->get_nr(), model->get_state()->get_ndx()),
        v_error_u(model->get_nr(), model->get_nu()) {
    task->compute_acceleration = true;
    v_error.setZero();
    v_error_x.setZero();
    v_error_u.setZero();
  }

  virtual ~ResidualDataTaskSecondOrderTpl() = default;

  std::shared_ptr<TaskDataAbstract> task;
  std::shared_ptr<GuidanceDataAbstract> guidance;
  VectorXs v_error;
  MatrixXs v_error_x;
  MatrixXs v_error_u;

  using Base::Arr_Ru;
  using Base::Arr_Rx;
  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/residuals/task-second-order.hxx"

#endif  // CROCODDYL_CORE_RESIDUALS_TASK_SECOND_ORDER_HPP_
