///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_RESIDUALS_TASK_FIRST_ORDER_HPP_
#define CROCODDYL_CORE_RESIDUALS_TASK_FIRST_ORDER_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/guidance-base.hpp"
#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/core/task-base.hpp"

namespace crocoddyl {

/**
 * @brief First-order task residual for a generic task and guidance model.
 *
 * The running residual is
 * \f[
 *   r = v(x, u) - g(y(x, u)),
 * \f]
 * where \f$y\f$ is the task quantity, \f$v\f$ is the task rate, and
 * \f$g=\phi(y)\f$ is a guidance model.
 *
 * This class deliberately keeps the abstraction narrow: the task model owns
 * the task geometry and its derivatives, while the guidance model owns the
 * desired-rate profile.
 *
 * The residual Jacobians are
 * \f[
 *   R_x = V_x - G_e Y_x, \qquad R_u = V_u - G_e Y_u,
 * \f]
 * where \f$G_e = \partial g / \partial y\f$.
 */
template <typename _Scalar>
class ResidualModelTaskFirstOrderTpl
    : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ResidualModelBase, ResidualModelTaskFirstOrderTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataTaskFirstOrderTpl<Scalar> Data;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef GuidanceModelAbstractTpl<Scalar> GuidanceModelAbstract;
  typedef TaskModelAbstractTpl<Scalar> TaskModelAbstract;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize the first-order task residual.
   *
   * The residual is
   * \f[
   *   r = v - g(y),
   * \f]
   * where \f$y\f$ is the task quantity, \f$v\f$ is the task rate, and
   * \f$g(y)\f$ is the guidance output.
   *
   * The residual Jacobians are
   * \f[
   *   R_x = V_x - G_e Y_x, \qquad R_u = V_u - G_e Y_u,
   * \f]
   * where \f$G_e = \partial g / \partial y\f$.
   *
   * @param[in] task      Task model
   * @param[in] guidance  Guidance model
   */
  ResidualModelTaskFirstOrderTpl(
      std::shared_ptr<TaskModelAbstract> task,
      std::shared_ptr<GuidanceModelAbstract> guidance);
  virtual ~ResidualModelTaskFirstOrderTpl() = default;

  /**
   * @brief Compute the first-order task residual.
   *
   * This evaluates
   * \f[
   *   r = v - g(y).
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
   *   R_x = V_x - G_e Y_x, \qquad R_u = V_u - G_e Y_u.
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
  ResidualModelTaskFirstOrderTpl<NewScalar> cast() const;

  const std::shared_ptr<TaskModelAbstract>& get_task() const;
  const std::shared_ptr<GuidanceModelAbstract>& get_guidance() const;
  void set_guidance(std::shared_ptr<GuidanceModelAbstract> guidance);

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
};

/**
 * @brief Data for the first-order task residual.
 *
 * The task data stores \f$y\f$, \f$v\f$, and the task Jacobians. The
 * guidance data stores \f$g\f$ and \f$G_e\f$. The temporary matrices cache
 * \f$V_x - G_e Y_x\f$ and \f$V_u - G_e Y_u\f$ for the Jacobian computation.
 */
template <typename _Scalar>
struct ResidualDataTaskFirstOrderTpl : public ResidualDataAbstractTpl<_Scalar> {
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
   * The task and guidance data are allocated from the model and shared data.
   */
  template <template <typename Scalar> class Model>
  ResidualDataTaskFirstOrderTpl(Model<Scalar>* const model,
                                DataCollectorAbstract* const data)
      : Base(model, data),
        task(model->get_task()->createData(data)),
        guidance(model->get_guidance()->createData()) {
    task->compute_acceleration = false;
  }

  virtual ~ResidualDataTaskFirstOrderTpl() = default;

  std::shared_ptr<TaskDataAbstract> task;
  std::shared_ptr<GuidanceDataAbstract> guidance;

  using Base::Arr_Ru;
  using Base::Arr_Rx;
  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/residuals/task-first-order.hxx"

#endif  // CROCODDYL_CORE_RESIDUALS_TASK_FIRST_ORDER_HPP_
