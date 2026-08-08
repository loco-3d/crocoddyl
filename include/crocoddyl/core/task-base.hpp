///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_TASK_BASE_HPP_
#define CROCODDYL_CORE_TASK_BASE_HPP_

#include "crocoddyl/core/data-collector-base.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/state-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief Abstract task model for convergence residuals.
 *
 * A task model supplies a same-dimensional quantity, rate, and acceleration:
 * \f[
 *   y(x,u), \qquad v(x,u)=\dot y, \qquad a(x,u)=\ddot y,
 * \f]
 * together with their Jacobians with respect to Crocoddyl's state tangent and
 * control coordinates.
 *
 * Concrete implementations own the task geometry and obtain cached robot
 * quantities from the shared DataCollectorAbstract passed to createData().
 * They must not launch a second dynamics pipeline inside calc() or calcDiff().
 *
 * A first-order model needs to populate y, v, Yx, Yu, Vx, and Vu. A
 * second-order model additionally populates a, Ax, and Au. Unused quantities
 * should remain zero.
 */
template <typename _Scalar>
class TaskModelAbstractTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef StateAbstractTpl<Scalar> StateAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef TaskDataAbstractTpl<Scalar> TaskDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  TaskModelAbstractTpl(std::shared_ptr<StateAbstract> state,
                       const std::size_t nr, const std::size_t nu,
                       const bool q_dependent = true,
                       const bool v_dependent = true,
                       const bool u_dependent = true);
  virtual ~TaskModelAbstractTpl() = default;

  /**
   * @brief Compute the task quantity, rate, and acceleration.
   */
  virtual void calc(const std::shared_ptr<TaskDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) = 0;

  /**
   * @brief Compute task Jacobians using values stored by calc().
   */
  virtual void calcDiff(const std::shared_ptr<TaskDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) = 0;

  /**
   * @brief Allocate task data and bind it to the action model's shared data.
   */
  virtual std::shared_ptr<TaskDataAbstract> createData(
      DataCollectorAbstract* const data);

  const std::shared_ptr<StateAbstract>& get_state() const;
  std::size_t get_nr() const;
  std::size_t get_nu() const;
  bool get_q_dependent() const;
  bool get_v_dependent() const;
  bool get_u_dependent() const;

  /**
   * @brief Print relevant information about the task model.
   */
  virtual void print(std::ostream& os) const;

 protected:
  std::shared_ptr<StateAbstract> state_;
  std::size_t nr_;
  std::size_t nu_;
  bool q_dependent_;
  bool v_dependent_;
  bool u_dependent_;
};

/**
 * @brief Preallocated values and derivatives produced by a task model.
 */
template <typename _Scalar>
struct TaskDataAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef TaskModelAbstractTpl<Scalar> TaskModelAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  TaskDataAbstractTpl(TaskModelAbstract* const model,
                      DataCollectorAbstract* const data)
      : shared(data),
        y(model->get_nr()),
        v(model->get_nr()),
        a(model->get_nr()),
        Yx(model->get_nr(), model->get_state()->get_ndx()),
        Yu(model->get_nr(), model->get_nu()),
        Vx(model->get_nr(), model->get_state()->get_ndx()),
        Vu(model->get_nr(), model->get_nu()),
        Ax(model->get_nr(), model->get_state()->get_ndx()),
        Au(model->get_nr(), model->get_nu()) {
    y.setZero();
    v.setZero();
    a.setZero();
    Yx.setZero();
    Yu.setZero();
    Vx.setZero();
    Vu.setZero();
    Ax.setZero();
    Au.setZero();
  }
  virtual ~TaskDataAbstractTpl() = default;

  DataCollectorAbstract* shared;

  VectorXs y;
  VectorXs v;
  VectorXs a;

  MatrixXs Yx;
  MatrixXs Yu;
  MatrixXs Vx;
  MatrixXs Vu;
  MatrixXs Ax;
  MatrixXs Au;
};

template <typename Scalar>
std::ostream& operator<<(std::ostream& os,
                         const TaskModelAbstractTpl<Scalar>& model);

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/task-base.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::TaskModelAbstractTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::TaskDataAbstractTpl)

#endif  // CROCODDYL_CORE_TASK_BASE_HPP_
