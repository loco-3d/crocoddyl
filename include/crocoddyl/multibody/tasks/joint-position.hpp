///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_TASKS_JOINT_POSITION_HPP_
#define CROCODDYL_MULTIBODY_TASKS_JOINT_POSITION_HPP_

#include "crocoddyl/core/data/joint.hpp"
#include "crocoddyl/core/task-base.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

/**
 * @brief Joint-position task model.
 *
 * Let
 * \f[
 *   \operatorname{diff}(x_{\mathrm{ref}},x)=
 *   \begin{bmatrix}e_q\\e_v\end{bmatrix}.
 * \f]
 * The task defines
 * \f[
 *   y=e_q, \qquad v=e_v, \qquad
 *   a=\ddot q-\ddot q_{\mathrm{ref}}.
 * \f]
 * Combined with proportional and derivative convergence gains, this yields
 * \f$\ddot q-\ddot q_{\mathrm{ref}}+K_d e_v+K_p e_q\f$.
 *
 * The generalized acceleration is read from shared joint data when available.
 * Otherwise, it is read from the shared Pinocchio `ddq` cache. Acceleration
 * derivatives with respect to the action-model variables require shared joint
 * data.
 */
template <typename _Scalar>
class TaskModelJointPositionTpl : public TaskModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(TaskModelBase, TaskModelJointPositionTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef TaskModelAbstractTpl<Scalar> Base;
  typedef TaskDataJointPositionTpl<Scalar> Data;
  typedef TaskDataAbstractTpl<Scalar> TaskDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize with state and acceleration references.
   *
   * @param[in] state  State of the multibody system
   * @param[in] xref   Reference state containing configuration and velocity
   * @param[in] aref   Reference generalized acceleration
   * @param[in] nu     Dimension of the control vector
   */
  TaskModelJointPositionTpl(std::shared_ptr<StateMultibody> state,
                            const VectorXs& xref, const VectorXs& aref,
                            const std::size_t nu);

  /**
   * @brief Initialize with state and acceleration references.
   *
   * The default control dimension is state.nv.
   *
   * @param[in] state  State of the multibody system
   * @param[in] xref   Reference state containing configuration and velocity
   * @param[in] aref   Reference generalized acceleration
   */
  TaskModelJointPositionTpl(std::shared_ptr<StateMultibody> state,
                            const VectorXs& xref, const VectorXs& aref);

  /**
   * @brief Initialize with a state reference and zero acceleration reference.
   *
   * @param[in] state  State of the multibody system
   * @param[in] xref   Reference state containing configuration and velocity
   * @param[in] nu     Dimension of the control vector
   */
  TaskModelJointPositionTpl(std::shared_ptr<StateMultibody> state,
                            const VectorXs& xref, const std::size_t nu);

  /**
   * @brief Initialize with a state reference and zero acceleration reference.
   *
   * The default control dimension is state.nv.
   *
   * @param[in] state  State of the multibody system
   * @param[in] xref   Reference state containing configuration and velocity
   */
  TaskModelJointPositionTpl(std::shared_ptr<StateMultibody> state,
                            const VectorXs& xref);
  virtual ~TaskModelJointPositionTpl() = default;

  virtual void calc(const std::shared_ptr<TaskDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  virtual void calcDiff(const std::shared_ptr<TaskDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;

  virtual std::shared_ptr<TaskDataAbstract> createData(
      DataCollectorAbstract* const data) override;

  template <typename NewScalar>
  TaskModelJointPositionTpl<NewScalar> cast() const;

  const VectorXs& get_reference() const;
  const VectorXs& get_acceleration_reference() const;
  void set_reference(const VectorXs& xref);
  void set_acceleration_reference(const VectorXs& aref);

  virtual void print(std::ostream& os) const override;

 protected:
  using Base::nr_;
  using Base::nu_;
  using Base::state_;

 private:
  VectorXs xref_;
  VectorXs aref_;
};

/**
 * @brief Cached values and derivatives for the joint-position task.
 */
template <typename _Scalar>
struct TaskDataJointPositionTpl : public TaskDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef TaskDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Bind the task data to the available generalized-acceleration data.
   *
   * @param[in] model  Joint-position task model
   * @param[in] data   Shared data optionally derived from
   *                   DataCollectorJointTpl and/or DataCollectorMultibodyTpl
   */
  template <template <typename Scalar> class Model>
  TaskDataJointPositionTpl(Model<Scalar>* const model,
                           DataCollectorAbstract* const data)
      : Base(model, data),
        pinocchio(nullptr),
        dx(model->get_state()->get_ndx()),
        Jdiff(model->get_state()->get_ndx(), model->get_state()->get_ndx()) {
    dx.setZero();
    Jdiff.setZero();

    DataCollectorJointTpl<Scalar>* j =
        dynamic_cast<DataCollectorJointTpl<Scalar>*>(shared);
    if (j != nullptr) {
      joint = j->joint;
    }
    DataCollectorMultibodyTpl<Scalar>* d =
        dynamic_cast<DataCollectorMultibodyTpl<Scalar>*>(shared);
    if (d != nullptr) {
      pinocchio = d->pinocchio;
    }
  }

  virtual ~TaskDataJointPositionTpl() = default;

  pinocchio::DataTpl<Scalar>*
      pinocchio;  //!< Optional shared Pinocchio acceleration data
  std::shared_ptr<JointDataAbstractTpl<Scalar>>
      joint;       //!< Shared generalized-acceleration data
  VectorXs dx;     //!< State difference [configuration; velocity]
  MatrixXs Jdiff;  //!< Jacobian of the state difference w.r.t. current state

  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/tasks/joint-position.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::TaskModelJointPositionTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::TaskDataJointPositionTpl)

#endif  // CROCODDYL_MULTIBODY_TASKS_JOINT_POSITION_HPP_
