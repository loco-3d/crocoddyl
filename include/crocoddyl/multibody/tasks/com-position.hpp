///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_TASKS_COM_POSITION_HPP_
#define CROCODDYL_MULTIBODY_TASKS_COM_POSITION_HPP_

#include <pinocchio/algorithm/center-of-mass-derivatives.hpp>
#include <pinocchio/algorithm/center-of-mass.hpp>

#include "crocoddyl/core/task-base.hpp"
#include "crocoddyl/core/utils/conversions.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

/**
 * @brief CoM position task model.
 *
 * The task quantity is the 3D center-of-mass tracking error
 * \f[
 *   y = c - c_\mathrm{ref},
 * \f]
 * where \f$c\f$ is the current center-of-mass position expressed in the world
 * frame and \f$c_\mathrm{ref}\f$ is the reference position.
 *
 * The task rate is the center-of-mass linear velocity
 * \f[
 *   v = \dot c.
 * \f]
 *
 * The acceleration term is the center-of-mass linear acceleration
 * \f[
 *   a = \ddot c.
 * \f]
 */
template <typename _Scalar>
class TaskModelCoMPositionTpl : public TaskModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(TaskModelBase, TaskModelCoMPositionTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef TaskModelAbstractTpl<Scalar> Base;
  typedef TaskDataAbstractTpl<Scalar> TaskDataAbstract;
  typedef TaskDataCoMPositionTpl<Scalar> Data;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef typename MathBase::Matrix3xs Matrix3xs;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Construct a CoM position task.
   */
  TaskModelCoMPositionTpl(std::shared_ptr<StateMultibody> state,
                          const Vector3s& cref, const std::size_t nu);

  /**
   * @brief Construct a CoM position task using the default control
   * dimension.
   */
  TaskModelCoMPositionTpl(std::shared_ptr<StateMultibody> state,
                          const Vector3s& cref);

  virtual ~TaskModelCoMPositionTpl() = default;

  /**
   * @brief Compute the task value and task rate.
   */
  virtual void calc(const std::shared_ptr<TaskDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Compute the task Jacobians.
   */
  virtual void calcDiff(const std::shared_ptr<TaskDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Allocate task data.
   */
  virtual std::shared_ptr<TaskDataAbstract> createData(
      DataCollectorAbstract* const data) override;

  /**
   * @brief Cast the task model to a different scalar type.
   */
  template <typename NewScalar>
  TaskModelCoMPositionTpl<NewScalar> cast() const;

  /** @brief Return the reference CoM position. */
  const Vector3s& get_reference() const;
  /** @brief Update the reference CoM position. */
  void set_reference(const Vector3s& cref);

  /** @brief Print relevant information. */
  virtual void print(std::ostream& os) const override;

 protected:
  using Base::nr_;
  using Base::nu_;
  using Base::q_dependent_;
  using Base::state_;
  using Base::u_dependent_;
  using Base::v_dependent_;

 private:
  Vector3s cref_;
  std::shared_ptr<typename StateMultibody::PinocchioModel> pin_model_;
};

/**
 * @brief Cached values and derivatives for the center-of-mass position task.
 */
template <typename _Scalar>
struct TaskDataCoMPositionTpl : public TaskDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef TaskDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::Matrix3xs Matrix3xs;
  typedef typename MathBase::Vector3s Vector3s;

  template <template <typename Scalar> class Model>
  TaskDataCoMPositionTpl(Model<Scalar>* const model,
                         DataCollectorAbstract* const data)
      : Base(model, data),
        com(Vector3s::Zero()),
        vcom(Vector3s::Zero()),
        acom(Vector3s::Zero()),
        dvc_dq(3, model->get_state()->get_nv()),
        dacom_dq(3, model->get_state()->get_nv()),
        dacom_dv(3, model->get_state()->get_nv()) {
    dvc_dq.setZero();
    dacom_dq.setZero();
    dacom_dv.setZero();

    DataCollectorMultibodyTpl<Scalar>* d =
        dynamic_cast<DataCollectorMultibodyTpl<Scalar>*>(shared);
    if (d == nullptr) {
      throw_pretty(
          "Invalid argument: the shared data should be derived from "
          "DataCollectorMultibody");
    }

    pinocchio = d->pinocchio;
    DataCollectorJointTpl<Scalar>* j =
        dynamic_cast<DataCollectorJointTpl<Scalar>*>(shared);
    if (j != nullptr) {
      joint = j->joint;
    }
  }

  virtual ~TaskDataCoMPositionTpl() = default;

  pinocchio::DataTpl<Scalar>* pinocchio;  //!< Shared Pinocchio data
  std::shared_ptr<JointDataAbstractTpl<Scalar>>
      joint;           //!< Shared generalized-acceleration data, when available
  Vector3s com;        //!< Center-of-mass position
  Vector3s vcom;       //!< Center-of-mass velocity
  Vector3s acom;       //!< Center-of-mass acceleration
  Matrix3xs dvc_dq;    //!< Partial derivative of CoM velocity w.r.t. q
  Matrix3xs dacom_dq;  //!< Partial derivative of CoM acceleration w.r.t. q
  Matrix3xs dacom_dv;  //!< Partial derivative of CoM acceleration w.r.t. v

  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/tasks/com-position.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::TaskModelCoMPositionTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::TaskDataCoMPositionTpl)

#endif  // CROCODDYL_MULTIBODY_TASKS_COM_POSITION_HPP_
