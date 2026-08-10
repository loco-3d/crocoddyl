///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_TASKS_FRAME_TRANSLATION_HPP_
#define CROCODDYL_MULTIBODY_TASKS_FRAME_TRANSLATION_HPP_

#include "crocoddyl/core/task-base.hpp"
#include "crocoddyl/core/utils/conversions.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

/**
 * @brief Frame translation task model.
 *
 * The task quantity is a 3D translation error expressed in the selected
 * reference frame, where `xref` is an absolute world translation:
 *  - LOCAL: \f$y = R^T (t - t_\mathrm{ref})\f$,
 *  - WORLD and LOCAL_WORLD_ALIGNED: \f$y = t - t_\mathrm{ref}\f$.
 *
 * The task rate \f$v\f$ is the corresponding linear velocity in the same
 * convention:
 * \f[
 *   v = \big({}^{\mathrm{rf}}\!V_f\big)_{\mathrm{lin}}.
 * \f]
 *
 * The acceleration term is the corresponding linear acceleration:
 * \f[
 *   a = \big({}^{\mathrm{rf}}\!A_f\big)_{\mathrm{lin}}.
 * \f]
 * WORLD and LOCAL_WORLD_ALIGNED are equivalent for the linear part.
 */
template <typename _Scalar>
class TaskModelFrameTranslationTpl : public TaskModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(TaskModelBase, TaskModelFrameTranslationTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef TaskModelAbstractTpl<Scalar> Base;
  typedef TaskDataAbstractTpl<Scalar> TaskDataAbstract;
  typedef TaskDataFrameTranslationTpl<Scalar> Data;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef typename MathBase::Matrix3s Matrix3s;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef pinocchio::MotionTpl<Scalar> Motion;

  /**
   * @brief Construct a frame translation task using LOCAL coordinates.
   */
  TaskModelFrameTranslationTpl(std::shared_ptr<StateMultibody> state,
                               const pinocchio::FrameIndex id,
                               const Vector3s& xref, const std::size_t nu);

  /**
   * @brief Construct a frame translation task with an explicit reference
   * frame convention.
   */
  TaskModelFrameTranslationTpl(std::shared_ptr<StateMultibody> state,
                               const pinocchio::FrameIndex id,
                               const Vector3s& xref,
                               const pinocchio::ReferenceFrame type,
                               const std::size_t nu);

  /**
   * @brief Construct a frame translation task using the default control
   * dimension and LOCAL coordinates.
   */
  TaskModelFrameTranslationTpl(std::shared_ptr<StateMultibody> state,
                               const pinocchio::FrameIndex id,
                               const Vector3s& xref);

  /**
   * @brief Construct a frame translation task with an explicit reference
   * frame convention.
   */
  TaskModelFrameTranslationTpl(std::shared_ptr<StateMultibody> state,
                               const pinocchio::FrameIndex id,
                               const Vector3s& xref,
                               const pinocchio::ReferenceFrame type);

  virtual ~TaskModelFrameTranslationTpl() = default;

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
  TaskModelFrameTranslationTpl<NewScalar> cast() const;

  /** @brief Return the frame index. */
  pinocchio::FrameIndex get_id() const;
  /** @brief Return the reference translation. */
  const Vector3s& get_reference() const;
  /** @brief Return the reference frame convention. */
  pinocchio::ReferenceFrame get_type() const;
  /** @brief Update the frame index. */
  void set_id(const pinocchio::FrameIndex id);
  /** @brief Update the reference translation. */
  void set_reference(const Vector3s& reference);
  /** @brief Update the reference frame convention. */
  void set_type(const pinocchio::ReferenceFrame type);

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
  pinocchio::FrameIndex id_;
  Vector3s xref_;
  pinocchio::ReferenceFrame type_;
  std::shared_ptr<typename StateMultibody::PinocchioModel> pin_model_;
};

/**
 * @brief Cached values and derivatives for the frame-translation task.
 */
template <typename _Scalar>
struct TaskDataFrameTranslationTpl : public TaskDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef TaskDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  typedef pinocchio::MotionTpl<Scalar> Motion;

  template <template <typename Scalar> class Model>
  TaskDataFrameTranslationTpl(Model<Scalar>* const model,
                              DataCollectorAbstract* const data)
      : Base(model, data),
        vf(Motion::Zero()),
        af(Motion::Zero()),
        fJf(6, model->get_state()->get_nv()),
        fVdq(6, model->get_state()->get_nv()),
        fVdv(6, model->get_state()->get_nv()),
        fAdq(6, model->get_state()->get_nv()),
        fAdv(6, model->get_state()->get_nv()) {
    fJf.setZero();
    fVdq.setZero();
    fVdv.setZero();
    fAdq.setZero();
    fAdv.setZero();

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

  virtual ~TaskDataFrameTranslationTpl() = default;

  pinocchio::DataTpl<Scalar>* pinocchio;  //!< Shared Pinocchio data
  std::shared_ptr<JointDataAbstractTpl<Scalar>>
      joint;  //!< Shared generalized-acceleration data, when available
  Motion vf;  //!< Frame velocity in the task's reference-frame convention
  Motion af;  //!< Frame acceleration in the task's reference-frame convention
  Matrix6xs fJf;   //!< Local Jacobian of the frame
  Matrix6xs fVdq;  //!< Partial derivative of frame velocity with respect to q
  Matrix6xs fVdv;  //!< Partial derivative of frame velocity with respect to v
  Matrix6xs fAdq;  //!< Partial derivative of frame acceleration w.r.t. q
  Matrix6xs fAdv;  //!< Partial derivative of frame acceleration w.r.t. v

  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/tasks/frame-translation.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::TaskModelFrameTranslationTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::TaskDataFrameTranslationTpl)

#endif  // CROCODDYL_MULTIBODY_TASKS_FRAME_TRANSLATION_HPP_
