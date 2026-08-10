///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_TASKS_FRAME_ROTATION_HPP_
#define CROCODDYL_MULTIBODY_TASKS_FRAME_ROTATION_HPP_

#include "crocoddyl/core/task-base.hpp"
#include "crocoddyl/core/utils/conversions.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

/**
 * @brief Frame rotation task model.
 *
 * The task quantity \f$y\f$ is a log-map rotation error. The reference frame
 * selects the convention:
 *  - LOCAL: \f$y = \log(R_{\mathrm{ref}}^T R)\f$,
 *    \f$v = J_{\log}(y)\,{}^f\omega\f$, and
 *    \f$a = J_{\log}(y)\,{}^f\alpha\f$;
 *  - WORLD and LOCAL_WORLD_ALIGNED:
 *    \f$y = \log(R R_{\mathrm{ref}}^T)\f$,
 *    \f$v = J_{\log}(y)\,{}^o\omega\f$, and
 *    \f$a = J_{\log}(y)\,{}^o\alpha\f$.
 *
 * WORLD and LOCAL_WORLD_ALIGNED are equivalent here because only the angular
 * component is used.
 */
template <typename _Scalar>
class TaskModelFrameRotationTpl : public TaskModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(TaskModelBase, TaskModelFrameRotationTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef TaskModelAbstractTpl<Scalar> Base;
  typedef TaskDataAbstractTpl<Scalar> TaskDataAbstract;
  typedef TaskDataFrameRotationTpl<Scalar> Data;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef typename MathBase::Matrix3s Matrix3s;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef pinocchio::MotionTpl<Scalar> Motion;

  /**
   * @brief Construct a frame rotation task with LOCAL reference frame.
   *
   * The task dimension is fixed to 3 and the task does not depend on the
   * control input.
   */
  TaskModelFrameRotationTpl(std::shared_ptr<StateMultibody> state,
                            const pinocchio::FrameIndex id,
                            const Matrix3s& Rref, const std::size_t nu);

  /**
   * @brief Construct a frame rotation task with LOCAL reference frame.
   */
  TaskModelFrameRotationTpl(std::shared_ptr<StateMultibody> state,
                            const pinocchio::FrameIndex id,
                            const Matrix3s& Rref);

  /**
   * @brief Construct a frame rotation task with an explicit reference frame.
   */
  TaskModelFrameRotationTpl(std::shared_ptr<StateMultibody> state,
                            const pinocchio::FrameIndex id,
                            const Matrix3s& Rref,
                            const pinocchio::ReferenceFrame type,
                            const std::size_t nu);

  virtual ~TaskModelFrameRotationTpl() = default;

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
  TaskModelFrameRotationTpl<NewScalar> cast() const;

  /** @brief Return the frame index. */
  pinocchio::FrameIndex get_id() const;
  /** @brief Return the reference rotation. */
  const Matrix3s& get_reference() const;
  /** @brief Return the reference frame convention. */
  pinocchio::ReferenceFrame get_type() const;
  /** @brief Update the frame index. */
  void set_id(const pinocchio::FrameIndex id);
  /** @brief Update the reference rotation. */
  void set_reference(const Matrix3s& reference);
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
  Matrix3s Rref_;
  Matrix3s oRf_inv_;
  pinocchio::ReferenceFrame type_;
  std::shared_ptr<typename StateMultibody::PinocchioModel> pin_model_;
};

/**
 * @brief Cached values and derivatives for the frame-rotation task.
 */
template <typename _Scalar>
struct TaskDataFrameRotationTpl : public TaskDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef TaskDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::Matrix3s Matrix3s;
  typedef typename MathBase::Matrix3xs Matrix3xs;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  typedef pinocchio::MotionTpl<Scalar> Motion;

  template <template <typename Scalar> class Model>
  TaskDataFrameRotationTpl(Model<Scalar>* const model,
                           DataCollectorAbstract* const data)
      : Base(model, data),
        rRf(Matrix3s::Identity()),
        rJf(Matrix3s::Zero()),
        vf(Motion::Zero()),
        af(Motion::Zero()),
        Hlogf(Matrix3s::Zero()),
        dJ_v(Matrix3s::Zero()),
        dJ_a(Matrix3s::Zero()),
        a_partial_da(3, model->get_state()->get_nv()),
        fJf(6, model->get_state()->get_nv()),
        fVdq(6, model->get_state()->get_nv()),
        fVdv(6, model->get_state()->get_nv()),
        fAdq(6, model->get_state()->get_nv()),
        fAdv(6, model->get_state()->get_nv()) {
    a_partial_da.setZero();
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

  virtual ~TaskDataFrameRotationTpl() = default;

  pinocchio::DataTpl<Scalar>* pinocchio;  //!< Shared Pinocchio data
  std::shared_ptr<JointDataAbstractTpl<Scalar>>
      joint;       //!< Shared generalized-acceleration data, when available
  Matrix3s rRf;    //!< Rotation-error composition in the selected convention
  Matrix3s rJf;    //!< Jacobian of the SO(3) logarithm
  Motion vf;       //!< Frame velocity in the selected reference frame
  Motion af;       //!< Frame acceleration in the selected reference frame
  Matrix3s Hlogf;  //!< Hessian of the SO(3) logarithm
  Matrix3s dJ_v;   //!< Derivative of rJf contracted with frame velocity
  Matrix3s dJ_a;   //!< Derivative of rJf contracted with frame acceleration
  Matrix3xs a_partial_da;  //!< Derivative of task acceleration w.r.t. ddq
  Matrix6xs fJf;           //!< Frame Jacobian in the selected reference frame
  Matrix6xs fVdq;  //!< Partial derivative of frame velocity with respect to q
  Matrix6xs fVdv;  //!< Partial derivative of frame velocity with respect to v
  Matrix6xs fAdq;  //!< Partial derivative of frame acceleration w.r.t. q
  Matrix6xs fAdv;  //!< Partial derivative of frame acceleration w.r.t. v

  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/tasks/frame-rotation.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::TaskModelFrameRotationTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::TaskDataFrameRotationTpl)

#endif  // CROCODDYL_MULTIBODY_TASKS_FRAME_ROTATION_HPP_
