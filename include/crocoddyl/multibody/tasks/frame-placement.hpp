///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_TASKS_FRAME_PLACEMENT_HPP_
#define CROCODDYL_MULTIBODY_TASKS_FRAME_PLACEMENT_HPP_

#include "crocoddyl/core/task-base.hpp"
#include "crocoddyl/core/utils/conversions.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

/**
 * @brief Frame placement task model.
 *
 * The task quantity is a 6D pose error expressed in the selected reference
 * frame. Let \f$M=(R,t)\f$ be the current frame placement,
 * \f$M_{\mathrm{ref}}\f$ the absolute reference placement, and
 * \f[
 *   y_L = \log_6\!\left(M_{\mathrm{ref}}^{-1} M\right).
 * \f]
 * The three conventions are
 * \f[
 * \begin{aligned}
 *   \mathrm{LOCAL}:\quad & y = y_L, \\
 *   \mathrm{WORLD}:\quad & y =
 *       \log_6\!\left(M M_{\mathrm{ref}}^{-1}\right), \\
 *   \mathrm{LOCAL\_WORLD\_ALIGNED}:\quad & y =
 *       \begin{bmatrix}R&0\\0&R\end{bmatrix} y_L.
 * \end{aligned}
 * \f]
 * The reference placement is always an absolute pose in the world. The task
 * rate and acceleration use the same reference-frame convention as the error:
 * \f[
 *   v = {}^{\mathrm{rf}}\!V_f,\qquad
 *   a = {}^{\mathrm{rf}}\!A_f.
 * \f]
 * Here \f${}^{\mathrm{rf}}\!V_f\f$ and \f${}^{\mathrm{rf}}\!A_f\f$ are the
 * frame spatial velocity and acceleration expressed in the selected reference
 * frame.
 */
template <typename _Scalar>
class TaskModelFramePlacementTpl : public TaskModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(TaskModelBase, TaskModelFramePlacementTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef TaskModelAbstractTpl<Scalar> Base;
  typedef TaskDataAbstractTpl<Scalar> TaskDataAbstract;
  typedef TaskDataFramePlacementTpl<Scalar> Data;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef typename MathBase::Matrix3s Matrix3s;
  typedef typename MathBase::Matrix6s Matrix6s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef pinocchio::MotionTpl<Scalar> Motion;
  typedef pinocchio::SE3Tpl<Scalar> SE3;

  /**
   * @brief Construct a frame placement task.
   */
  TaskModelFramePlacementTpl(std::shared_ptr<StateMultibody> state,
                             const pinocchio::FrameIndex id, const SE3& pref,
                             const std::size_t nu);

  /**
   * @brief Construct a frame placement task using the default control
   * dimension.
   */
  TaskModelFramePlacementTpl(std::shared_ptr<StateMultibody> state,
                             const pinocchio::FrameIndex id, const SE3& pref);

  /**
   * @brief Construct a frame placement task with an explicit reference frame.
   */
  TaskModelFramePlacementTpl(std::shared_ptr<StateMultibody> state,
                             const pinocchio::FrameIndex id, const SE3& pref,
                             const pinocchio::ReferenceFrame type,
                             const std::size_t nu);

  /**
   * @brief Construct a frame placement task with an explicit reference frame.
   */
  TaskModelFramePlacementTpl(std::shared_ptr<StateMultibody> state,
                             const pinocchio::FrameIndex id, const SE3& pref,
                             const pinocchio::ReferenceFrame type);

  virtual ~TaskModelFramePlacementTpl() = default;

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
  TaskModelFramePlacementTpl<NewScalar> cast() const;

  /** @brief Return the frame index. */
  pinocchio::FrameIndex get_id() const;
  /** @brief Return the reference placement. */
  const SE3& get_reference() const;
  /** @brief Return the reference frame convention. */
  pinocchio::ReferenceFrame get_type() const;
  /** @brief Update the frame index. */
  void set_id(const pinocchio::FrameIndex id);
  /** @brief Update the reference placement. */
  void set_reference(const SE3& reference);
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
  SE3 pref_;
  SE3 oMf_inv_;
  pinocchio::ReferenceFrame type_;
  std::shared_ptr<typename StateMultibody::PinocchioModel> pin_model_;
};

/**
 * @brief Cached values and derivatives for the frame-placement task.
 */
template <typename _Scalar>
struct TaskDataFramePlacementTpl : public TaskDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef TaskDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::Matrix6s Matrix6s;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  typedef pinocchio::MotionTpl<Scalar> Motion;
  typedef pinocchio::SE3Tpl<Scalar> SE3;

  template <template <typename Scalar> class Model>
  TaskDataFramePlacementTpl(Model<Scalar>* const model,
                            DataCollectorAbstract* const data)
      : Base(model, data),
        rMf(SE3::Identity()),
        rJf(Matrix6s::Zero()),
        y_local(Motion::Zero()),
        vf(Motion::Zero()),
        af(Motion::Zero()),
        Yx_local(6, model->get_state()->get_nv()),
        fJf(6, model->get_state()->get_nv()),
        fVdq(6, model->get_state()->get_nv()),
        fVdv(6, model->get_state()->get_nv()),
        fAdq(6, model->get_state()->get_nv()),
        fAdv(6, model->get_state()->get_nv()) {
    Yx_local.setZero();
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

  virtual ~TaskDataFramePlacementTpl() = default;

  pinocchio::DataTpl<Scalar>* pinocchio;  //!< Shared Pinocchio data
  std::shared_ptr<JointDataAbstractTpl<Scalar>>
      joint;           //!< Shared generalized-acceleration data, when available
  SE3 rMf;             //!< Pose-error composition for the selected convention
  Matrix6s rJf;        //!< Jacobian of the SE(3) logarithm
  Motion y_local;      //!< Local pose error used by LOCAL_WORLD_ALIGNED
  Motion vf;           //!< Frame velocity in the selected reference frame
  Motion af;           //!< Frame acceleration in the selected reference frame
  Matrix6xs Yx_local;  //!< Jacobian of the local pose error
  Matrix6xs fJf;       //!< Frame Jacobian in the selected intermediate frame
  Matrix6xs fVdq;  //!< Partial derivative of frame velocity with respect to q
  Matrix6xs fVdv;  //!< Partial derivative of frame velocity with respect to v
  Matrix6xs fAdq;  //!< Partial derivative of frame acceleration w.r.t. q
  Matrix6xs fAdv;  //!< Partial derivative of frame acceleration w.r.t. v

  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/tasks/frame-placement.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::TaskModelFramePlacementTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::TaskDataFramePlacementTpl)

#endif  // CROCODDYL_MULTIBODY_TASKS_FRAME_PLACEMENT_HPP_
