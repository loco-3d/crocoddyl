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
 * The task quantity is the 6D pose error
 * \f[
 *   y = \log_6\!\left(M_{\mathrm{ref}}^{-1} M\right),
 * \f]
 * where \f$M\f$ is the current frame placement and \f$M_{\mathrm{ref}}\f$ is
 * the reference placement. The reference placement is always an absolute pose
 * in the world, while the `ReferenceFrame` type selects how the task rate and
 * acceleration are expressed:
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
  }

  virtual ~TaskDataFramePlacementTpl() = default;

  pinocchio::DataTpl<Scalar>* pinocchio;
  SE3 rMf;
  Matrix6s rJf;
  Motion vf;
  Motion af;
  Matrix6xs fJf;
  Matrix6xs fVdq;
  Matrix6xs fVdv;
  Matrix6xs fAdq;
  Matrix6xs fAdv;

  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/tasks/frame-placement.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::TaskModelFramePlacementTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::TaskDataFramePlacementTpl)

#endif  // CROCODDYL_MULTIBODY_TASKS_FRAME_PLACEMENT_HPP_
