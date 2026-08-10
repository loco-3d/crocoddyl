///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_TASKS_CENTROIDAL_MOMENTUM_HPP_
#define CROCODDYL_MULTIBODY_TASKS_CENTROIDAL_MOMENTUM_HPP_

#include "crocoddyl/core/task-base.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

/**
 * @brief Centroidal-momentum task model.
 *
 * The task quantity and rate are
 * \f[
 *   y = h-h_{\mathrm{ref}}, \qquad
 *   v = \dot h-\dot h_{\mathrm{ref}},
 * \f]
 * with
 * \f[
 *   h=A_g(q)\dot q, \qquad
 *   \dot h=\dot A_g(q,\dot q)\dot q+A_g(q)\ddot q.
 * \f]
 * The model is first-order only: a centroidal-momentum acceleration would
 * require generalized jerk, which is not supplied by differential action
 * models. Therefore, a, Ax, and Au remain zero.
 *
 * The momentum and momentum-rate values, as well as their partial derivatives
 * at fixed generalized acceleration, are read from the shared Pinocchio data.
 * When shared joint data is available, its acceleration derivatives are used
 * to apply the chain rule with respect to the action-model variables.
 */
template <typename _Scalar>
class TaskModelCentroidalMomentumTpl : public TaskModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(TaskModelBase, TaskModelCentroidalMomentumTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef TaskModelAbstractTpl<Scalar> Base;
  typedef TaskDataCentroidalMomentumTpl<Scalar> Data;
  typedef TaskDataAbstractTpl<Scalar> TaskDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef typename MathBase::Vector6s Vector6s;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize with momentum and momentum-rate references.
   *
   * @param[in] state      State of the multibody system
   * @param[in] href       Reference centroidal momentum
   * @param[in] hdot_ref  Reference centroidal-momentum rate
   * @param[in] nu         Dimension of the control vector
   */
  TaskModelCentroidalMomentumTpl(std::shared_ptr<StateMultibody> state,
                                 const Vector6s& href, const Vector6s& hdot_ref,
                                 const std::size_t nu);

  /**
   * @brief Initialize with momentum and momentum-rate references.
   *
   * The default control dimension is state.nv.
   *
   * @param[in] state      State of the multibody system
   * @param[in] href       Reference centroidal momentum
   * @param[in] hdot_ref  Reference centroidal-momentum rate
   */
  TaskModelCentroidalMomentumTpl(std::shared_ptr<StateMultibody> state,
                                 const Vector6s& href,
                                 const Vector6s& hdot_ref);

  /**
   * @brief Initialize with a constant momentum reference.
   *
   * The momentum-rate reference is zero.
   *
   * @param[in] state  State of the multibody system
   * @param[in] href   Reference centroidal momentum
   * @param[in] nu     Dimension of the control vector
   */
  TaskModelCentroidalMomentumTpl(std::shared_ptr<StateMultibody> state,
                                 const Vector6s& href, const std::size_t nu);

  /**
   * @brief Initialize with a constant momentum reference.
   *
   * The momentum-rate reference is zero and the default control dimension is
   * state.nv.
   *
   * @param[in] state  State of the multibody system
   * @param[in] href   Reference centroidal momentum
   */
  TaskModelCentroidalMomentumTpl(std::shared_ptr<StateMultibody> state,
                                 const Vector6s& href);
  virtual ~TaskModelCentroidalMomentumTpl() = default;

  virtual void calc(const std::shared_ptr<TaskDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  virtual void calcDiff(const std::shared_ptr<TaskDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;

  virtual std::shared_ptr<TaskDataAbstract> createData(
      DataCollectorAbstract* const data) override;

  template <typename NewScalar>
  TaskModelCentroidalMomentumTpl<NewScalar> cast() const;

  const Vector6s& get_reference() const;
  const Vector6s& get_rate_reference() const;
  void set_reference(const Vector6s& href);
  void set_rate_reference(const Vector6s& hdot_ref);

  virtual void print(std::ostream& os) const override;

 protected:
  using Base::nr_;
  using Base::nu_;
  using Base::state_;

 private:
  Vector6s href_;
  Vector6s hdot_ref_;
  std::shared_ptr<typename StateMultibody::PinocchioModel> pin_model_;
};

/**
 * @brief Cached values and derivatives for the centroidal-momentum task.
 */
template <typename _Scalar>
struct TaskDataCentroidalMomentumTpl : public TaskDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef TaskDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::Vector6s Vector6s;
  typedef typename MathBase::Matrix6xs Matrix6xs;

  /**
   * @brief Bind the task data to the shared Pinocchio data and, optionally,
   * joint data.
   *
   * @param[in] model  Centroidal-momentum task model
   * @param[in] data   Shared data derived from DataCollectorMultibodyTpl and,
   *                   optionally, DataCollectorJointTpl
   */
  template <template <typename Scalar> class Model>
  TaskDataCentroidalMomentumTpl(Model<Scalar>* const model,
                                DataCollectorAbstract* const data)
      : Base(model, data),
        hdot(Vector6s::Zero()),
        dh_dq(6, model->get_state()->get_nv()),
        dhdot_dq(6, model->get_state()->get_nv()),
        dhdot_dv(6, model->get_state()->get_nv()),
        dhdot_da(6, model->get_state()->get_nv()) {
    dh_dq.setZero();
    dhdot_dq.setZero();
    dhdot_dv.setZero();
    dhdot_da.setZero();

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

  virtual ~TaskDataCentroidalMomentumTpl() = default;

  pinocchio::DataTpl<Scalar>* pinocchio;  //!< Shared Pinocchio data
  std::shared_ptr<JointDataAbstractTpl<Scalar>>
      joint;           //!< Shared generalized-acceleration data
  Vector6s hdot;       //!< Current centroidal-momentum rate
  Matrix6xs dh_dq;     //!< Partial derivative of h with respect to q
  Matrix6xs dhdot_dq;  //!< Partial derivative of hdot with respect to q
  Matrix6xs dhdot_dv;  //!< Partial derivative of hdot with respect to v
  Matrix6xs dhdot_da;  //!< Partial derivative of hdot with respect to ddq

  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/tasks/centroidal-momentum.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::TaskModelCentroidalMomentumTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::TaskDataCentroidalMomentumTpl)

#endif  // CROCODDYL_MULTIBODY_TASKS_CENTROIDAL_MOMENTUM_HPP_
