///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_TASK_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_TASK_BASE_HPP_

#include "crocoddyl/core/task-base.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

template <typename _Scalar>
class TaskModelAbstractTpl_wrap
    : public TaskModelAbstractTpl<_Scalar>,
      public bp::wrapper<TaskModelAbstractTpl<_Scalar>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(TaskModelBase, TaskModelAbstractTpl_wrap)

  typedef _Scalar Scalar;
  typedef crocoddyl::TaskModelAbstractTpl<Scalar> TaskModel;
  typedef crocoddyl::TaskDataAbstractTpl<Scalar> TaskData;
  typedef crocoddyl::TaskDataAbstractTpl<Scalar> TaskDataAbstract;
  typedef typename TaskModel::StateAbstract State;
  typedef typename TaskModel::VectorXs VectorXs;
  typedef typename TaskModel::DataCollectorAbstract DataCollectorAbstract;
  using TaskModel::has_acceleration_;
  using TaskModel::nr_;
  using TaskModel::nu_;
  using TaskModel::q_dependent_;
  using TaskModel::state_;
  using TaskModel::u_dependent_;
  using TaskModel::v_dependent_;

  TaskModelAbstractTpl_wrap(std::shared_ptr<State> state, const std::size_t nr,
                            const std::size_t nu, const bool q_dependent = true,
                            const bool v_dependent = true,
                            const bool u_dependent = true,
                            const bool has_acceleration = true)
      : TaskModel(state, nr, nu, q_dependent, v_dependent, u_dependent,
                  has_acceleration),
        bp::wrapper<TaskModel>() {}

  void calc(const std::shared_ptr<TaskData>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u) override {
    if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
      throw_pretty(
          "Invalid argument: " << "x has wrong dimension (it should be " +
                                      std::to_string(state_->get_nx()) + ")");
    }
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw_pretty(
          "Invalid argument: " << "u has wrong dimension (it should be " +
                                      std::to_string(nu_) + ")");
    }
    return bp::call<void>(this->get_override("calc").ptr(), data, (VectorXs)x,
                          (VectorXs)u);
  }

  void calcDiff(const std::shared_ptr<TaskData>& data,
                const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u) override {
    if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
      throw_pretty(
          "Invalid argument: " << "x has wrong dimension (it should be " +
                                      std::to_string(state_->get_nx()) + ")");
    }
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw_pretty(
          "Invalid argument: " << "u has wrong dimension (it should be " +
                                      std::to_string(nu_) + ")");
    }
    return bp::call<void>(this->get_override("calcDiff").ptr(), data,
                          (VectorXs)x, (VectorXs)u);
  }

  std::shared_ptr<TaskData> createData(
      DataCollectorAbstract* const data) override {
    enableMultithreading() = false;
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<std::shared_ptr<TaskData>>(createData.ptr(),
                                                 boost::ref(data));
    }
    return TaskModel::createData(data);
  }

  std::shared_ptr<TaskData> default_createData(
      DataCollectorAbstract* const data) {
    return this->TaskModel::createData(data);
  }

  template <typename NewScalar>
  TaskModelAbstractTpl_wrap<NewScalar> cast() const {
    typedef TaskModelAbstractTpl_wrap<NewScalar> ReturnType;
    typedef StateAbstractTpl<NewScalar> StateType;
    ReturnType ret(
        std::static_pointer_cast<StateType>(state_->template cast<NewScalar>()),
        nr_, nu_, q_dependent_, v_dependent_, u_dependent_, has_acceleration_);
    return ret;
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_TASK_BASE_HPP_
