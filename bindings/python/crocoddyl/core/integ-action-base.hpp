///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, LAAS-CNRS, University of Edinburgh,
//                          University of Trento, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files. All
// rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_INTEGRATED_ACTION_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_INTEGRATED_ACTION_BASE_HPP_

#include "crocoddyl/core/integ-action-base.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

template <typename Scalar>
class IntegratedActionModelAbstractTpl_wrap
    : public IntegratedActionModelAbstractTpl<Scalar>,
      public bp::wrapper<IntegratedActionModelAbstractTpl<Scalar>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ActionModelBase, IntegratedActionModelAbstractTpl_wrap)

  typedef typename crocoddyl::IntegratedActionModelAbstractTpl<Scalar>
      IntegratedActionModel;
  typedef typename crocoddyl::IntegratedActionDataAbstractTpl<Scalar>
      IntegratedActionData;
  typedef typename crocoddyl::DifferentialActionModelAbstractTpl<Scalar>
      DifferentialActionModel;
  typedef typename crocoddyl::ActionDataAbstractTpl<Scalar> ActionData;
  typedef typename crocoddyl::StateAbstractTpl<Scalar> State;
  typedef typename IntegratedActionModel::ControlParametrizationModelAbstract
      ControlModel;
  typedef typename IntegratedActionModel::VectorXs VectorXs;
  using IntegratedActionModel::control_;
  using IntegratedActionModel::differential_;
  using IntegratedActionModel::nu_;
  using IntegratedActionModel::state_;
  using IntegratedActionModel::time_step_;
  using IntegratedActionModel::with_cost_residual_;

  IntegratedActionModelAbstractTpl_wrap(
      std::shared_ptr<DifferentialActionModel> model,
      const Scalar timestep = Scalar(1e-3),
      const bool with_cost_residual = true)
      : IntegratedActionModel(model, timestep, with_cost_residual),
        bp::wrapper<IntegratedActionModel>() {}

  IntegratedActionModelAbstractTpl_wrap(
      std::shared_ptr<DifferentialActionModel> model,
      std::shared_ptr<ControlModel> control,
      const Scalar timestep = Scalar(1e-3),
      const bool with_cost_residual = true)
      : IntegratedActionModel(model, control, timestep, with_cost_residual),
        bp::wrapper<IntegratedActionModel>() {}

  void calc(const std::shared_ptr<ActionData>& data,
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

  void calcDiff(const std::shared_ptr<ActionData>& data,
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

  std::shared_ptr<ActionData> createData() override {
    enableMultithreading() = false;
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<std::shared_ptr<IntegratedActionData>>(createData.ptr());
    }
    return IntegratedActionModel::createData();
  }

  std::shared_ptr<ActionData> default_createData() {
    return this->IntegratedActionModel::createData();
  }

  template <typename NewScalar>
  IntegratedActionModelAbstractTpl_wrap<NewScalar> cast() const {
    typedef IntegratedActionModelAbstractTpl_wrap<NewScalar> ReturnType;
    if (control_) {
      ReturnType ret(differential_->template cast<NewScalar>(),
                     control_->template cast<NewScalar>(),
                     scalar_cast<NewScalar>(time_step_), with_cost_residual_);
      return ret;
    } else {
      ReturnType ret(differential_->template cast<NewScalar>(),
                     scalar_cast<NewScalar>(time_step_), with_cost_residual_);
      return ret;
    }
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_INTEGRATED_ACTION_BASE_HPP_
