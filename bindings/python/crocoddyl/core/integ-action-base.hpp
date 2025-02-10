///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2022, LAAS-CNRS, University of Edinburgh, University of
// Trento Copyright note valid unless otherwise stated in individual files. All
// rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_INTEGRATED_ACTION_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_INTEGRATED_ACTION_BASE_HPP_

#include "crocoddyl/core/integ-action-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

class IntegratedActionModelAbstract_wrap
    : public IntegratedActionModelAbstract,
      public bp::wrapper<IntegratedActionModelAbstract> {
 public:
  IntegratedActionModelAbstract_wrap(
      std::shared_ptr<DifferentialActionModelAbstract> model,
      const double timestep = 1e-3, const bool with_cost_residual = true)
      : IntegratedActionModelAbstract(model, timestep, with_cost_residual),
        bp::wrapper<IntegratedActionModelAbstract>() {}

  IntegratedActionModelAbstract_wrap(
      std::shared_ptr<DifferentialActionModelAbstract> model,
      std::shared_ptr<ControlParametrizationModelAbstract> control,
      const double timestep = 1e-3, const bool with_cost_residual = true)
      : IntegratedActionModelAbstract(model, control, timestep,
                                      with_cost_residual),
        bp::wrapper<IntegratedActionModelAbstract>() {}

  void calc(const std::shared_ptr<ActionDataAbstract>& data,
            const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u) {
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
    return bp::call<void>(this->get_override("calc").ptr(), data,
                          (Eigen::VectorXd)x, (Eigen::VectorXd)u);
  }

  void calcDiff(const std::shared_ptr<ActionDataAbstract>& data,
                const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u) {
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
                          (Eigen::VectorXd)x, (Eigen::VectorXd)u);
  }

  std::shared_ptr<ActionDataAbstract> createData() {
    enableMultithreading() = false;
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<std::shared_ptr<IntegratedActionDataAbstract> >(
          createData.ptr());
    }
    return IntegratedActionModelAbstract::createData();
  }

  std::shared_ptr<ActionDataAbstract> default_createData() {
    return this->IntegratedActionModelAbstract::createData();
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_INTEGRATED_ACTION_BASE_HPP_
