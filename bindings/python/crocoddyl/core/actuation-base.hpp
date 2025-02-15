
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2022, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_ACTUATION_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_ACTUATION_BASE_HPP_

#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

class ActuationModelAbstract_wrap : public ActuationModelAbstract,
                                    public bp::wrapper<ActuationModelAbstract> {
 public:
  ActuationModelAbstract_wrap(std::shared_ptr<StateAbstract> state,
                              const std::size_t nu)
      : ActuationModelAbstract(state, nu),
        bp::wrapper<ActuationModelAbstract>() {}

  void calc(const std::shared_ptr<ActuationDataAbstract>& data,
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

  void calcDiff(const std::shared_ptr<ActuationDataAbstract>& data,
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

  void commands(const std::shared_ptr<ActuationDataAbstract>& data,
                const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& tau) {
    if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
      throw_pretty(
          "Invalid argument: " << "x has wrong dimension (it should be " +
                                      std::to_string(state_->get_nx()) + ")");
    }
    if (static_cast<std::size_t>(tau.size()) != state_->get_nv()) {
      throw_pretty(
          "Invalid argument: " << "tau has wrong dimension (it should be " +
                                      std::to_string(state_->get_nv()) + ")");
    }
    return bp::call<void>(this->get_override("commands").ptr(), data,
                          (Eigen::VectorXd)x, (Eigen::VectorXd)tau);
  }

  void torqueTransform(const std::shared_ptr<ActuationDataAbstract>& data,
                       const Eigen::Ref<const VectorXs>& x,
                       const Eigen::Ref<const VectorXs>& u) {
    if (boost::python::override torqueTransform =
            this->get_override("torqueTransform")) {
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
      return bp::call<void>(torqueTransform.ptr(), data, (Eigen::VectorXd)x,
                            (Eigen::VectorXd)u);
    }
    return ActuationModelAbstract::torqueTransform(data, x, u);
  }

  void default_torqueTransform(
      const std::shared_ptr<ActuationDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x,
      const Eigen::Ref<const VectorXs>& u) {
    return this->ActuationModelAbstract::torqueTransform(data, x, u);
  }

  std::shared_ptr<ActuationDataAbstract> createData() {
    enableMultithreading() = false;
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<std::shared_ptr<ActuationDataAbstract> >(
          createData.ptr());
    }
    return ActuationModelAbstract::createData();
  }

  std::shared_ptr<ActuationDataAbstract> default_createData() {
    return this->ActuationModelAbstract::createData();
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_ACTUATION_BASE_HPP_
