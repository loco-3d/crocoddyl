///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_COST_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_COST_BASE_HPP_

#include "crocoddyl/core/cost-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

class CostModelAbstract_wrap : public CostModelAbstract,
                               public bp::wrapper<CostModelAbstract> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using CostModelAbstract::nu_;
  using CostModelAbstract::unone_;

  CostModelAbstract_wrap(std::shared_ptr<StateAbstract> state,
                         std::shared_ptr<ActivationModelAbstract> activation,
                         std::shared_ptr<ResidualModelAbstract> residual)
      : CostModelAbstract(state, activation, residual) {
    unone_ = NAN * MathBase::VectorXs::Ones(nu_);
  }

  CostModelAbstract_wrap(std::shared_ptr<StateAbstract> state,
                         std::shared_ptr<ActivationModelAbstract> activation,
                         const std::size_t nu)
      : CostModelAbstract(state, activation, nu) {
    unone_ = NAN * MathBase::VectorXs::Ones(nu);
  }

  CostModelAbstract_wrap(std::shared_ptr<StateAbstract> state,
                         std::shared_ptr<ActivationModelAbstract> activation)
      : CostModelAbstract(state, activation) {
    unone_ = NAN * MathBase::VectorXs::Ones(nu_);
  }

  CostModelAbstract_wrap(std::shared_ptr<StateAbstract> state,
                         std::shared_ptr<ResidualModelAbstract> residual)
      : CostModelAbstract(state, residual) {
    unone_ = NAN * MathBase::VectorXs::Ones(nu_);
  }

  CostModelAbstract_wrap(std::shared_ptr<StateAbstract> state,
                         const std::size_t nr, const std::size_t nu)
      : CostModelAbstract(state, nr, nu), bp::wrapper<CostModelAbstract>() {
    unone_ = NAN * MathBase::VectorXs::Ones(nu);
  }

  CostModelAbstract_wrap(std::shared_ptr<StateAbstract> state,
                         const std::size_t nr)
      : CostModelAbstract(state, nr), bp::wrapper<CostModelAbstract>() {
    unone_ = NAN * MathBase::VectorXs::Ones(nu_);
  }

  void calc(const std::shared_ptr<CostDataAbstract>& data,
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
    if (std::isnan(u.lpNorm<Eigen::Infinity>())) {
      return bp::call<void>(this->get_override("calc").ptr(), data,
                            (Eigen::VectorXd)x);
    } else {
      return bp::call<void>(this->get_override("calc").ptr(), data,
                            (Eigen::VectorXd)x, (Eigen::VectorXd)u);
    }
  }

  void calcDiff(const std::shared_ptr<CostDataAbstract>& data,
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
    if (std::isnan(u.lpNorm<Eigen::Infinity>())) {
      return bp::call<void>(this->get_override("calcDiff").ptr(), data,
                            (Eigen::VectorXd)x);
    } else {
      return bp::call<void>(this->get_override("calcDiff").ptr(), data,
                            (Eigen::VectorXd)x, (Eigen::VectorXd)u);
    }
  }

  std::shared_ptr<CostDataAbstract> createData(
      DataCollectorAbstract* const data) {
    enableMultithreading() = false;
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<std::shared_ptr<CostDataAbstract> >(createData.ptr(),
                                                          boost::ref(data));
    }
    return CostModelAbstract::createData(data);
  }

  std::shared_ptr<CostDataAbstract> default_createData(
      DataCollectorAbstract* const data) {
    return this->CostModelAbstract::createData(data);
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_COST_BASE_HPP_
