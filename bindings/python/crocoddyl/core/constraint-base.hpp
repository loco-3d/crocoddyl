///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2024, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_CONSTRAINT_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_CONSTRAINT_BASE_HPP_

#include "crocoddyl/core/constraint-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

class ConstraintModelAbstract_wrap
    : public ConstraintModelAbstract,
      public bp::wrapper<ConstraintModelAbstract> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using ConstraintModelAbstract::nu_;
  using ConstraintModelAbstract::unone_;

  ConstraintModelAbstract_wrap(std::shared_ptr<StateAbstract> state,
                               std::shared_ptr<ResidualModelAbstract> residual,
                               const std::size_t ng, const std::size_t nh)
      : ConstraintModelAbstract(state, residual, ng, nh),
        bp::wrapper<ConstraintModelAbstract>() {
    unone_ = NAN * MathBase::VectorXs::Ones(nu_);
  }

  ConstraintModelAbstract_wrap(std::shared_ptr<StateAbstract> state,
                               const std::size_t nu, const std::size_t ng,
                               const std::size_t nh, const bool T_const = true)
      : ConstraintModelAbstract(state, nu, ng, nh, T_const),
        bp::wrapper<ConstraintModelAbstract>() {
    unone_ = NAN * MathBase::VectorXs::Ones(nu);
  }

  ConstraintModelAbstract_wrap(std::shared_ptr<StateAbstract> state,
                               const std::size_t ng, const std::size_t nh,
                               const bool T_const = true)
      : ConstraintModelAbstract(state, ng, nh, T_const) {
    unone_ = NAN * MathBase::VectorXs::Ones(nu_);
  }

  void calc(const std::shared_ptr<ConstraintDataAbstract>& data,
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

  void calcDiff(const std::shared_ptr<ConstraintDataAbstract>& data,
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

  std::shared_ptr<ConstraintDataAbstract> createData(
      DataCollectorAbstract* const data) {
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<std::shared_ptr<ConstraintDataAbstract> >(
          createData.ptr(), boost::ref(data));
    }
    return ConstraintModelAbstract::createData(data);
  }

  std::shared_ptr<ConstraintDataAbstract> default_createData(
      DataCollectorAbstract* const data) {
    return this->ConstraintModelAbstract::createData(data);
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_CONSTRAINT_BASE_HPP_
