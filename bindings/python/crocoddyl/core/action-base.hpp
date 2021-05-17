///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_ACTION_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_ACTION_BASE_HPP_

#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace python {

class ActionModelAbstract_wrap : public ActionModelAbstract, public bp::wrapper<ActionModelAbstract> {
 public:
  ActionModelAbstract_wrap(boost::shared_ptr<StateAbstract> state, std::size_t nu, std::size_t nr = 1)
      : ActionModelAbstract(state, nu, nr), bp::wrapper<ActionModelAbstract>() {}

  void calc(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u) {
    if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
    }
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw_pretty("Invalid argument: "
                   << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
    }
    return bp::call<void>(this->get_override("calc").ptr(), data, (Eigen::VectorXd)x, (Eigen::VectorXd)u);
  }

  void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u) {
    if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
    }
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw_pretty("Invalid argument: "
                   << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
    }
    return bp::call<void>(this->get_override("calcDiff").ptr(), data, (Eigen::VectorXd)x, (Eigen::VectorXd)u);
  }

  boost::shared_ptr<ActionDataAbstract> createData() {
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<boost::shared_ptr<ActionDataAbstract> >(createData.ptr());
    }
    return ActionModelAbstract::createData();
  }

  boost::shared_ptr<ActionDataAbstract> default_createData() { return this->ActionModelAbstract::createData(); }

  void quasiStatic(const boost::shared_ptr<ActionDataAbstract>& data, Eigen::Ref<Eigen::VectorXd> u,
                   const Eigen::Ref<const Eigen::VectorXd>& x, const std::size_t maxiter, const double tol) {
    if (boost::python::override quasiStatic = this->get_override("quasiStatic")) {
      Eigen::VectorXd u_tmp = bp::call<Eigen::VectorXd>(quasiStatic.ptr(), data, (Eigen::VectorXd)x, maxiter, tol);
      if (static_cast<std::size_t>(u_tmp.size()) == nu_) {
        u = u_tmp;
      } else {
        throw_pretty("Invalid argument: "
                     << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
      }
      return;
    }
    return ActionModelAbstract::quasiStatic(data, u, x, maxiter, tol);
  }

  void default_quasiStatic(const boost::shared_ptr<ActionDataAbstract>& data, Eigen::Ref<Eigen::VectorXd> u,
                           const Eigen::Ref<const Eigen::VectorXd>& x, const std::size_t maxiter, const double tol) {
    return this->ActionModelAbstract::quasiStatic(data, u, x, maxiter, tol);
  }
};

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(ActionModel_quasiStatic_wraps, ActionModelAbstract::quasiStatic_x, 2, 4)

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_ACTION_BASE_HPP_
