///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_DIFF_ACTION_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_DIFF_ACTION_BASE_HPP_

#include "python/crocoddyl/core/core.hpp"
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace python {

class DifferentialActionModelAbstract_wrap : public DifferentialActionModelAbstract,
                                             public bp::wrapper<DifferentialActionModelAbstract> {
 public:
  DifferentialActionModelAbstract_wrap(boost::shared_ptr<StateAbstract> state, int nu, int nr = 1)
      : DifferentialActionModelAbstract(state, nu, nr), bp::wrapper<DifferentialActionModelAbstract>() {}

  void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
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

  void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u) {
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

  boost::shared_ptr<DifferentialActionDataAbstract> createData() {
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<boost::shared_ptr<DifferentialActionDataAbstract> >(createData.ptr());
    }
    return DifferentialActionModelAbstract::createData();
  }

  boost::shared_ptr<DifferentialActionDataAbstract> default_createData() {
    return this->DifferentialActionModelAbstract::createData();
  }

  void quasiStatic(const boost::shared_ptr<DifferentialActionDataAbstract>& data, Eigen::Ref<Eigen::VectorXd> u,
                   const Eigen::Ref<const Eigen::VectorXd>& x, const std::size_t maxiter, const double tol) {
    if (boost::python::override createData = this->get_override("quasiStatic")) {
      Eigen::VectorXd u_tmp =
          bp::call<Eigen::VectorXd>(this->get_override("quasiStatic").ptr(), data, (Eigen::VectorXd)x, maxiter, tol);
      if (static_cast<std::size_t>(u_tmp.size()) == nu_) {
        u = u_tmp;
      } else {
        throw_pretty("Invalid argument: "
                     << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
      }
      return;
    }
    return DifferentialActionModelAbstract::quasiStatic(data, u, x, maxiter, tol);
  }

  void default_quasiStatic(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                           Eigen::Ref<Eigen::VectorXd> u, const Eigen::Ref<const Eigen::VectorXd>& x,
                           const std::size_t maxiter, const double tol) {
    return this->DifferentialActionModelAbstract::quasiStatic(data, u, x, maxiter, tol);
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_DIFF_ACTION_BASE_HPP_