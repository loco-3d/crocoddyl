///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_DIFF_ACTION_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_DIFF_ACTION_BASE_HPP_

#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

class DifferentialActionModelAbstract_wrap
    : public DifferentialActionModelAbstract,
      public bp::wrapper<DifferentialActionModelAbstract> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  using DifferentialActionModelAbstract::unone_;

  DifferentialActionModelAbstract_wrap(std::shared_ptr<StateAbstract> state,
                                       const std::size_t nu,
                                       const std::size_t nr = 1,
                                       const std::size_t ng = 0,
                                       const std::size_t nh = 0,
                                       const std::size_t ng_T = 0,
                                       const std::size_t nh_T = 0)
      : DifferentialActionModelAbstract(state, nu, nr, ng, nh, ng_T, nh_T),
        bp::wrapper<DifferentialActionModelAbstract>() {
    unone_ = NAN * MathBase::VectorXs::Ones(nu);
  }

  void calc(const std::shared_ptr<DifferentialActionDataAbstract>& data,
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

  void calcDiff(const std::shared_ptr<DifferentialActionDataAbstract>& data,
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

  std::shared_ptr<DifferentialActionDataAbstract> createData() {
    enableMultithreading() = false;
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<std::shared_ptr<DifferentialActionDataAbstract> >(
          createData.ptr());
    }
    return DifferentialActionModelAbstract::createData();
  }

  std::shared_ptr<DifferentialActionDataAbstract> default_createData() {
    return this->DifferentialActionModelAbstract::createData();
  }

  void quasiStatic(const std::shared_ptr<DifferentialActionDataAbstract>& data,
                   Eigen::Ref<Eigen::VectorXd> u,
                   const Eigen::Ref<const Eigen::VectorXd>& x,
                   const std::size_t maxiter, const double tol) {
    if (boost::python::override quasiStatic =
            this->get_override("quasiStatic")) {
      u = bp::call<Eigen::VectorXd>(quasiStatic.ptr(), data, (Eigen::VectorXd)x,
                                    maxiter, tol);
      if (static_cast<std::size_t>(u.size()) != nu_) {
        throw_pretty(
            "Invalid argument: " << "u has wrong dimension (it should be " +
                                        std::to_string(nu_) + ")");
      }
      return;
    }
    return DifferentialActionModelAbstract::quasiStatic(data, u, x, maxiter,
                                                        tol);
  }

  void default_quasiStatic(
      const std::shared_ptr<DifferentialActionDataAbstract>& data,
      Eigen::Ref<Eigen::VectorXd> u, const Eigen::Ref<const Eigen::VectorXd>& x,
      const std::size_t maxiter, const double tol) {
    return this->DifferentialActionModelAbstract::quasiStatic(data, u, x,
                                                              maxiter, tol);
  }
};

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(
    DifferentialActionModel_quasiStatic_wraps,
    DifferentialActionModelAbstract::quasiStatic_x, 2, 4)

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_DIFF_ACTION_BASE_HPP_
