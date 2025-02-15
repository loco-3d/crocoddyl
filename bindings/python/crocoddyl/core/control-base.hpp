
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2022, LAAS-CNRS, University of Edinburgh, University of
// Trento Copyright note valid unless otherwise stated in individual files. All
// rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_CONTROL_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_CONTROL_BASE_HPP_

#include "crocoddyl/core/control-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(
    ControlParametrizationModelAbstract_multiplyByJacobian_J_wrap,
    ControlParametrizationModelAbstract::multiplyByJacobian_J, 2, 3)
BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(
    ControlParametrizationModelAbstract_multiplyJacobianTransposeBy_J_wrap,
    ControlParametrizationModelAbstract::multiplyJacobianTransposeBy_J, 2, 3)

class ControlParametrizationModelAbstract_wrap
    : public ControlParametrizationModelAbstract,
      public bp::wrapper<ControlParametrizationModelAbstract> {
 public:
  ControlParametrizationModelAbstract_wrap(std::size_t nw, std::size_t nu)
      : ControlParametrizationModelAbstract(nw, nu),
        bp::wrapper<ControlParametrizationModelAbstract>() {}

  void calc(const std::shared_ptr<ControlParametrizationDataAbstract>& data,
            double t, const Eigen::Ref<const Eigen::VectorXd>& u) const {
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw_pretty(
          "Invalid argument: " << "u has wrong dimension (it should be " +
                                      std::to_string(nu_) + ")");
    }
    return bp::call<void>(this->get_override("calc").ptr(), data, t,
                          (Eigen::VectorXd)u);
  }

  void calcDiff(const std::shared_ptr<ControlParametrizationDataAbstract>& data,
                double t, const Eigen::Ref<const Eigen::VectorXd>& u) const {
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw_pretty(
          "Invalid argument: " << "u has wrong dimension (it should be " +
                                      std::to_string(nu_) + ")");
    }
    return bp::call<void>(this->get_override("calcDiff").ptr(), data, t,
                          (Eigen::VectorXd)u);
  }

  void params(const std::shared_ptr<ControlParametrizationDataAbstract>& data,
              double t, const Eigen::Ref<const Eigen::VectorXd>& w) const {
    if (static_cast<std::size_t>(w.size()) != nw_) {
      throw_pretty(
          "Invalid argument: " << "w has wrong dimension (it should be " +
                                      std::to_string(nw_) + ")");
    }
    return bp::call<void>(this->get_override("params").ptr(), data, t,
                          (Eigen::VectorXd)w);
  }

  std::shared_ptr<ControlParametrizationDataAbstract> createData() {
    enableMultithreading() = false;
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<std::shared_ptr<ControlParametrizationDataAbstract> >(
          createData.ptr());
    }
    return ControlParametrizationModelAbstract::createData();
  }

  std::shared_ptr<ControlParametrizationDataAbstract> default_createData() {
    return this->ControlParametrizationModelAbstract::createData();
  }

  void convertBounds(const Eigen::Ref<const Eigen::VectorXd>& w_lb,
                     const Eigen::Ref<const Eigen::VectorXd>& w_ub,
                     Eigen::Ref<Eigen::VectorXd> u_lb,
                     Eigen::Ref<Eigen::VectorXd> u_ub) const {
    bp::list res = convertBounds_wrap(w_lb, w_ub);
    u_lb.derived() = bp::extract<Eigen::VectorXd>(res[0])();
    u_ub.derived() = bp::extract<Eigen::VectorXd>(res[1])();
  }

  bp::list convertBounds_wrap(
      const Eigen::Ref<const Eigen::VectorXd>& w_lb,
      const Eigen::Ref<const Eigen::VectorXd>& w_ub) const {
    bp::list p_bounds =
        bp::call<bp::list>(this->get_override("convertBounds").ptr(),
                           (Eigen::VectorXd)w_lb, (Eigen::VectorXd)w_ub);
    return p_bounds;
  }

  void multiplyByJacobian(
      const std::shared_ptr<ControlParametrizationDataAbstract>& data,
      const Eigen::Ref<const Eigen::MatrixXd>& A,
      Eigen::Ref<Eigen::MatrixXd> out, const AssignmentOp op) const {
    switch (op) {
      case setto: {
        out = multiplyByJacobian_wrap(data, A);
        break;
      }
      case addto: {
        out += multiplyByJacobian_wrap(data, A);
        break;
      }
      case rmfrom: {
        out -= multiplyByJacobian_wrap(data, A);
        break;
      }
      default: {
        throw_pretty(
            "Invalid argument: allowed operators: setto, addto, rmfrom");
        break;
      }
    }
  }

  Eigen::MatrixXd multiplyByJacobian_wrap(
      const std::shared_ptr<ControlParametrizationDataAbstract>& data,
      const Eigen::Ref<const Eigen::MatrixXd>& A) const {
    return bp::call<Eigen::MatrixXd>(
        this->get_override("multiplyByJacobian").ptr(), data,
        (Eigen::MatrixXd)A);
  }

  void multiplyJacobianTransposeBy(
      const std::shared_ptr<ControlParametrizationDataAbstract>& data,
      const Eigen::Ref<const Eigen::MatrixXd>& A,
      Eigen::Ref<Eigen::MatrixXd> out, const AssignmentOp op) const {
    switch (op) {
      case setto: {
        out = multiplyJacobianTransposeBy_wrap(data, A);
        break;
      }
      case addto: {
        out += multiplyJacobianTransposeBy_wrap(data, A);
        break;
      }
      case rmfrom: {
        out -= multiplyJacobianTransposeBy_wrap(data, A);
        break;
      }
      default: {
        throw_pretty(
            "Invalid argument: allowed operators: setto, addto, rmfrom");
        break;
      }
    }
  }

  Eigen::MatrixXd multiplyJacobianTransposeBy_wrap(
      const std::shared_ptr<ControlParametrizationDataAbstract>& data,
      const Eigen::Ref<const Eigen::MatrixXd>& A) const {
    return bp::call<Eigen::MatrixXd>(
        this->get_override("multiplyJacobianTransposeBy").ptr(), data,
        (Eigen::MatrixXd)A);
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_CONTROL_BASE_HPP_
