
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_CONTROL_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_CONTROL_BASE_HPP_

#include "python/crocoddyl/core/core.hpp"
#include "crocoddyl/core/control-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace python {

class ControlParametrizationModelAbstract_wrap : public ControlParametrizationModelAbstract,
                                                 public bp::wrapper<ControlParametrizationModelAbstract> {
 public:
  ControlParametrizationModelAbstract_wrap(std::size_t nu, std::size_t np)
      : ControlParametrizationModelAbstract(nu, np), bp::wrapper<ControlParametrizationModelAbstract>() {}

  void calc(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double t,
            const Eigen::Ref<const Eigen::VectorXd>& p) const {
    if (static_cast<std::size_t>(p.size()) != np_) {
      throw_pretty("Invalid argument: "
                   << "p has wrong dimension (it should be " + std::to_string(np_) + ")");
    }
    return bp::call<void>(this->get_override("calc").ptr(), data, t, (Eigen::VectorXd)p);
  }

  void calcDiff(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double t,
                const Eigen::Ref<const Eigen::VectorXd>& p) const {
    if (static_cast<std::size_t>(p.size()) != np_) {
      throw_pretty("Invalid argument: "
                   << "p has wrong dimension (it should be " + std::to_string(np_) + ")");
    }
    return bp::call<void>(this->get_override("calcDiff").ptr(), data, t, (Eigen::VectorXd)p);
  }

  void params(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, double t,
              const Eigen::Ref<const Eigen::VectorXd>& u) const {
    if (static_cast<std::size_t>(u.size()) != nw_) {
      throw_pretty("Invalid argument: "
                   << "u has wrong dimension (it should be " + std::to_string(nw_) + ")");
    }
    return bp::call<void>(this->get_override("params").ptr(), data, t, (Eigen::VectorXd)u);
  }

  boost::shared_ptr<ControlParametrizationDataAbstract> createData() {
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<boost::shared_ptr<ControlParametrizationDataAbstract> >(createData.ptr());
    }
    return ControlParametrizationModelAbstract::createData();
  }

  boost::shared_ptr<ControlParametrizationDataAbstract> default_createData() {
    return this->ControlParametrizationModelAbstract::createData();
  }

  void convertBounds(const Eigen::Ref<const Eigen::VectorXd>& w_lb, const Eigen::Ref<const Eigen::VectorXd>& w_ub,
                      Eigen::Ref<Eigen::VectorXd> p_lb, Eigen::Ref<Eigen::VectorXd> p_ub) const {
    bp::list res = convertBounds_wrap(w_lb, w_ub);
    p_lb.derived() = bp::extract<Eigen::VectorXd>(res[0])();
    p_ub.derived() = bp::extract<Eigen::VectorXd>(res[1])();
  }

  bp::list convertBounds_wrap(const Eigen::Ref<const Eigen::VectorXd>& w_lb,
                               const Eigen::Ref<const Eigen::VectorXd>& w_ub) const {
    bp::list p_bounds =
        bp::call<bp::list>(this->get_override("convertBounds").ptr(), (Eigen::VectorXd)w_lb, (Eigen::VectorXd)w_ub);
    return p_bounds;
  }

  void multiplyByJacobian(double t, const Eigen::Ref<const Eigen::VectorXd>& p,
                          const Eigen::Ref<const Eigen::MatrixXd>& A, Eigen::Ref<Eigen::MatrixXd> out) const {
    out = multiplyByJacobian_wrap(t, p, A);
  }

  Eigen::MatrixXd multiplyByJacobian_wrap(double t, const Eigen::Ref<const Eigen::VectorXd>& p,
                                          const Eigen::Ref<const Eigen::MatrixXd>& A) const {
    return bp::call<Eigen::MatrixXd>(this->get_override("multiplyByJacobian").ptr(), t, (Eigen::VectorXd)p,
                                     (Eigen::MatrixXd)A);
  }

  void multiplyJacobianTransposeBy(double t, const Eigen::Ref<const Eigen::VectorXd>& p,
                                   const Eigen::Ref<const Eigen::MatrixXd>& A, Eigen::Ref<Eigen::MatrixXd> out) const {
    out = multiplyJacobianTransposeBy_wrap(t, p, A);
  }

  Eigen::MatrixXd multiplyJacobianTransposeBy_wrap(double t, const Eigen::Ref<const Eigen::VectorXd>& p,
                                                   const Eigen::Ref<const Eigen::MatrixXd>& A) const {
    return bp::call<Eigen::MatrixXd>(this->get_override("multiplyJacobianTransposeBy").ptr(), t, (Eigen::VectorXd)p,
                                     (Eigen::MatrixXd)A);
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_CONTROL_BASE_HPP_
