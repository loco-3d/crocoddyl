
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_CONTROL_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_CONTROL_BASE_HPP_

#include <string>
#include "python/crocoddyl/core/core.hpp"
#include "crocoddyl/core/control-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace python {

class ControlParametrizationModelAbstract_wrap : 
  public ControlParametrizationModelAbstract, 
  public bp::wrapper<ControlParametrizationModelAbstract> {
 public:

  ControlParametrizationModelAbstract_wrap(int nu, int np) : 
    ControlParametrizationModelAbstract(nu, np), bp::wrapper<ControlParametrizationModelAbstract>() {}

  void calc(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, 
            double t, const Eigen::Ref<const Eigen::VectorXd>& p) const {
    if (static_cast<std::size_t>(p.size()) != np_) {
      throw_pretty("Invalid argument: "
                   << "p has wrong dimension (it should be " + std::to_string(np_) + ")");
    }
    return bp::call<void>(this->get_override("calc").ptr(), data, t, (Eigen::VectorXd)p);
  }

  void calcDiff(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, 
                double t, const Eigen::Ref<const Eigen::VectorXd>& p) const {
    if (static_cast<std::size_t>(p.size()) != np_) {
      throw_pretty("Invalid argument: "
                   << "p has wrong dimension (it should be " + std::to_string(np_) + ")");
    }
    return bp::call<void>(this->get_override("calcDiff").ptr(), data, t, (Eigen::VectorXd)p);
  }

  void params(const boost::shared_ptr<ControlParametrizationDataAbstract>& data, 
              double t, const Eigen::Ref<const Eigen::VectorXd>& u) const {
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw_pretty("Invalid argument: "
                   << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
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

  // void value(double t, const Eigen::Ref<const Eigen::VectorXd>& p, Eigen::Ref<Eigen::VectorXd> u_out) const{
  //   u_out = value_wrap(t, p);
  // }

  // Eigen::VectorXd value_wrap(double t, const Eigen::Ref<const Eigen::VectorXd>& p) const{
  //   return bp::call<Eigen::VectorXd>(this->get_override("value").ptr(), t, (Eigen::VectorXd)p);
  // }

  // void value_inv(double t, const Eigen::Ref<const Eigen::VectorXd>& u, Eigen::Ref<Eigen::VectorXd> p_out) const{
  //   p_out = value_inv_wrap(t, u);
  // }

  // Eigen::VectorXd value_inv_wrap(double t, const Eigen::Ref<const Eigen::VectorXd>& u) const{
  //   return bp::call<Eigen::VectorXd>(this->get_override("value_inv").ptr(), t, (Eigen::VectorXd)u);
  // }

  void convert_bounds(const Eigen::Ref<const Eigen::VectorXd>& u_lb, const Eigen::Ref<const Eigen::VectorXd>& u_ub,
                      Eigen::Ref<Eigen::VectorXd> p_lb, Eigen::Ref<Eigen::VectorXd> p_ub) const{
    bp::list res = convert_bounds_wrap(u_lb, u_ub);
    p_lb.derived() = bp::extract<Eigen::VectorXd>(res[0])();
    p_ub.derived() = bp::extract<Eigen::VectorXd>(res[1])();
  }

  bp::list convert_bounds_wrap(const Eigen::Ref<const Eigen::VectorXd>& u_lb, 
                               const Eigen::Ref<const Eigen::VectorXd>& u_ub) const{
    bp::list p_bounds = bp::call<bp::list>(this->get_override("convert_bounds").ptr(), 
                                          (Eigen::VectorXd)u_lb, (Eigen::VectorXd)u_ub);
    return p_bounds;
  }

  // void dValue(double t, const Eigen::Ref<const Eigen::VectorXd>& p, Eigen::Ref<Eigen::MatrixXd> J_out) const{
  //   J_out = dValue_wrap(t, p);
  // }

  // Eigen::MatrixXd dValue_wrap(double t, const Eigen::Ref<const Eigen::VectorXd>& p) const{
  //   return bp::call<Eigen::MatrixXd>(this->get_override("dValue").ptr(), t, (Eigen::VectorXd)p);
  // }

  void multiplyByJacobian(double t, const Eigen::Ref<const Eigen::VectorXd>& p, 
                          const Eigen::Ref<const Eigen::MatrixXd>& A, 
                          Eigen::Ref<Eigen::MatrixXd> out) const{
    out = multiplyByJacobian_wrap(t, p, A);
  }

  Eigen::MatrixXd multiplyByJacobian_wrap(double t, const Eigen::Ref<const Eigen::VectorXd>& p, 
                                          const Eigen::Ref<const Eigen::MatrixXd>& A) const{
    return bp::call<Eigen::MatrixXd>(this->get_override("multiplyByJacobian").ptr(), t, (Eigen::VectorXd)p, (Eigen::MatrixXd)A);
  }

  void multiplyJacobianTransposeBy(double t, const Eigen::Ref<const Eigen::VectorXd>& p, 
        const Eigen::Ref<const Eigen::MatrixXd>& A, Eigen::Ref<Eigen::MatrixXd> out) const{
    out = multiplyJacobianTransposeBy_wrap(t, p, A);
  }

  Eigen::MatrixXd multiplyJacobianTransposeBy_wrap(double t, const Eigen::Ref<const Eigen::VectorXd>& p, 
                                                 const Eigen::Ref<const Eigen::MatrixXd>& A) const{
    return bp::call<Eigen::MatrixXd>(this->get_override("multiplyJacobianTransposeBy").ptr(), t, (Eigen::VectorXd)p, (Eigen::MatrixXd)A);
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_CONTROL_BASE_HPP_
