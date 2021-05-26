
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

class ControlAbstract_wrap : public ControlAbstract, public bp::wrapper<ControlAbstract> {
 public:

  ControlAbstract_wrap(int nu, int np) : ControlAbstract(nu, np), bp::wrapper<ControlAbstract>() {}

  void resize(const std::size_t nu){
    return bp::call<void>(this->get_override("resize").ptr(), nu);
  }

  void value(double t, const Eigen::Ref<const Eigen::VectorXd>& p, Eigen::Ref<Eigen::VectorXd> u_out) const{
    u_out = value_wrap(t, p);
  }

  Eigen::VectorXd value_wrap(double t, const Eigen::Ref<const Eigen::VectorXd>& p) const{
    return bp::call<Eigen::VectorXd>(this->get_override("value").ptr(), t, (Eigen::VectorXd)p);
  }

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

  void dValue(double t, const Eigen::Ref<const Eigen::VectorXd>& p, Eigen::Ref<Eigen::MatrixXd> J_out) const{
    J_out = dValue_wrap(t, p);
  }

  Eigen::MatrixXd dValue_wrap(double t, const Eigen::Ref<const Eigen::VectorXd>& p) const{
    return bp::call<Eigen::MatrixXd>(this->get_override("dValue").ptr(), t, (Eigen::VectorXd)p);
  }

  void multiplyByDValue(double t, const Eigen::Ref<const Eigen::VectorXd>& p, 
                        const Eigen::Ref<const Eigen::MatrixXd>& A, 
                        Eigen::Ref<Eigen::MatrixXd> out) const{
    out = multiplyByDValue_wrap(t, p, A);
  }

  Eigen::MatrixXd multiplyByDValue_wrap(double t, const Eigen::Ref<const Eigen::VectorXd>& p, 
                                        const Eigen::Ref<const Eigen::MatrixXd>& A) const{
    return bp::call<Eigen::MatrixXd>(this->get_override("multiplyByDValue").ptr(), t, (Eigen::VectorXd)p, (Eigen::MatrixXd)A);
  }

  void multiplyDValueTransposeBy(double t, const Eigen::Ref<const Eigen::VectorXd>& p, 
        const Eigen::Ref<const Eigen::MatrixXd>& A, Eigen::Ref<Eigen::MatrixXd> out) const{
    out = multiplyDValueTransposeBy_wrap(t, p, A);
  }

  Eigen::MatrixXd multiplyDValueTransposeBy_wrap(double t, const Eigen::Ref<const Eigen::VectorXd>& p, 
                                                 const Eigen::Ref<const Eigen::MatrixXd>& A) const{
    return bp::call<Eigen::MatrixXd>(this->get_override("multiplyDValueTransposeBy").ptr(), t, (Eigen::VectorXd)p, (Eigen::MatrixXd)A);
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_CONTROL_BASE_HPP_
