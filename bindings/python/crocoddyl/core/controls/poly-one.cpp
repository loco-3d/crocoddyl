///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/control-base.hpp"
#include "crocoddyl/core/controls/poly-one.hpp"

namespace crocoddyl {
namespace python {

void exposeControlParametrizationPolyOne() {
  bp::register_ptr_to_python<boost::shared_ptr<ControlParametrizationModelPolyOne> >();

  bp::class_<ControlParametrizationModelPolyOne, bp::bases<ControlParametrizationModelAbstract> >(
      "ControlParametrizationModelPolyOne",
      "Constant control.\n\n"
      "This control is a line function of time (normalized in [0,1])."
      "The first half of the parameter vector contains the initial value of u, "
      "whereas the second half contains the value of u at t=0.5.",
      bp::init<std::size_t>(bp::args("self", "nu"),
                            "Initialize the control dimensions.\n\n"
                            ":param nu: dimension of control space\n"))
      .def<void (ControlParametrizationModelPolyOne::*)(const boost::shared_ptr<ControlParametrizationDataAbstract>&,
                                                        double, const Eigen::Ref<const Eigen::VectorXd>&) const>(
          "calc", &ControlParametrizationModelPolyOne::calc, bp::args("self", "data", "t", "p"),
          "Compute the control value.\n\n"
          ":param data: the data on which the method operates.\n"
          ":param t: normalized time in [0, 1].\n"
          ":param p: control parameters (dim control.np).")
      .def<void (ControlParametrizationModelPolyOne::*)(const boost::shared_ptr<ControlParametrizationDataAbstract>&,
                                                        double, const Eigen::Ref<const Eigen::VectorXd>&) const>(
          "params", &ControlParametrizationModelPolyOne::params, bp::args("self", "data", "t", "u"),
          "Compute the control parameters.\n\n"
          ":param data: the data on which the method operates.\n"
          ":param t: normalized time in [0, 1].\n"
          ":param u: control value (dim control.nu).")
      .def("convert_bounds", &ControlParametrizationModelPolyOne::convert_bounds, bp::args("self", "u_lb", "u_ub"),
           "Convert the bounds on the control to bounds on the control parameters.\n\n"
           ":param u_lb: lower bounds on u (dim control.nu).\n"
           ":param u_ub: upper bounds on u (dim control.nu).\n"
           ":return p_lb, p_ub: lower and upper bounds on the control parameters (dim control.np).")
      .def<void (ControlParametrizationModelPolyOne::*)(const boost::shared_ptr<ControlParametrizationDataAbstract>&,
                                                        double, const Eigen::Ref<const Eigen::VectorXd>&) const>(
          "calcDiff", &ControlParametrizationModelPolyOne::calcDiff, bp::args("self", "data", "t", "p"),
          "Compute the derivative of the control with respect to the parameters.\n\n"
          ":param data: the data on which the method operates.\n"
          ":param t: normalized time in [0, 1].\n"
          ":param p: control parameters (dim control.np).")
      .def("multiplyByJacobian", &ControlParametrizationModelPolyOne::multiplyByJacobian_J,
           bp::args("self", "t", "p", "A"),
           "Compute the product between the given matrix A and the derivative of the control with respect to the "
           "parameters.\n\n"
           ":param t: normalized time in [0, 1].\n"
           ":param p: control parameters (dim control.np).\n"
           ":param A: matrix to multiply (dim na x control.nu).\n"
           ":return Product between A and the partial derivative of the value function (dim na x control.np).")
      .def(
          "multiplyJacobianTransposeBy", &ControlParametrizationModelPolyOne::multiplyJacobianTransposeBy_J,
          bp::args("self", "t", "p", "A"),
          "Compute the product between the transpose of the derivative of the control with respect to the parameters\n"
          "and a given matrix A.\n\n"
          ":param t: normalized time in [0, 1].\n"
          ":param p: control parameters (dim control.np).\n"
          ":param A: matrix to multiply (dim control.nu x na).\n"
          ":return Product between the partial derivative of the value function (transposed) and A (dim control.np x "
          "na).")
      .add_property("nu", bp::make_function(&ControlParametrizationModelPolyOne::get_nu), "dimension of control tuple")
      .add_property("np", bp::make_function(&ControlParametrizationModelPolyOne::get_np),
                    "dimension of the control parameters");
}

}  // namespace python
}  // namespace crocoddyl
