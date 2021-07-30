///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/control-base.hpp"
#include "crocoddyl/core/controls/poly-zero.hpp"

namespace crocoddyl {
namespace python {

void exposeControlParametrizationPolyZero() {
  bp::register_ptr_to_python<boost::shared_ptr<ControlParametrizationModelPolyZero> >();

  bp::class_<ControlParametrizationModelPolyZero, bp::bases<ControlParametrizationModelAbstract> >(
      "ControlParametrizationModelPolyZero",
      "Constant control.\n\n"
      "This control is a line function of time (normalized in [0,1])."
      "The first half of the parameter vector contains the initial value of u, "
      "whereas the second half contains the value of u at t=0.5.",
      bp::init<std::size_t>(bp::args("self", "nu"),
                            "Initialize the control dimensions.\n\n"
                            ":param nu: dimension of control space\n"))
      .def<void (ControlParametrizationModelPolyZero::*)(const boost::shared_ptr<ControlParametrizationDataAbstract>&,
                                                         double, const Eigen::Ref<const Eigen::VectorXd>&) const>(
          "calc", &ControlParametrizationModelPolyZero::calc, bp::args("self", "data", "t", "p"),
          "Compute the control value.\n\n"
          ":param data: the data on which the method operates.\n"
          ":param t: normalized time in [0, 1].\n"
          ":param p: control parameters (dim control.np).")
      .def<void (ControlParametrizationModelPolyZero::*)(const boost::shared_ptr<ControlParametrizationDataAbstract>&,
                                                         double, const Eigen::Ref<const Eigen::VectorXd>&) const>(
          "params", &ControlParametrizationModelPolyZero::params, bp::args("self", "data", "t", "u"),
          "Compute the control parameters.\n\n"
          ":param data: the data on which the method operates.\n"
          ":param t: normalized time in [0, 1].\n"
          ":param u: control value (dim control.nu).")
      .def("convert_bounds", &ControlParametrizationModelPolyZero::convert_bounds, bp::args("self", "u_lb", "u_ub"),
           "Convert the bounds on the control to bounds on the control parameters.\n\n"
           ":param u_lb: lower bounds on u (dim control.nu).\n"
           ":param u_ub: upper bounds on u (dim control.nu).\n"
           ":return p_lb, p_ub: lower and upper bounds on the control parameters (dim control.np).")
      .def<void (ControlParametrizationModelPolyZero::*)(const boost::shared_ptr<ControlParametrizationDataAbstract>&,
                                                         double, const Eigen::Ref<const Eigen::VectorXd>&) const>(
          "calcDiff", &ControlParametrizationModelPolyZero::calcDiff, bp::args("self", "data", "t", "p"),
          "Compute the Jacobian of the control value with respect to the control parameters.\n"
          "It assumes that calc has been run first.\n\n"
          ":param data: the data on which the method operates.\n"
          ":param t: normalized time in [0, 1].\n"
          ":param p: control parameters (dim control.np).")
      .def("multiplyByJacobian", &ControlParametrizationModelPolyZero::multiplyByJacobian_J,
           bp::args("self", "t", "p", "A"),
           "Compute the product between the given matrix A and the derivative of the control with respect to the "
           "parameters.\n\n"
           ":param t: normalized time in [0, 1].\n"
           ":param p: control parameters (dim control.np).\n"
           ":param A: matrix to multiply (dim na x control.nu).\n"
           ":return Product between A and the partial derivative of the value function (dim na x control.np).")
      .def(
          "multiplyJacobianTransposeBy", &ControlParametrizationModelPolyZero::multiplyJacobianTransposeBy_J,
          bp::args("self", "t", "p", "A"),
          "Compute the product between the transpose of the derivative of the control with respect to the parameters\n"
          "and a given matrix A.\n\n"
          ":param t: normalized time in [0, 1].\n"
          ":param p: control parameters (dim control.np).\n"
          ":param A: matrix to multiply (dim control.nu x na).\n"
          ":return Product between the partial derivative of the value function (transposed) and A (dim control.np x "
          "na).")
      .add_property("nw", bp::make_function(&ControlParametrizationModelPolyZero::get_nw),
                    "dimension of control tuple")
      .add_property("np", bp::make_function(&ControlParametrizationModelPolyZero::get_np),
                    "dimension of the control parameters");
}

}  // namespace python
}  // namespace crocoddyl
