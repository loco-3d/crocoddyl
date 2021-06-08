///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh, University of Trento
// Copyright note valid unless otherwise controld in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/control-base.hpp"

namespace crocoddyl {
namespace python {

void exposeControlAbstract() {
  bp::register_ptr_to_python<boost::shared_ptr<ControlAbstract> >();

  bp::class_<ControlAbstract_wrap, boost::noncopyable>(
      "ControlAbstract",
      "Abstract class for the control representation.\n\n"
      "A control is a function of time (normalized in [0,1]) and the control parameters p.",
      bp::init<int, int>(bp::args("self", "nu", "np"),
                         "Initialize the control dimensions.\n\n"
                         ":param nu: dimension of control space\n"
                         ":param np: dimension of control parameter space"))
      .def("value", pure_virtual(&ControlAbstract_wrap::value_wrap), bp::args("self", "t", "p"),
           "Compute the control value.\n\n"
           ":param t: normalized time in [0, 1].\n"
           ":param p: control parameters (dim control.np).\n"
           ":return u value (dim control.nu).")
      .def("value_inv", pure_virtual(&ControlAbstract_wrap::value_inv_wrap), bp::args("self", "t", "u"),
           "Compute a value of the control parameters corresponding to the given control value.\n\n"
           ":param t: normalized time in [0, 1].\n"
           ":param u: control parameters (dim control.nu).\n"
           ":return p value (dim control.np).")
      .def("convert_bounds", pure_virtual(&ControlAbstract_wrap::convert_bounds_wrap), bp::args("self", "u_lb", "u_ub"),
           "Convert the bounds on the control to bounds on the control parameters.\n\n"
           ":param u_lb: lower bounds on u (dim control.nu).\n"
           ":param u_ub: upper bounds on u (dim control.nu).\n"
           ":return p_lb, p_ub: lower and upper bounds on the control parameters (dim control.np).")
      .def("dValue", pure_virtual(&ControlAbstract_wrap::dValue_wrap), bp::args("self", "t", "p"),
           "Compute the derivative of the control with respect to the parameters.\n\n"
           ":param t: normalized time in [0, 1].\n"
           ":param p: control parameters (dim control.np).\n"
           ":return Partial derivative of the value function (dim control.nu x control.np).")
      .def("multiplyByDValue", pure_virtual(&ControlAbstract_wrap::multiplyByDValue_wrap), bp::args("self", "t", "p", "A"),
           "Compute the product between the given matrix A and the derivative of the control with respect to the parameters.\n\n"
           ":param t: normalized time in [0, 1].\n"
           ":param p: control parameters (dim control.np).\n"
           ":param A: matrix to multiply (dim na x control.nu).\n"
           ":return Product between A and the partial derivative of the value function (dim na x control.np).")
      .def("multiplyDValueTransposeBy", pure_virtual(&ControlAbstract_wrap::multiplyDValueTransposeBy_wrap), bp::args("self", "t", "p", "A"),
           "Compute the product between the transpose of the derivative of the control with respect to the parameters\n"
           "and a given matrix A.\n\n"
           ":param t: normalized time in [0, 1].\n"
           ":param p: control parameters (dim control.np).\n"
           ":param A: matrix to multiply (dim control.nu x na).\n"
           ":return Product between the partial derivative of the value function (transposed) and A (dim control.np x na).")
      .add_property("nu", bp::make_function(&ControlAbstract_wrap::get_nu), "dimension of control tuple")
      .add_property("np", bp::make_function(&ControlAbstract_wrap::get_np),
                    "dimension of the control parameters");
}

}  // namespace python
}  // namespace crocoddyl
