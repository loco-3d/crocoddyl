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

void exposeControlParametrizationAbstract() {
  bp::register_ptr_to_python<boost::shared_ptr<ControlParametrizationModelAbstract> >();

  bp::class_<ControlParametrizationModelAbstract_wrap, boost::noncopyable>(
      "ControlParametrizationModelAbstract",
      "Abstract class for the control parametrization.\n\n"
      "A control is a function of time (normalized in [0,1]) and the control parameters p.",
      bp::init<std::size_t, std::size_t>(bp::args("self", "nu", "np"),
                                         "Initialize the control dimensions.\n\n"
                                         ":param nu: dimension of control space\n"
                                         ":param np: dimension of control parameter space"))
      .def("createData", &ControlParametrizationModelAbstract_wrap::createData,
           &ControlParametrizationModelAbstract_wrap::default_createData, bp::args("self"),
           "Create the control-parametrization data.\n\n"
           "Each control parametrization model has its own data that needs to be allocated.\n"
           "This function returns the allocated data for a predefined control parametrization model.\n"
           ":return data.")
      .def("calc", pure_virtual(&ControlParametrizationModelAbstract_wrap::calc), bp::args("self", "t", "p"),
           "Compute the control value.\n\n"
           ":param data: the data on which the method operates.\n"
           ":param t: normalized time in [0, 1].\n"
           ":param p: control parameters (dim control.np).")
      .def("calcDiff", pure_virtual(&ControlParametrizationModelAbstract_wrap::calcDiff),
           bp::args("self", "data", "t", "p"),
           "Compute a value of the control parameters corresponding to the given control value.\n\n"
           ":param data: the data on which the method operates.\n"
           ":param t: normalized time in [0, 1].\n"
           ":param u: control parameters (dim control.nu).")
      .def("convert_bounds", pure_virtual(&ControlParametrizationModelAbstract_wrap::convert_bounds_wrap),
           bp::args("self", "u_lb", "u_ub"),
           "Convert the bounds on the control to bounds on the control parameters.\n\n"
           ":param u_lb: lower bounds on u (dim control.nu).\n"
           ":param u_ub: upper bounds on u (dim control.nu).\n"
           ":return p_lb, p_ub: lower and upper bounds on the control parameters (dim control.np).")
      .def("multiplyByJacobian", pure_virtual(&ControlParametrizationModelAbstract_wrap::multiplyByJacobian_wrap),
           bp::args("self", "t", "p", "A"),
           "Compute the product between the given matrix A and the derivative of the control with respect to the "
           "parameters.\n\n"
           ":param t: normalized time in [0, 1].\n"
           ":param p: control parameters (dim control.np).\n"
           ":param A: matrix to multiply (dim na x control.nu).\n"
           ":return Product between A and the partial derivative of the value function (dim na x control.np).")
      .def(
          "multiplyJacobianTransposeBy",
          pure_virtual(&ControlParametrizationModelAbstract_wrap::multiplyJacobianTransposeBy_wrap),
          bp::args("self", "t", "p", "A"),
          "Compute the product between the transpose of the derivative of the control with respect to the parameters\n"
          "and a given matrix A.\n\n"
          ":param t: normalized time in [0, 1].\n"
          ":param p: control parameters (dim control.np).\n"
          ":param A: matrix to multiply (dim control.nu x na).\n"
          ":return Product between the partial derivative of the value function (transposed) and A (dim control.np x "
          "na).")
      .add_property("nu", bp::make_function(&ControlParametrizationModelAbstract_wrap::get_nu),
                    "dimension of control tuple")
      .add_property("np", bp::make_function(&ControlParametrizationModelAbstract_wrap::get_np),
                    "dimension of the control parameters");
}

}  // namespace python
}  // namespace crocoddyl
