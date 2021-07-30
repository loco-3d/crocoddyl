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
      bp::init<std::size_t, std::size_t>(bp::args("self", "nw", "np"),
                                         "Initialize the control dimensions.\n\n"
                                         ":param nw: dimension of differential control space\n"
                                         ":param np: dimension of control parameter space"))
      .def("createData", &ControlParametrizationModelAbstract_wrap::createData,
           &ControlParametrizationModelAbstract_wrap::default_createData, bp::args("self"),
           "Create the control-parametrization data.\n\n"
           "Each control parametrization model has its own data that needs to be allocated.\n"
           "This function returns the allocated data for a predefined control parametrization model.\n"
           ":return data.")
      .def("calc", pure_virtual(&ControlParametrizationModelAbstract_wrap::calc), bp::args("self", "t", "u"),
           "Compute the differential control value.\n\n"
           ":param data: the data on which the method operates.\n"
           ":param t: normalized time in [0, 1].\n"
           ":param u: control parameters (dim control.nu).")
      .def("calcDiff", pure_virtual(&ControlParametrizationModelAbstract_wrap::calcDiff),
           bp::args("self", "data", "t", "u"),
           "Compute the Jacobian of the differential control with respect to the control parameters.\n"
           "It assumes that calc has been run first.\n\n"
           ":param data: the data on which the method operates.\n"
           ":param t: normalized time in [0, 1].\n"
           ":param u: control parameters (dim control.nu).")
      .def("convertBounds", pure_virtual(&ControlParametrizationModelAbstract_wrap::convertBounds_wrap),
           bp::args("self", "w_lb", "w_ub"),
           "Convert the bounds on the differential control w to bounds on the control parameters.\n\n"
           ":param w_lb: lower bounds on w (dim control.nw).\n"
           ":param w_ub: upper bounds on w (dim control.nw).\n"
           ":return p_lb, p_ub: lower and upper bounds on the control parameters (dim control.nu).")
      .def("params", pure_virtual(&ControlParametrizationModelAbstract_wrap::params),
           bp::args("self", "data", "t", "w"),
           "Compute a value of the parameters p resulting in the specified value of the differential control w.\n\n"
           ":param data: the data on which the method operates.\n"
           ":param t: normalized time in [0, 1].\n"
           ":param w: differential control value (dim control.nw).")
      .def("multiplyByJacobian", pure_virtual(&ControlParametrizationModelAbstract_wrap::multiplyByJacobian_wrap),
           bp::args("self", "t", "u", "A"),
           "Compute the product between the given matrix A and the derivative of the control with respect to the "
           "parameters.\n\n"
           ":param t: normalized time in [0, 1].\n"
           ":param u: control parameters (dim control.nu).\n"
           ":param A: matrix to multiply (dim na x control.nw).\n"
           ":return Product between A and the partial derivative of the calc function (dim na x control.nu).")
      .def(
          "multiplyJacobianTransposeBy",
          pure_virtual(&ControlParametrizationModelAbstract_wrap::multiplyJacobianTransposeBy_wrap),
          bp::args("self", "t", "u", "A"),
          "Compute the product between the transpose of the derivative of the control with respect to the parameters\n"
          "and a given matrix A.\n\n"
          ":param t: normalized time in [0, 1].\n"
          ":param u: control parameters (dim control.nu).\n"
          ":param A: matrix to multiply (dim control.nw x na).\n"
          ":return Product between the partial derivative of the calc function (transposed) and A (dim control.nu x "
          "na).")
      .add_property("nw", bp::make_function(&ControlParametrizationModelAbstract_wrap::get_nw),
                    "dimension of control tuple")
      .add_property("nu", bp::make_function(&ControlParametrizationModelAbstract_wrap::get_nu),
                    "dimension of the control parameters");

  bp::class_<ControlParametrizationDataAbstract, boost::noncopyable>(
      "ControlParametrizationDataAbstract",
      "Abstract class for control parametrization data.\n",
      bp::init<ControlParametrizationModelAbstract*>(bp::args("self", "model"),
                                     "Create common data shared between control parametrization models.\n\n"
                                     ":param model: control parametrization model"))
      .add_property("w", bp::make_getter(&ControlParametrizationDataAbstract::w, bp::return_internal_reference<>()),
                    bp::make_setter(&ControlParametrizationDataAbstract::w), "differential control")
      .add_property("u", bp::make_getter(&ControlParametrizationDataAbstract::u, bp::return_internal_reference<>()),
                    bp::make_setter(&ControlParametrizationDataAbstract::u), "control parameters")
      .add_property("J", bp::make_getter(&ControlParametrizationDataAbstract::J, bp::return_internal_reference<>()),
                    bp::make_setter(&ControlParametrizationDataAbstract::J), "Jacobian of the differential control wrt the control parameters");
}

}  // namespace python
}  // namespace crocoddyl
