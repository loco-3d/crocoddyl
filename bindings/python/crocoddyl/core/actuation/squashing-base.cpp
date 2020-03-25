///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh, IRI: CSIC-UPC
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/actuation/squashing-base.hpp"

namespace crocoddyl {
namespace python {

// BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(SquashingModel_calcDiff_wraps, SquashingModelAbstract::calcDiff_wrap, 2, 3)

void exposeSquashingAbstract() {
  bp::register_ptr_to_python<boost::shared_ptr<SquashingModelAbstract> >();

  bp::class_<SquashingModelAbstract_wrap, boost::noncopyable>(
    "SquashingModelAbstract",
    "Abstract class for squashing functions. \n\n"
    "A squashing function is any sigmoid function that maps from R to a bounded domain\n"
    "Its input can be any value and its output will be a value between a lower bound and an upper bound.\n"
    "The computation of the output value is done using calc() while its derivative is computed using calcDiff(), respectively.",
    bp::init<int>(bp::args("self", "ns"),
                                    "Initialize the squashing model. \n\n"
                                    ":param ns: dimension of the input vector"))
    .def("calc", pure_virtual(&SquashingModelAbstract_wrap::calc), bp::args("self", "data", "u"),
      "Compute the squashing value for a given value of u, component-wise. \n\n"
      ":param data: squashing data\n"
      ":param u: control input")
    .def("calcDiff", pure_virtual(&SquashingModelAbstract_wrap::calcDiff),
      bp::args("self", "data", "u"),
      "Compute the derivative of the squashing function.\n\n"
      ":param data: squashing data\n"
      ":param u: control input")
    .def("createData", &SquashingModelAbstract_wrap::createData, bp::args("self"),
      "Create the squashing data.\n\n")
    .add_property(
          "ns",
          bp::make_function(&SquashingModelAbstract_wrap::get_ns, bp::return_value_policy<bp::return_by_value>()),
          "dimension of control vector")
    .add_property(
          "u_lb",
          bp::make_function(&SquashingModelAbstract_wrap::get_u_lb, bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&SquashingModelAbstract_wrap::set_u_lb),
          "lower bound for the active zone of the squashing function")
    .add_property(
          "u_ub",
          bp::make_function(&SquashingModelAbstract_wrap::get_u_ub, bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&SquashingModelAbstract_wrap::set_u_ub),
          "upper bound for the active zone of the squashing function");
  
  bp::register_ptr_to_python<boost::shared_ptr<SquashingDataAbstract> >();

  bp::class_<SquashingDataAbstract, boost::noncopyable>(
    "SquashingDataAbstract",
    "Abstract class for squashing datas. \n\n"
    "In crocoddyl, an squashing data contains all the required information for processing a \n" 
    "user-defined squashing model. The squashing data is typically allocated once per running.\n"
    "model.createData().",
    bp::init<SquashingModelAbstract*>(bp::args("self", "model"),
                                      "Create common data shared between squashing models. \n\n"
                                      "The squashing data uses the model in order to first process it. \n"
                                      ":param model: squashing model"))
    .add_property("s",
                  bp::make_getter(&SquashingDataAbstract::s,
                  bp::return_value_policy<bp::return_by_value>()),
                  bp::make_setter(&SquashingDataAbstract::s), "squashing-output")
    .add_property("ds_du",
                  bp::make_getter(&SquashingDataAbstract::ds_du,
                  bp::return_value_policy<bp::return_by_value>()),
                  bp::make_setter(&SquashingDataAbstract::ds_du), "Jacobian of the squashing function");
}

} // namespace python
} // namespace crocoddyl 