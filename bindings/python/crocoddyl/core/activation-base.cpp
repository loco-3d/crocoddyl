///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/activation-base.hpp"

namespace crocoddyl {
namespace python {

void exposeActivationAbstract() {
  bp::register_ptr_to_python<boost::shared_ptr<ActivationModelAbstract> >();

  bp::class_<ActivationModelAbstract_wrap, boost::noncopyable>(
      "ActivationModelAbstract",
      "Abstract class for activation models.\n\n"
      "In crocoddyl, an activation model takes the residual vector and computes the activation\n"
      "value and its derivatives from it. Activation value and its derivatives are computed by\n"
      "calc() and calcDiff(), respectively.",
      bp::init<int>(bp::args("self", "nr"),
                    "Initialize the activation model.\n\n"
                    ":param nr: dimension of the cost-residual vector"))
      .def("calc", pure_virtual(&ActivationModelAbstract_wrap::calc), bp::args("self", "data", "r"),
           "Compute the activation value.\n\n"
           ":param data: activation data\n"
           ":param r: residual vector")
      .def("calcDiff", pure_virtual(&ActivationModelAbstract_wrap::calcDiff), bp::args("self", "data", "r"),
           "Compute the derivatives of the residual.\n\n"
           "It computes the partial derivatives of the residual vector function\n"
           ":param data: activation data\n"
           ":param r: residual vector \n")
      .def("createData", &ActivationModelAbstract_wrap::createData, bp::args("self"),
           "Create the activation data.\n\n")
      .add_property(
          "nr",
          bp::make_function(&ActivationModelAbstract_wrap::get_nr, bp::return_value_policy<bp::return_by_value>()),
          "dimension of cost-residual vector");

  bp::register_ptr_to_python<boost::shared_ptr<ActivationDataAbstract> >();

  bp::class_<ActivationDataAbstract, boost::noncopyable>(
      "ActivationDataAbstract", "Abstract class for activation data.\n\n",
      bp::init<ActivationModelAbstract*>(bp::args("self", "model"),
                                         "Create common data shared between AMs.\n\n"
                                         "The action data uses the model in order to first process it.\n"
                                         ":param model: action model"))
      .add_property("a",
                    bp::make_getter(&ActivationDataAbstract::a_value, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActivationDataAbstract::a_value), "cost value")
      .add_property("Ar", bp::make_getter(&ActivationDataAbstract::Ar, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActivationDataAbstract::Ar), "Jacobian of the residual")
      .add_property("Arr",
                    bp::make_getter(&ActivationDataAbstract::Arr, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActivationDataAbstract::Arr), "Hessian of the residual");
}

}  // namespace python
}  // namespace crocoddyl
