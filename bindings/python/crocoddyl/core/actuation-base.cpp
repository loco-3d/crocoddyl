///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "crocoddyl/core/actuation-base.hpp"
#include "python/crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace python {

void exposeActuationAbstract() {
  bp::register_ptr_to_python<boost::shared_ptr<ActuationModelAbstract> >();

  bp::class_<ActuationModelAbstract_wrap, boost::noncopyable>(
      "ActuationModelAbstract",
      "Abstract class for actuation models.\n\n"
      "In crocoddyl, an actuation model is a block that takes u and outputs a (in continouos\n"
      "time), where a is the actuation signal of our system, and it also computes the derivatives\n"
      "of this model. These computations are mainly carry on inside calc() and calcDiff(),\n"
      "respectively.",
      bp::init<boost::shared_ptr<StateAbstract>, int>(bp::args("self", "state", "nu"),
                                                      "Initialize the actuation model.\n\n"
                                                      ":param state: state description,\n"
                                                      ":param nu: dimension of control vector"))
      .def("calc", pure_virtual(&ActuationModelAbstract_wrap::calc), bp::args("self", "data", "x", "u"),
           "Compute the actuation signal from the control input u.\n\n"
           "It describes the time-continuos evolution of the actuation model.\n"
           ":param data: actuation data\n"
           ":param x: state vector\n"
           ":param u: control input")
      .def("calcDiff", pure_virtual(&ActuationModelAbstract_wrap::calcDiff), bp::args("self", "data", "x", "u"),
           "Compute the derivatives of the actuation model.\n\n"
           "It computes the partial derivatives of the actuation model which is\n"
           "describes in continouos time.\n"
           ":param data: actuation data\n"
           ":param x: state vector\n"
           ":param u: control input\n")
      .def("createData", &ActuationModelAbstract_wrap::createData, bp::args("self"),
           "Create the actuation data.\n\n"
           "Each actuation model (AM) has its own data that needs to be allocated.\n"
           "This function returns the allocated data for a predefined AM.\n"
           ":return AM data.")
      .add_property(
          "nu",
          bp::make_function(&ActuationModelAbstract_wrap::get_nu, bp::return_value_policy<bp::return_by_value>()),
          "dimension of control vector")
      .add_property(
          "state",
          bp::make_function(&ActuationModelAbstract_wrap::get_state, bp::return_value_policy<bp::return_by_value>()),
          "state");

  bp::register_ptr_to_python<boost::shared_ptr<ActuationDataAbstract> >();

  bp::class_<ActuationDataAbstract, boost::noncopyable>(
      "ActuationDataAbstract",
      "Abstract class for actuation datas.\n\n"
      "In crocoddyl, an actuation data contains all the required information for processing an\n"
      "user-defined actuation model. The actuation data typically is allocated onces by running\n"
      "model.createData().",
      bp::init<ActuationModelAbstract*>(bp::args("self", "model"),
                                        "Create common data shared between actuation models.\n\n"
                                        "The actuation data uses the model in order to first process it.\n"
                                        ":param model: actuation model"))
      .add_property("tau",
                    bp::make_getter(&ActuationDataAbstract::tau, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActuationDataAbstract::tau), "actuation-force signal")
      .add_property("dtau_dx",
                    bp::make_getter(&ActuationDataAbstract::dtau_dx, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActuationDataAbstract::dtau_dx), "Jacobian of the actuation model")
      .add_property("dtau_du",
                    bp::make_getter(&ActuationDataAbstract::dtau_du, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActuationDataAbstract::dtau_du), "Jacobian of the actuation model");
}

}  // namespace python
}  // namespace crocoddyl
