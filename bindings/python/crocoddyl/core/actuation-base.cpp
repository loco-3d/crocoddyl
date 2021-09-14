///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/actuation-base.hpp"

namespace crocoddyl {
namespace python {

void exposeActuationAbstract() {
  bp::register_ptr_to_python<boost::shared_ptr<ActuationModelAbstract> >();

  bp::class_<ActuationModelAbstract_wrap, boost::noncopyable>(
      "ActuationModelAbstract",
      "Abstract class for actuation-mapping models.\n\n"
      "An actuation model is a function that maps state x and control inputs u into generalized\n"
      "torques tau, where tau is also named as the actuation signal of our system.\n"
      "The computation of the actuation signal and its partial derivatives are mainly carry on\n"
      "inside calc() and calcDiff(), respectively.",
      bp::init<boost::shared_ptr<StateAbstract>, int>(bp::args("self", "state", "nu"),
                                                      "Initialize the actuation model.\n\n"
                                                      ":param state: state description,\n"
                                                      ":param nu: dimension of control vector"))
      .def("calc", pure_virtual(&ActuationModelAbstract_wrap::calc), bp::args("self", "data", "x", "u"),
           "Compute the actuation signal from the control input u.\n\n"
           "It describes the time-continuos evolution of the actuation model.\n"
           ":param data: actuation data\n"
           ":param x: state point (dim. state.nx)\n"
           ":param u: control input (dim. nu)")
      .def<void (ActuationModelAbstract::*)(const boost::shared_ptr<ActuationDataAbstract>&,
                                            const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ActuationModelAbstract::calc, bp::args("self", "data", "x"),
          "Ignore the computation of the actuation signal.\n\n"
          "It does not update the actuation signal as this function is used in the\n"
          "terminal nodes of an optimal control problem.\n"
          ":param data: actuation data\n"
          ":param x: state point (dim. state.nx)")
      .def("calcDiff", pure_virtual(&ActuationModelAbstract_wrap::calcDiff), bp::args("self", "data", "x", "u"),
           "Compute the Jacobians of the actuation model.\n\n"
           "It computes the partial derivatives of the actuation model which is\n"
           "describes in continouos time.\n"
           ":param data: actuation data\n"
           ":param x: state point (dim. state.nx)\n"
           ":param u: control input (dim. nu)")
      .def<void (ActuationModelAbstract::*)(const boost::shared_ptr<ActuationDataAbstract>&,
                                            const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ActuationModelAbstract::calcDiff, bp::args("self", "data", "x"),
          "Ignore the computation of the Jacobians of the actuation function.\n\n"
          "It does not update the Jacobians of the actuation function as this function\n"
          "is used in the terminal nodes of an optimal control problem.\n"
          ":param data: actuation data\n"
          ":param x: state point (dim. state.nx)")
      .def("createData", &ActuationModelAbstract_wrap::createData, &ActuationModelAbstract_wrap::default_createData,
           bp::args("self"),
           "Create the actuation data.\n\n"
           "Each actuation model (AM) has its own data that needs to be allocated.\n"
           "This function returns the allocated data for a predefined AM.\n"
           ":return AM data.")
      .add_property("nu", bp::make_function(&ActuationModelAbstract_wrap::get_nu), "dimension of control vector")
      .add_property(
          "state",
          bp::make_function(&ActuationModelAbstract_wrap::get_state, bp::return_value_policy<bp::return_by_value>()),
          "state");

  bp::register_ptr_to_python<boost::shared_ptr<ActuationDataAbstract> >();

  bp::class_<ActuationDataAbstract, boost::noncopyable>(
      "ActuationDataAbstract",
      "Abstract class for actuation datas.\n\n"
      "An actuation data contains all the required information for processing an user-defined \n"
      "actuation model. The actuation data typically is allocated onces by running model.createData().",
      bp::init<ActuationModelAbstract*>(bp::args("self", "model"),
                                        "Create common data shared between actuation models.\n\n"
                                        "The actuation data uses the model in order to first process it.\n"
                                        ":param model: actuation model"))
      .add_property("tau", bp::make_getter(&ActuationDataAbstract::tau, bp::return_internal_reference<>()),
                    bp::make_setter(&ActuationDataAbstract::tau), "actuation (generalized force) signal")
      .add_property("dtau_dx", bp::make_getter(&ActuationDataAbstract::dtau_dx, bp::return_internal_reference<>()),
                    bp::make_setter(&ActuationDataAbstract::dtau_dx),
                    "partial derivatives of the actuation model w.r.t. the state point")
      .add_property("dtau_du", bp::make_getter(&ActuationDataAbstract::dtau_du, bp::return_internal_reference<>()),
                    bp::make_setter(&ActuationDataAbstract::dtau_du),
                    "partial derivatives of the actuation model w.r.t. the control input");
}

}  // namespace python
}  // namespace crocoddyl
