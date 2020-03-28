///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh, IRI: CSIC-UPC
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"

#include "crocoddyl/core/actuation/actuation-squashing.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace python {

void exposeActuationSquashing() {
  bp::register_ptr_to_python<boost::shared_ptr<ActuationSquashingModelAbstract> >();

  bp::class_<ActuationSquashingModel, bp::bases<ActuationModelAbstract> >(
      "ActuationSquashingModel",
      "Class for squashing an actuation model.\n\n",
      bp::init<boost::shared_ptr<StateMultibody>,boost::shared_ptr<SquashingModelAbstract>,int>(bp::args("self", "state","squashing","nu"),
                                                   "Initialize the actuation model with squashing function.\n\n"
                                                   ":param state: state of multibody system,\n"
                                                   ":param squashing: squashjng function,\n"
                                                   ":param nu: number of controls"))
      .def("calc", pure_virtual(&ActuationModelSquashingAbstract_wrap::calc), bp::args("self", "data", "x", "u"),
           "Compute the actuation signal from the control input u.\n\n"
           "It describes the time-continuos evolution of the actuation model.\n"
           ":param data: actuation data\n"
           ":param x: state vector\n"
           ":param u: control input")
      .def("calcDiff", pure_virtual(&ActuationModelSquashingAbstract_wrap::calcDiff),
           bp::args("self", "data", "x", "u", "recalc"),
           "Compute the derivatives of the actuation model.\n\n"
           "It computes the partial derivatives of the actuation model which is\n"
           "describes in continouos time.\n"
           ":param data: actuation data\n"
           ":param x: state vector\n"
           ":param u: control input\n"
           ":param recalc: If true, it updates the actuation signal.")
      .def("createData", &ActuationModelSquashingAbstract_wrap::createData, bp::args("self"),
           "Create the actuation squashing abstract actuation data.\n\n"
           "Each actuation model (AM) has its own data that needs to be allocated.\n"
           "This function returns the allocated data for a predefined AM.\n"
           ":return AM data.")
      .add_property(
          "squashing",
          bp::make_function(&ActuationModelSquashingAbstract_wrap::get_squashing, bp::return_value_policy<bp::return_by_value>()),
          "squashing");

  bp::register_ptr_to_python<boost::shared_ptr<ActuationDataSquashing> >();

  bp::class_<ActuationDataSquashing, bp::bases<ActuationDataAbstract> >(
      "ActuationDataSquashing",
      "Abstract class for actuation datas.\n\n"
      "In crocoddyl, an actuation data contains all the required information for processing an\n"
      "user-defined actuation model. The actuation data typically is allocated onces by running\n"
      "model.createData().",
      bp::init<ActuationModelSquashingAbstract*>(bp::args("self", "model"),
                                        "Create common data shared between actuation models.\n\n"
                                        "The actuation data uses the model in order to first process it.\n"
                                        ":param model: actuation model"))
      .add_property("squashing",
                    bp::make_getter(&ActuationDataSquashing::squashing, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActuationDataSquashing::squashing), "Jacobian of the actuation model");

}

} // namespace python
} // namespace crocoddyl

#endif