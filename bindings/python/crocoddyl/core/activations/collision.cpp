///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, LAAS-CNRS, Airbus
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/activations/collision.hpp"

namespace crocoddyl {
namespace python {

void exposeActivationCollision() {
  bp::class_<ActivationThreshold>("ActivationThreshold",
                               "Activation threshold.\n\n"
                               "The threshold is the barrier of activation.",
                               bp::init<bp::optional<double> >(
                                   bp::args("self", "threshold"),
                                   "Initialize the activation threshold.\n\n"
                                   ":param threshold: threshold of activation (positive, default: 0.3)"))
      .add_property("threshold", &ActivationThreshold::threshold, "threshold");
  
  bp::class_<ActivationModelCollision, bp::bases<ActivationModelAbstract> >(
      "ActivationModelCollision",
      "Collision pair activation model.\n\n"
      "This model activates quadratically if the distance (norm of residual vector) between the"
      " objects is inferior to a tunable threshold. In extenso, the result is 0 if distance > threshold" 
      " and 0.5 *(||r|| - threshold)^2 if distance < threshold",
      bp::init<int, ActivationThreshold>(bp::args("self", "nr", "threshold"),
                    "Initialize the activation model.\n\n"
                    ":param nr: dimension of the cost-residual vector\n"
                    ":param threshold: activation threshold"))
      .def("calc", &ActivationModelCollision::calc, bp::args("self", "data", "r"),
           "Compute the activation value.\n\n"
           ":param data: activation data\n"
           ":param r: residual vector")
      .def("calcDiff", &ActivationModelCollision::calcDiff, bp::args("self", "data", "r"),
           "Compute the derivatives of the collision function.\n\n"
           ":param data: activation data\n"
           ":param r: residual vector \n")
      .def("createData", &ActivationModelCollision::createData, bp::args("self"),
           "Create the collision activation data.\n\n")
      .add_property("threshold",
                    bp::make_function(&ActivationModelCollision::get_threshold,
                                      bp::return_value_policy<bp::return_by_value>()),
                    bp::make_function(&ActivationModelCollision::set_threshold),
                    "threshold");
}

}  // namespace python
}  // namespace crocoddyl
