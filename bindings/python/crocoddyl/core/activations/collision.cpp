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
  bp::class_<ActivationModelCollision, bp::bases<ActivationModelAbstract> >(
      "ActivationModelCollision",
      "Collision pair activation model.\n\n"
      "This model activates quadratically if the distance (norm of residual vector) between the"
      " objects is inferior to a tunable threshold. In extenso, the result is 0 if distance > threshold" 
      " and 0.5 *(||r|| - threshold)^2 if distance < threshold",
      bp::init<int, double>(bp::args("self", "nr", "threshold"),
                    "Initialize the activation model.\n\n"
                    ":param nr: dimension of the cost-residual vector\n"
                    ":param threshold: activation distance threshold (default: 0.3m)"))
      .def("calc", &ActivationModelCollision::calc, bp::args("self", "data", "r"),
           "Compute the activation value.\n\n"
           ":param data: activation data\n"
           ":param r: residual vector")
      .def("calcDiff", &ActivationModelCollision::calcDiff, bp::args("self", "data", "r"),
           "Compute the derivatives of the collision function.\n\n"
           ":param data: activation data\n"
           ":param r: residual vector \n")
      .def("createData", &ActivationModelCollision::createData, bp::args("self"),
           "Create the collision activation data.\n\n");
}

}  // namespace python
}  // namespace crocoddyl
