///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, LAAS-CNRS, Airbus
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/activations/norm2-barrier.hpp"

namespace crocoddyl {
namespace python {

void exposeActivationNorm2Barrier() {
  
  bp::class_<ActivationModelNorm2Barrier, bp::bases<ActivationModelAbstract> >(
      "ActivationModelNorm2Barrier",
      "Norm2 activation model, with barrier.\n\n"
      "This model activates quadratically if the norm2 of the residual vector r"
      " is inferior to a tunable threshold. In extenso, the result is 0 if norm > threshold" 
      " and 0.5 *(||r|| - threshold)^2 if norm < threshold",
      bp::init<int, double>(bp::args("self", "nr", "threshold"),
                    "Initialize the activation model.\n\n"
                    ":param nr: dimension of the cost-residual vector\n"
                    ":param threshold: activation threshold"))
      .def("calc", &ActivationModelNorm2Barrier::calc, bp::args("self", "data", "r"),
           "Compute the activation value.\n\n"
           ":param data: activation data\n"
           ":param r: residual vector")
      .def("calcDiff", &ActivationModelNorm2Barrier::calcDiff, bp::args("self", "data", "r"),
           "Compute the derivatives of the collision function.\n\n"
           ":param data: activation data\n"
           ":param r: residual vector \n")
      .def("createData", &ActivationModelNorm2Barrier::createData, bp::args("self"),
           "Create the collision activation data.\n\n")
      .add_property("threshold",
                    bp::make_function(&ActivationModelNorm2Barrier::get_threshold,
                                      bp::return_value_policy<bp::return_by_value>()),
                    bp::make_function(&ActivationModelNorm2Barrier::set_threshold),
                    "threshold");
}

}  // namespace python
}  // namespace crocoddyl
