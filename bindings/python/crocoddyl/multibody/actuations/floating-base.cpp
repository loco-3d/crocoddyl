///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/actuations/floating-base.hpp"

namespace crocoddyl {
namespace python {

void exposeActuationFloatingBase() {
  bp::class_<ActuationModelFloatingBase, bp::bases<ActuationModelAbstract> >(
      "ActuationModelFloatingBase",
      "Floating-base actuation models.\n\n"
      "It simplies consider a floating-base actuation model, where the first 6 elements are unactuated.",
      bp::init<boost::shared_ptr<StateMultibody> >(bp::args("self", "state"),
                                                   "Initialize the floating-base actuation model.\n\n"
                                                   ":param state: state of multibody system"))
      .def("calc", &ActuationModelFloatingBase::calc, bp::args("self", "data", "x", "u"),
           "Compute the actuation signal from the control input u.\n\n"
           "It describes the time-continuos evolution of the floating-base actuation model.\n"
           ":param data: floating-base actuation data\n"
           ":param x: state vector\n"
           ":param u: control input")
      .def("calcDiff", &ActuationModelFloatingBase::calcDiff, bp::args("self", "data", "x", "u"),
           "Compute the derivatives of the actuation model.\n\n"
           "It computes the partial derivatives of the floating-base actuation. It assumes that you\n"
           "create the data using this class. The reason is that the derivatives are constant and\n"
           "defined in createData. The derivatives are constant, so we don't write again these values.\n"
           ":param data: floating-base actuation data\n"
           ":param x: state vector\n"
           ":param u: control input\n")
      .def("createData", &ActuationModelFloatingBase::createData, bp::args("self"),
           "Create the floating-base actuation data.\n\n"
           "Each actuation model (AM) has its own data that needs to be allocated.\n"
           "This function returns the allocated data for a predefined AM.\n"
           ":return AM data.");
}

}  // namespace python
}  // namespace crocoddyl
