///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/actuations/full.hpp"

namespace crocoddyl {
namespace python {

void exposeActuationFull() {
  bp::class_<ActuationModelFull, bp::bases<ActuationModelAbstract> >(
      "ActuationModelFull", "Full actuation models.",
      bp::init<boost::shared_ptr<StateMultibody> >(bp::args("self", "state"),
                                                   "Initialize the full actuation model.\n\n"
                                                   ":param state: state of multibody system"))
      .def("calc", &ActuationModelFull::calc, bp::args("self", "data", "x", "u"),
           "Compute the actuation signal from the control input u.\n\n"
           ":param data: full actuation data\n"
           ":param x: state vector\n"
           ":param u: control input")
      .def("calcDiff", &ActuationModelFull::calcDiff, bp::args("self", "data", "x", "u"),
           "Compute the derivatives of the actuation model.\n\n"
           "It computes the partial derivatives of the full actuation. It assumes that you\n"
           "create the data using this class. The reason is that the derivatives are constant and\n"
           "defined in createData. The Hessian is constant, so we don't write again this value.\n"
           ":param data: full actuation data\n"
           ":param x: state vector\n"
           ":param u: control input\n")
      .def("createData", &ActuationModelFull::createData, bp::args("self"),
           "Create the full actuation data.\n\n"
           "Each actuation model (AM) has its own data that needs to be allocated.\n"
           "This function returns the allocated data for a predefined AM.\n"
           ":return AM data.");
}

}  // namespace python
}  // namespace crocoddyl
