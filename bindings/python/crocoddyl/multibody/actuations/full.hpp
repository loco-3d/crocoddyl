///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_ACTUATIONS_FULL_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_ACTUATIONS_FULL_HPP_

#include "crocoddyl/multibody/actuations/full.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeActuationFull() {
  bp::class_<ActuationModelFull, bp::bases<ActuationModelAbstract> >(
      "ActuationModelFull", "Full actuation models.",
      bp::init<StateMultibody&>(bp::args(" self", " state"),
                                "Initialize the full actuation model.\n\n"
                                ":param state: state of multibody system")[bp::with_custodian_and_ward<1, 2>()])
      .def("calc", &ActuationModelFull::calc_wrap, bp::args(" self", " data", " x", " u"),
           "Compute the actuation signal from the control input u.\n\n"
           ":param data: full actuation data\n"
           ":param x: state vector\n"
           ":param u: control input")
      .def("calcDiff", &ActuationModelFull::calcDiff_wrap,
           ActuationModel_calcDiff_wraps(
               bp::args(" self", " data", " x", " u", " recalc=True"),
               "Compute the derivatives of the actuation model.\n\n"
               "It computes the partial derivatives of the full actuation. It assumes that you\n"
               "create the data using this class. The reason is that the derivatives are constant and\n"
               "defined in createData. The Hessian is constant, so we don't write again this value.\n"
               ":param data: full actuation data\n"
               ":param x: state vector\n"
               ":param u: control input\n"
               ":param recalc: If true, it updates the actuation signal."))
      .def("createData", &ActuationModelFull::createData, bp::args(" self"),
           "Create the full actuation data.\n\n"
           "Each actuation model (AM) has its own data that needs to be allocated.\n"
           "This function returns the allocated data for a predefined AM.\n"
           ":return AM data.");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_ACTUATIONS_FULL_HPP_
