///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_ACTUATIONS_FLOATING_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_ACTUATIONS_FLOATING_BASE_HPP_

#include "crocoddyl/multibody/actuations/floating-base.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeActuationFloatingBase() {
  bp::class_<ActuationModelFloatingBase, bp::bases<ActuationModelAbstract> >(
      "ActuationModelFloatingBase",
      "Floating-base actuation models.\n\n"
      "It simplies consider a floating-base actuation model, where the first 6 elements are unactuated.",
      bp::init<StateMultibody&>(bp::args(" self", " state"),
                                "Initialize the floating-base actuation model.\n\n"
                                ":param state: state of multibody system")[bp::with_custodian_and_ward<1, 2>()])
      .def("calc", &ActuationModelFloatingBase::calc_wrap, bp::args(" self", " data", " x", " u"),
           "Compute the actuation signal from the control input u.\n\n"
           "It describes the time-continuos evolution of the floating-base actuation model.\n"
           ":param data: floating-base actuation data\n"
           ":param x: state vector\n"
           ":param u: control input")
      .def("calcDiff", &ActuationModelFloatingBase::calcDiff_wrap,
           ActuationModel_calcDiff_wraps(
               bp::args(" self", " data", " x", " u", " recalc=True"),
               "Compute the derivatives of the actuation model.\n\n"
               "It computes the partial derivatives of the floating-base actuation. It assumes that you\n"
               "create the data using this class. The reason is that the derivatives are constant and\n"
               "defined in createData."
               ":param data: floating-base actuation data\n"
               ":param x: state vector\n"
               ":param u: control input\n"
               ":param recalc: If true, it updates the actuation signal."))
      .def("createData", &ActuationModelFloatingBase::createData, bp::args(" self"),
           "Create the floating-base actuation data.\n\n"
           "Each actuation model (AM) has its own data that needs to be allocated.\n"
           "This function returns the allocated data for a predefined AM.\n"
           ":return AM data.");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_ACTUATIONS_FLOATING_BASE_HPP_
