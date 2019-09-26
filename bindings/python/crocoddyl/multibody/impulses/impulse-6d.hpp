///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_IMPULSES_IMPULSE_6D_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_IMPULSES_IMPULSE_6D_HPP_

#include "crocoddyl/multibody/impulses/impulse-6d.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeImpulse6D() {
  bp::class_<ImpulseModel6D, bp::bases<ImpulseModelAbstract> >(
      "ImpulseModel6D",
      "Rigid 6D impulse model.\n\n"
      "It defines a rigid 6D impulse models based on acceleration-based holonomic constraints.\n"
      "The calc and calcDiff functions compute the impulse Jacobian and drift (holonomic constraint) or\n"
      "the derivatives of the holonomic constraint, respectively.",
      bp::init<StateMultibody&, int>(bp::args(" self", " state", " frame"),
                                     "Initialize the impulse model.\n\n"
                                     ":param state: state of the multibody system\n"
                                     ":param frame: reference frame id")[bp::with_custodian_and_ward<1, 2>()])
      .def("calc", &ImpulseModel6D::calc_wrap, bp::args(" self", " data", " x"),
           "Compute the 6D impulse Jacobian and drift.\n\n"
           "The rigid impulse model throught acceleration-base holonomic constraint\n"
           "of the impulse frame placement.\n"
           ":param data: impulse data\n"
           ":param x: state vector")
      .def("calcDiff", &ImpulseModel6D::calcDiff_wrap,
           ImpulseModel_calcDiff_wraps(bp::args(" self", " data", " x", " recalc=True"),
                                       "Compute the derivatives of the 6D impulse holonomic constraint.\n\n"
                                       "The rigid impulse model throught acceleration-base holonomic constraint\n"
                                       "of the impulse frame placement.\n"
                                       ":param data: cost data\n"
                                       ":param x: state vector\n"
                                       ":param recalc: If true, it updates the impulse Jacobian and drift."))
      .def("updateForce", &ImpulseModel6D::updateForce, bp::args(" self", " data", " force"),
           "Convert the force into a stack of spatial forces.\n\n"
           ":param data: cost data\n"
           ":param lambda: force vector (dimension 6)")
      .def("createData", &ImpulseModel6D::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args(" self", " data"),
           "Create the 6D impulse data.\n\n"
           "Each impulse model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: Pinocchio data\n"
           ":return impulse data.")
      .add_property("frame",
                    bp::make_function(&ImpulseModel6D::get_frame, bp::return_value_policy<bp::return_by_value>()),
                    "reference frame id");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_IMPULSES_IMPULSE_6D_HPP_
