///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_IMPULSES_IMPULSE_3D_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_IMPULSES_IMPULSE_3D_HPP_

#include "crocoddyl/multibody/impulses/impulse-3d.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeImpulse3D() {
  bp::class_<ImpulseModel3D, bp::bases<ImpulseModelAbstract> >(
      "ImpulseModel3D",
      "Rigid 3D impulse model.\n\n"
      "It defines a rigid 3D impulse models (point impulse) based on acceleration-based holonomic constraints.\n"
      "The calc and calcDiff functions compute the impulse Jacobian and drift (holonomic constraint) or\n"
      "the derivatives of the holonomic constraint, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, int>(
          bp::args(" self", " state", " frame"),
          "Initialize the 3D impulse model.\n\n"
          ":param state: state of the multibody system\n"
          ":param frame: reference frame id"))
      .def("calc", &ImpulseModel3D::calc_wrap, bp::args(" self", " data", " x"),
           "Compute the 3D impulse Jacobian and drift.\n\n"
           "The rigid impulse model throught acceleration-base holonomic constraint\n"
           "of the impulse frame placement.\n"
           ":param data: impulse data\n"
           ":param x: state vector")
      .def("calcDiff", &ImpulseModel3D::calcDiff_wrap,
           ImpulseModel_calcDiff_wraps(bp::args(" self", " data", " x", " recalc=True"),
                                       "Compute the derivatives of the 3D impulse holonomic constraint.\n\n"
                                       "The rigid impulse model throught acceleration-base holonomic constraint\n"
                                       "of the impulse frame placement.\n"
                                       ":param data: cost data\n"
                                       ":param x: state vector\n"
                                       ":param recalc: If true, it updates the impulse Jacobian and drift."))
      .def("updateForce", &ImpulseModel3D::updateForce, bp::args(" self", " data", " force"),
           "Convert the force into a stack of spatial forces.\n\n"
           ":param data: cost data\n"
           ":param force: force vector (dimension 3)")
      .def("createData", &ImpulseModel3D::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args(" self", " data"),
           "Create the 3D impulse data.\n\n"
           "Each impulse model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: Pinocchio data\n"
           ":return impulse data.")
      .add_property("frame",
                    bp::make_function(&ImpulseModel3D::get_frame, bp::return_value_policy<bp::return_by_value>()),
                    "reference frame id");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_IMPULSES_IMPULSE_3D_HPP_
