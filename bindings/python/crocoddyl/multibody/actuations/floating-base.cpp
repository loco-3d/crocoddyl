///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, LAAS-CNRS, University of Edinburgh
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/actuations/floating-base.hpp"

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/utils/copyable.hpp"

namespace crocoddyl {
namespace python {

void exposeActuationFloatingBase() {
  bp::register_ptr_to_python<
      std::shared_ptr<crocoddyl::ActuationModelFloatingBase> >();

  bp::class_<ActuationModelFloatingBase, bp::bases<ActuationModelAbstract> >(
      "ActuationModelFloatingBase",
      "Floating-base actuation models.\n\n"
      "It considers the first joint, defined in the Pinocchio model, as the "
      "floating-base joints.\n"
      "Then, this joint (that might have various DoFs) is unactuated.",
      bp::init<std::shared_ptr<StateMultibody> >(
          bp::args("self", "state"),
          "Initialize the floating-base actuation model.\n\n"
          ":param state: state of multibody system"))
      .def("calc", &ActuationModelFloatingBase::calc,
           bp::args("self", "data", "x", "u"),
           "Compute the floating-base actuation signal and actuation set from "
           "the joint torque input u.\n\n"
           "It describes the time-continuos evolution of the floating-base "
           "actuation model.\n"
           ":param data: floating-base actuation data\n"
           ":param x: state point (dim. state.nx)\n"
           ":param u: joint-torque input (dim. nu)")
      .def("calcDiff", &ActuationModelFloatingBase::calcDiff,
           bp::args("self", "data", "x", "u"),
           "Compute the Jacobians of the floating-base actuation model.\n\n"
           "It computes the partial derivatives of the floating-base "
           "actuation. It assumes that calc\n"
           "has been run first. The reason is that the derivatives are "
           "constant and\n"
           "defined in createData. The derivatives are constant, so we don't "
           "write again these values.\n"
           ":param data: floating-base actuation data\n"
           ":param x: state point (dim. state.nx)\n"
           ":param u: joint-torque input (dim. nu)")
      .def("commands", &ActuationModelFloatingBase::commands,
           bp::args("self", "data", "x", "tau"),
           "Compute the joint-torque commands from the generalized torques.\n\n"
           "It stores the results in data.u.\n"
           ":param data: actuation data\n"
           ":param x: state point (dim. state.nx)\n"
           ":param tau: generalized torques (dim state.nv)")
      .def("torqueTransform", &ActuationModelFloatingBase::torqueTransform,
           bp::args("self", "data", "x", "tau"),
           "Compute the torque transform from generalized torques to "
           "joint-torque inputs.\n\n"
           "It stores the results in data.Mtau.\n"
           ":param data: actuation data\n"
           ":param x: state point (dim. state.nx)\n"
           ":param tau: generalized torques (dim state.nv)")
      .def("createData", &ActuationModelFloatingBase::createData,
           bp::args("self"),
           "Create the floating-base actuation data.\n\n"
           "Each actuation model (AM) has its own data that needs to be "
           "allocated.\n"
           "This function returns the allocated data for a predefined AM.\n"
           ":return AM data.")
      .def(CopyableVisitor<ActuationModelFloatingBase>());
}

}  // namespace python
}  // namespace crocoddyl
