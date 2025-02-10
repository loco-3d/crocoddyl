///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, LAAS-CNRS, University of Edinburgh
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/actuations/full.hpp"

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/utils/copyable.hpp"

namespace crocoddyl {
namespace python {

void exposeActuationFull() {
  bp::register_ptr_to_python<std::shared_ptr<crocoddyl::ActuationModelFull> >();

  bp::class_<ActuationModelFull, bp::bases<ActuationModelAbstract> >(
      "ActuationModelFull", "Full actuation models.",
      bp::init<std::shared_ptr<StateAbstract> >(
          bp::args("self", "state"),
          "Initialize the full actuation model.\n\n"
          ":param state: state of dynamical system"))
      .def("calc", &ActuationModelFull::calc,
           bp::args("self", "data", "x", "u"),
           "Compute the actuation signal and actuation set from the joint "
           "torque input u.\n\n"
           ":param data: full actuation data\n"
           ":param x: state point (dim. state.nx)\n"
           ":param u: joint torque input (dim. nu)")
      .def("calcDiff", &ActuationModelFull::calcDiff,
           bp::args("self", "data", "x", "u"),
           "Compute the derivatives of the actuation model.\n\n"
           "It computes the partial derivatives of the full actuation. It "
           "assumes that calc\n"
           "has been run first. The reason is that the derivatives are "
           "constant and\n"
           "defined in createData. The Hessian is constant, so we don't write "
           "again this value.\n"
           ":param data: full actuation data\n"
           ":param x: state point (dim. state.nx)\n"
           ":param u: joint torque input (dim. nu)")
      .def("commands", &ActuationModelFull::commands,
           bp::args("self", "data", "x", "tau"),
           "Compute the joint torque commands from the generalized torques.\n\n"
           "It stores the results in data.u.\n"
           ":param data: actuation data\n"
           ":param x: state point (dim. state.nx)\n"
           ":param tau: generalized torques (dim state.nv)")
      .def("torqueTransform", &ActuationModelFull::torqueTransform,
           bp::args("self", "data", "x", "tau"),
           "Compute the torque transform from generalized torques to joint "
           "torque inputs.\n\n"
           "It stores the results in data.Mtau.\n"
           ":param data: actuation data\n"
           ":param x: state point (dim. state.nx)\n"
           ":param tau: generalized torques (dim state.nv)")
      .def("createData", &ActuationModelFull::createData, bp::args("self"),
           "Create the full actuation data.\n\n"
           "Each actuation model (AM) has its own data that needs to be "
           "allocated.\n"
           "This function returns the allocated data for a predefined AM.\n"
           ":return AM data.")
      .def(CopyableVisitor<ActuationModelFull>());
}

}  // namespace python
}  // namespace crocoddyl
