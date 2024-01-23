///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024-2024, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/actuations/floating-base-propellers.hpp"

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/utils/copyable.hpp"
#include "python/crocoddyl/utils/printable.hpp"
#include "python/crocoddyl/utils/vector-converter.hpp"

namespace crocoddyl {

namespace python {

void exposeActuationFloatingBasePropeller() {
  bp::enum_<PropellerType>("PropellerType")
      .value("CW", CW)
      .value("CCW", CCW)
      .export_values();

  bp::class_<Propeller>(
      "Propeller", "Model for propellers",
      bp::init<pinocchio::SE3, double, double, bp::optional<PropellerType>>(
          bp::args("self", "M", "cthrust", "ctau", "type"),
          "Initialize the propeller in a give pose from the root joint.\n\n"
          ":param M: pose from root joint\n"
          ":param cthrust: coefficient of thrust (it relates propeller's "
          "(square) velocity to its thrust)\n"
          ":param ctau: coefficient of torque (it relates propeller's (square) "
          "velocity to its torque)\n"
          ":param type: type of propeller (clockwise or counterclockwise, "
          "default clockwise)"))
      .def(bp::init<double, double, bp::optional<PropellerType>>(
          bp::args("self", "cthrust", "ctau", "type"),
          "Initialize the propeller in a pose in the origin of the root "
          "joint.\n\n"
          ":param cthrust: coefficient of thrust (it relates propeller's "
          "(square) velocity to its thrust)\n"
          ":param ctau: coefficient of torque (it relates propeller's (square) "
          "velocity to its torque)\n"
          ":param type: type of propeller (clockwise or counterclockwise, "
          "default clockwise)"))
      .def_readwrite("pose", &Propeller::pose,
                     "propeller pose (traslation, rotation)")
      .def_readwrite("cthrust", &Propeller::cthrust, "coefficient of thrust")
      .def_readwrite("ctorque", &Propeller::ctorque, "coefficient of torque")
      .def_readwrite("type", &Propeller::type,
                     "type of propeller (clockwise or counterclockwise)")
      .def(PrintableVisitor<Propeller>())
      .def(CopyableVisitor<Propeller>());

  StdVectorPythonVisitor<std::vector<Propeller>, true>::expose(
      "StdVec_Propeller");

  bp::register_ptr_to_python<
      boost::shared_ptr<crocoddyl::ActuationModelFloatingBasePropellers>>();

  bp::class_<ActuationModelFloatingBasePropellers,
             bp::bases<ActuationModelAbstract>>(
      "ActuationModelFloatingBasePropellers",
      "Actuation models for floating base systems actuated with propellers "
      "(e.g. aerial "
      "manipulators).",
      bp::init<boost::shared_ptr<StateMultibody>, std::vector<Propeller>>(
          bp::args("self", "state", "propellers"),
          "Initialize the floating base actuation model equipped with "
          "propellers.\n\n"
          ":param state: state of multibody system\n"
          ":param propellers: vector of propellers"))
      .def<void (ActuationModelFloatingBasePropellers::*)(
          const boost::shared_ptr<ActuationDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ActuationModelFloatingBasePropellers::calc,
          bp::args("self", "data", "x", "u"),
          "Compute the actuation signal and actuation set from the thrust\n"
          "forces and joint torque inputs u.\n\n"
          ":param data: floating base propellers actuation data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: joint torque input (dim. nu)")
      .def(
          "calcDiff", &ActuationModelFloatingBasePropellers::calcDiff,
          bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the actuation model.\n\n"
          "It computes the partial derivatives of the propeller actuation. It\n"
          "assumes that calc has been run first. The reason is that the\n"
          "derivatives are constant and defined in createData. The Hessian\n"
          "is constant, so we don't write again this value.\n"
          ":param data: floating base propellers actuation data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: joint torque input (dim. nu)")
      .def("commands", &ActuationModelFloatingBasePropellers::commands,
           bp::args("self", "data", "x", "tau"),
           "Compute the thrust and joint torque commands from the generalized "
           "torques.\n\n"
           "It stores the results in data.u.\n"
           ":param data: actuation data\n"
           ":param x: state point (dim. state.nx)\n"
           ":param tau: generalized torques (dim state.nv)")
      .def("torqueTransform",
           &ActuationModelFloatingBasePropellers::torqueTransform,
           bp::args("self", "data", "x", "tau"),
           "Compute the torque transform from generalized torques to thrust "
           "and joint torque inputs.\n\n"
           "It stores the results in data.Mtau.\n"
           ":param data: floating base propellers actuation data\n"
           ":param x: state point (dim. state.nx)\n"
           ":param tau: generalized torques (dim state.nv)")
      .def("createData", &ActuationModelFloatingBasePropellers::createData,
           bp::args("self"),
           "Create the floating base propellers actuation data.")
      .add_property("propellers",
                    bp::make_function(
                        &ActuationModelFloatingBasePropellers::get_propellers,
                        bp::return_value_policy<bp::return_by_value>()),
                    bp::make_function(
                        &ActuationModelFloatingBasePropellers::set_propellers),
                    "vector of propellers")
      .add_property("npropellers",
                    bp::make_function(
                        &ActuationModelFloatingBasePropellers::get_npropellers),
                    "number of propellers")
      .add_property(
          "Wthrust",
          bp::make_function(&ActuationModelFloatingBasePropellers::get_Wthrust,
                            bp::return_value_policy<bp::return_by_value>()),
          "matrix mapping from thrusts to propeller wrenches")
      .add_property(
          "S",
          bp::make_function(&ActuationModelFloatingBasePropellers::get_S,
                            bp::return_value_policy<bp::return_by_value>()),
          "selection matrix for under-actuation part")
      .def(PrintableVisitor<ActuationModelFloatingBasePropellers>())
      .def(CopyableVisitor<ActuationModelFloatingBasePropellers>());
}

}  // namespace python
}  // namespace crocoddyl
