///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024-2024, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/actuations/floating-base-thrusters.hpp"
#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/utils/copyable.hpp"
#include "python/crocoddyl/utils/printable.hpp"
#include "python/crocoddyl/utils/vector-converter.hpp"

namespace crocoddyl {

namespace python {

void exposeActuationFloatingBaseThruster() {
  bp::enum_<ThrusterType>("ThrusterType")
      .value("CW", CW)
      .value("CCW", CCW)
      .export_values();

  bp::class_<Thruster>(
      "Thruster", "Model for thrusters",
      bp::init<pinocchio::SE3, double,
               bp::optional<ThrusterType, double, double>>(
          bp::args("self", "M", "ctorque", "type", "min_thrust", "max_thrust"),
          "Initialize the thruster in a give pose from the root joint.\n\n"
          ":param M: pose from root joint\n"
          ":param ctorque: coefficient of generated torque per thrust\n"
          ":param type: type of thruster (clockwise or counterclockwise, "
          "default clockwise)\n"
          ":param min_thrust: minimum thrust (default 0.)\n"
          ":param max_thrust: maximum thrust (default np.inf)"))
      .def(bp::init<double, bp::optional<ThrusterType, double, double>>(
          bp::args("self", "ctorque", "type", "min_thrust", "max_thrust"),
          "Initialize the thruster in a give pose from the root joint.\n\n"
          ":param ctorque: coefficient of generated torque per thrust\n"
          ":param type: type of thruster (clockwise or counterclockwise, "
          "default clockwise)\n"
          ":param min_thrust: minimum thrust (default 0.)\n"
          ":param max_thrust: maximum thrust (default np.inf)"))
      .def_readwrite("pose", &Thruster::pose,
                     "thruster pose (traslation, rotation)")
      .def_readwrite("ctorque", &Thruster::ctorque,
                     "coefficient of generated torque per thrust")
      .def_readwrite("type", &Thruster::type,
                     "type of thruster (clockwise or counterclockwise)")
      .def_readwrite("min_thrust", &Thruster::min_thrust, "minimum thrust")
      .def_readwrite("max_thrust", &Thruster::min_thrust, "maximum thrust")
      .def(PrintableVisitor<Thruster>())
      .def(CopyableVisitor<Thruster>());

  StdVectorPythonVisitor<std::vector<Thruster>, true>::expose(
      "StdVec_Thruster");

  bp::register_ptr_to_python<
      std::shared_ptr<crocoddyl::ActuationModelFloatingBaseThrusters>>();

  bp::class_<ActuationModelFloatingBaseThrusters,
             bp::bases<ActuationModelAbstract>>(
      "ActuationModelFloatingBaseThrusters",
      "Actuation models for floating base systems actuated with thrusters "
      "(e.g. aerial "
      "manipulators).",
      bp::init<std::shared_ptr<StateMultibody>, std::vector<Thruster>>(
          bp::args("self", "state", "thrusters"),
          "Initialize the floating base actuation model equipped with "
          "thrusters.\n\n"
          ":param state: state of multibody system\n"
          ":param thrusters: vector of thrusters"))
      .def<void (ActuationModelFloatingBaseThrusters::*)(
          const std::shared_ptr<ActuationDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ActuationModelFloatingBaseThrusters::calc,
          bp::args("self", "data", "x", "u"),
          "Compute the actuation signal and actuation set from the thrust\n"
          "forces and joint torque inputs u.\n\n"
          ":param data: floating base thrusters actuation data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: joint torque input (dim. nu)")
      .def("calcDiff", &ActuationModelFloatingBaseThrusters::calcDiff,
           bp::args("self", "data", "x", "u"),
           "Compute the derivatives of the actuation model.\n\n"
           "It computes the partial derivatives of the thruster actuation. It\n"
           "assumes that calc has been run first. The reason is that the\n"
           "derivatives are constant and defined in createData. The Hessian\n"
           "is constant, so we don't write again this value.\n"
           ":param data: floating base thrusters actuation data\n"
           ":param x: state point (dim. state.nx)\n"
           ":param u: joint torque input (dim. nu)")
      .def("commands", &ActuationModelFloatingBaseThrusters::commands,
           bp::args("self", "data", "x", "tau"),
           "Compute the thrust and joint torque commands from the generalized "
           "torques.\n\n"
           "It stores the results in data.u.\n"
           ":param data: actuation data\n"
           ":param x: state point (dim. state.nx)\n"
           ":param tau: generalized torques (dim state.nv)")
      .def("torqueTransform",
           &ActuationModelFloatingBaseThrusters::torqueTransform,
           bp::args("self", "data", "x", "tau"),
           "Compute the torque transform from generalized torques to thrust "
           "and joint torque inputs.\n\n"
           "It stores the results in data.Mtau.\n"
           ":param data: floating base thrusters actuation data\n"
           ":param x: state point (dim. state.nx)\n"
           ":param tau: generalized torques (dim state.nv)")
      .def("createData", &ActuationModelFloatingBaseThrusters::createData,
           bp::args("self"),
           "Create the floating base thrusters actuation data.")
      .add_property(
          "thrusters",
          bp::make_function(&ActuationModelFloatingBaseThrusters::get_thrusters,
                            bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(
              &ActuationModelFloatingBaseThrusters::set_thrusters),
          "vector of thrusters")
      .add_property("nthrusters",
                    bp::make_function(
                        &ActuationModelFloatingBaseThrusters::get_nthrusters),
                    "number of thrusters")
      .add_property(
          "Wthrust",
          bp::make_function(&ActuationModelFloatingBaseThrusters::get_Wthrust,
                            bp::return_value_policy<bp::return_by_value>()),
          "matrix mapping from thrusts to thruster wrenches")
      .add_property(
          "S",
          bp::make_function(&ActuationModelFloatingBaseThrusters::get_S,
                            bp::return_value_policy<bp::return_by_value>()),
          "selection matrix for under-actuation part")
      .def(PrintableVisitor<ActuationModelFloatingBaseThrusters>())
      .def(CopyableVisitor<ActuationModelFloatingBaseThrusters>());
}

}  // namespace python
}  // namespace crocoddyl
