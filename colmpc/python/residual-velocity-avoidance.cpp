// BSD 3-Clause License
//
// Copyright (C) 2024, LAAS-CNRS.
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.

#include "colmpc/residual-velocity-avoidance.hpp"

#include "colmpc/python.hpp"

namespace colmpc {
namespace python {

namespace bp = boost::python;

void exposeResidualVelocityAvoidance() {
  bp::register_ptr_to_python<std::shared_ptr<ResidualModelVelocityAvoidance>>();

  bp::class_<ResidualModelVelocityAvoidance, bp::bases<ResidualModelAbstract>>(
      "ResidualModelVelocityAvoidance",
      bp::init<std::shared_ptr<crocoddyl::StateMultibody>,
               std::shared_ptr<pinocchio::GeometryModel>,
               const pinocchio::PairIndex,
               bp::optional<const double, const double, const double>>(
          bp::args("self", "state", "geom_model", "pair_id", "di", "ds", "ksi"),
          "Initialize the residual model.\n\n"
          ":param state: State of the multibody system\n"
          ":param geom_model: Pinocchio geometry model containing the "
          "collision pair\n"
          ":param pair_id: Index of the collision pair in the geometry "
          "model\n"
          ":param di: Distance at which the robot starts to slow down, "
          "defaults to 1.0e-2.\n"
          ":param ds: Security distance, defaults to 1.0e-5\n"
          ":param ksi: Convergence speed coefficient, defaults to 1.0e-2\n"))
      .def("calc", &ResidualModelVelocityAvoidance::calc,
           bp::args("self", "data", "x", "u"),
           "Compute the residual.\n\n"
           ":param data: residual data\n"
           ":param x: time-discrete state vector\n"
           ":param u: time-discrete control input")
      .def("calcDiff", &ResidualModelVelocityAvoidance::calcDiff,
           bp::args("self", "data", "x", "u"),
           "Compute the Jacobians of the residual.\n\n"
           "It assumes that calc has been run first.\n"
           ":param data: action data\n"
           ":param x: time-discrete state vector\n")
      .def("createData", &ResidualModelVelocityAvoidance::createData,
           bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the residual data.\n\n"
           "Each residual model has its own data that needs to be allocated. "
           "This function\n"
           "returns the allocated data for a predefined residual.\n"
           ":param data: shared data\n"
           ":return residual data.")
      .add_property(
          "geometry",
          bp::make_function(&ResidualModelVelocityAvoidance::get_geometry,
                            bp::return_value_policy<bp::return_by_value>()),
          "Pinocchio Geometry model")
      .add_property(
          "collision_pair_id",
          bp::make_function(&ResidualModelVelocityAvoidance::get_pair_id,
                            bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&ResidualModelVelocityAvoidance::set_pair_id),
          "reference collision pair id")
      .add_property(
          "di",
          bp::make_function(&ResidualModelVelocityAvoidance::get_di,
                            bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&ResidualModelVelocityAvoidance::set_di),
          "reference distance at which the robot starts to slow down")
      .add_property(
          "ds",
          bp::make_function(&ResidualModelVelocityAvoidance::get_ds,
                            bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&ResidualModelVelocityAvoidance::set_ds),
          "reference security distance")
      .add_property(
          "ksi",
          bp::make_function(&ResidualModelVelocityAvoidance::get_ksi,
                            bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&ResidualModelVelocityAvoidance::set_ksi),
          "reference convergence speed coefficient");

  bp::register_ptr_to_python<std::shared_ptr<ResidualDataVelocityAvoidance>>();

  bp::class_<ResidualDataVelocityAvoidance, bp::bases<ResidualDataAbstract>>(
      "ResidualDataVelocityAvoidance", "Data for vel collision residual.\n\n",
      bp::init<ResidualModelVelocityAvoidance*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create vel collision residual data.\n\n"
          ":param model: pair collision residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<
          1, 2, bp::with_custodian_and_ward<1, 3>>()])
      .add_property("geometry",
                    bp::make_getter(&ResidualDataVelocityAvoidance::geometry,
                                    bp::return_internal_reference<>()),
                    "pinocchio geometry data")
      .add_property("pinocchio",
                    bp::make_getter(&ResidualDataVelocityAvoidance::pinocchio,
                                    bp::return_internal_reference<>()),
                    "pinocchio data")
      .add_property("oMg_id_1",
                    bp::make_getter(&ResidualDataVelocityAvoidance::oMg_id_1,
                                    bp::return_internal_reference<>()),
                    "oMg_id_1")
      .add_property("oMg_id_2",
                    bp::make_getter(&ResidualDataVelocityAvoidance::oMg_id_2,
                                    bp::return_internal_reference<>()),
                    "oMg_id_2")
      .add_property("req",
                    bp::make_getter(&ResidualDataVelocityAvoidance::req,
                                    bp::return_internal_reference<>()),
                    "req")
      .add_property("res",
                    bp::make_getter(&ResidualDataVelocityAvoidance::res,
                                    bp::return_internal_reference<>()),
                    "res");
}

}  // namespace python
}  // namespace colmpc
