// BSD 3-Clause License
//
// Copyright (C) 2024, LAAS-CNRS.
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.

#include "colmpc/residual-distance-collision.hpp"

#include "colmpc/python.hpp"

namespace colmpc {
namespace python {

namespace bp = boost::python;

void exposeResidualDistanceCollision() {
  bp::register_ptr_to_python<std::shared_ptr<ResidualDistanceCollision>>();

  bp::class_<ResidualDistanceCollision, bp::bases<ResidualModelAbstract>>(
      "ResidualDistanceCollision",
      bp::init<std::shared_ptr<crocoddyl::StateMultibody>, const std::size_t,
               std::shared_ptr<pinocchio::GeometryModel>,
               const pinocchio::PairIndex>(
          bp::args("self", "state", "nu", "geom_model", "pair_id"),
          "Initialize the residual model.\n\n"
          ":param state: State of the multibody system\n"
          ":param nu: Dimension of the control vector\n"
          ":param geom_model: Pinocchio geometry model containing the "
          "collision pair\n"
          ":param pair_id: Index of the collision pair in the geometry "
          "model\n"))
      .def("calc", &ResidualDistanceCollision::calc,
           bp::args("self", "data", "x", "u"),
           "Compute the residual.\n\n"
           ":param data: residual data\n"
           ":param x: time-discrete state vector\n"
           ":param u: time-discrete control input")
      .def("calcDiff", &ResidualDistanceCollision::calcDiff,
           bp::args("self", "data", "x", "u"),
           "Compute the Jacobians of the residual.\n\n"
           "It assumes that calc has been run first.\n"
           ":param data: action data\n"
           ":param x: time-discrete state vector\n"
           ":param u: time-discrete control input\n")
      .def("createData", &ResidualDistanceCollision::createData,
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
          bp::make_function(&ResidualDistanceCollision::get_geometry,
                            bp::return_value_policy<bp::return_by_value>()),
          "Pinocchio Geometry model")
      .add_property(
          "collision_pair_id",
          bp::make_function(&ResidualDistanceCollision::get_pair_id,
                            bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&ResidualDistanceCollision::set_pair_id),
          "reference collision pair id");

  bp::register_ptr_to_python<std::shared_ptr<ResidualDataDistanceCollision>>();

  bp::class_<ResidualDataDistanceCollision, bp::bases<ResidualDataAbstract>>(
      "ResidualDataDistanceCollision", "Data for vel collision residual.\n\n",
      bp::init<ResidualDistanceCollision*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create vel collision residual data.\n\n"
          ":param model: pair collision residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<
          1, 2, bp::with_custodian_and_ward<1, 3>>()])
      .add_property("pinocchio",
                    bp::make_getter(&ResidualDataDistanceCollision::pinocchio,
                                    bp::return_internal_reference<>()),
                    "pinocchio data")
      .add_property("J1",
                    bp::make_getter(&ResidualDataDistanceCollision::J1,
                                    bp::return_internal_reference<>()),
                    "J1")
      .add_property("J2",
                    bp::make_getter(&ResidualDataDistanceCollision::J2,
                                    bp::return_internal_reference<>()),
                    "J2")
      .add_property("cp1",
                    bp::make_getter(&ResidualDataDistanceCollision::cp1,
                                    bp::return_internal_reference<>()),
                    "cp1")
      .add_property("cp2",
                    bp::make_getter(&ResidualDataDistanceCollision::cp2,
                                    bp::return_internal_reference<>()),
                    "cp2")
      .add_property("f1p1",
                    bp::make_getter(&ResidualDataDistanceCollision::f1p1,
                                    bp::return_internal_reference<>()),
                    "f1p1")
      .add_property("f2p2",
                    bp::make_getter(&ResidualDataDistanceCollision::f2p2,
                                    bp::return_internal_reference<>()),
                    "f2p2")
      .add_property("f1Mp1",
                    bp::make_getter(&ResidualDataDistanceCollision::f1Mp1,
                                    bp::return_internal_reference<>()),
                    "f1Mp1")
      .add_property("f2Mp2",
                    bp::make_getter(&ResidualDataDistanceCollision::f2Mp2,
                                    bp::return_internal_reference<>()),
                    "f2Mp2")
      .add_property("oMg_id_1",
                    bp::make_getter(&ResidualDataDistanceCollision::oMg_id_1,
                                    bp::return_internal_reference<>()),
                    "oMg_id_1")
      .add_property("oMg_id_2",
                    bp::make_getter(&ResidualDataDistanceCollision::oMg_id_2,
                                    bp::return_internal_reference<>()),
                    "oMg_id_2")
      .add_property("req",
                    bp::make_getter(&ResidualDataDistanceCollision::req,
                                    bp::return_internal_reference<>()),
                    "req")
      .add_property("res",
                    bp::make_getter(&ResidualDataDistanceCollision::res,
                                    bp::return_internal_reference<>()),
                    "res");
}

}  // namespace python
}  // namespace colmpc
