// BSD 3-Clause License
//
// Copyright (C) 2024, LAAS-CNRS.
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.

#include "colmpc/residual-distance-collision-2.hpp"

#include "colmpc/python.hpp"

namespace colmpc {
namespace python {

namespace bp = boost::python;

void exposeResidualDistanceCollision2() {
  bp::register_ptr_to_python<std::shared_ptr<ResidualDistanceCollision2>>();

  bp::class_<ResidualDistanceCollision2, bp::bases<ResidualModelAbstract>>(
      "ResidualDistanceCollision2",
      bp::init<std::shared_ptr<StateMultibody>, const std::size_t,
               const pinocchio::PairIndex>(
          bp::args("self", "state", "nu", "pair_id"),
          "Initialize the residual model.\n\n"
          ":param state: State of the multibody system\n"
          ":param nu: Dimension of the control vector\n"
          ":param pair_id: Index of the collision pair in the geometry "
          "model\n"))
      //   .def("calc", &ResidualDistanceCollision2::calc,
      //        bp::args("self", "data", "x", "u"),
      //        "Compute the residual.\n\n"
      //        ":param data: residual data\n"
      //        ":param x: time-discrete state vector\n"
      //        ":param u: time-discrete control input")
      //   .def("calcDiff", &ResidualDistanceCollision2::calcDiff,
      //        bp::args("self", "data", "x", "u"),
      //        "Compute the Jacobians of the residual.\n\n"
      //        "It assumes that calc has been run first.\n"
      //        ":param data: action data\n"
      //        ":param x: time-discrete state vector\n"
      //        ":param u: time-discrete control input\n")
      .def("createData", &ResidualDistanceCollision2::createData,
           bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the residual data.\n\n"
           "Each residual model has its own data that needs to be allocated. "
           "This function\n"
           "returns the allocated data for a predefined residual.\n"
           ":param data: shared data\n"
           ":return residual data.")
      .add_property(
          "collision_pair_id",
          bp::make_function(&ResidualDistanceCollision2::get_pair_id,
                            bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&ResidualDistanceCollision2::set_pair_id),
          "reference collision pair id");

  bp::register_ptr_to_python<std::shared_ptr<ResidualDataDistanceCollision2>>();

  bp::class_<ResidualDataDistanceCollision2, bp::bases<ResidualDataAbstract>>(
      "ResidualDataDistanceCollision2", "Data for vel collision residual.\n\n",
      bp::init<ResidualDistanceCollision2*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create vel collision residual data.\n\n"
          ":param model: pair collision residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<
          1, 2, bp::with_custodian_and_ward<1, 3>>()])
      .add_property("pinocchio",
                    bp::make_getter(&ResidualDataDistanceCollision2::pinocchio,
                                    bp::return_internal_reference<>()),
                    "pinocchio data")
      .add_property("J1",
                    bp::make_getter(&ResidualDataDistanceCollision2::J1,
                                    bp::return_internal_reference<>()),
                    "J1")
      .add_property("J2",
                    bp::make_getter(&ResidualDataDistanceCollision2::J2,
                                    bp::return_internal_reference<>()),
                    "J2")
      .add_property("cp1",
                    bp::make_getter(&ResidualDataDistanceCollision2::cp1,
                                    bp::return_internal_reference<>()),
                    "cp1")
      .add_property("cp2",
                    bp::make_getter(&ResidualDataDistanceCollision2::cp2,
                                    bp::return_internal_reference<>()),
                    "cp2")
      .add_property("f1p1",
                    bp::make_getter(&ResidualDataDistanceCollision2::f1p1,
                                    bp::return_internal_reference<>()),
                    "f1p1")
      .add_property("f2p2",
                    bp::make_getter(&ResidualDataDistanceCollision2::f2p2,
                                    bp::return_internal_reference<>()),
                    "f2p2")
      .add_property("f1Mp1",
                    bp::make_getter(&ResidualDataDistanceCollision2::f1Mp1,
                                    bp::return_internal_reference<>()),
                    "f1Mp1")
      .add_property("f2Mp2",
                    bp::make_getter(&ResidualDataDistanceCollision2::f2Mp2,
                                    bp::return_internal_reference<>()),
                    "f2Mp2")
      .add_property("oMg_id_1",
                    bp::make_getter(&ResidualDataDistanceCollision2::oMg_id_1,
                                    bp::return_internal_reference<>()),
                    "oMg_id_1")
      .add_property("oMg_id_2",
                    bp::make_getter(&ResidualDataDistanceCollision2::oMg_id_2,
                                    bp::return_internal_reference<>()),
                    "oMg_id_2");
}

}  // namespace python
}  // namespace colmpc
