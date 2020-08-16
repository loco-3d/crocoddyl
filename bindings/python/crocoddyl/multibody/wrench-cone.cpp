///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/wrench-cone.hpp"

namespace crocoddyl {
namespace python {

void exposeWrenchCone() {
  bp::class_<WrenchCone>("WrenchCone", "Model of the friction cone as lb <= Af <= ub",
                           bp::init<Eigen::Matrix3d, double, Eigen::Vector2d, bp::optional<std::size_t> >(
                               bp::args("self", "rot", "mu", "box", "nf"),
                               "Initialize the linearize friction cone.\n\n"
                               ":param rot: rotation matrix that defines the cone orientation\n"
                               ":param mu: friction coefficient\n"
                               ":param box: X, Y distance between the origin and edge of the link\n"
                               ":param nf: number of facets\n"))
      .def(bp::init<>(bp::args("self"), "Default initialization of the friction cone."))
      .def("update", &WrenchCone::update, bp::args("self", "rob", "mu", "box"),
           "Update the linear inequality (matrix and bounds).\n\n"
           ":param rot: rotation matrix that defines the cone orientation\n"
           ":param mu: friction coefficient\n"
           ":param box: X, Y distance between the origin and edge of the link\n")
      .add_property("A", bp::make_function(&WrenchCone::get_A, bp::return_internal_reference<>()),
                    "inequality matrix")
      .add_property("lb", bp::make_function(&WrenchCone::get_lb, bp::return_internal_reference<>()),
                    "inequality lower bound")
      .add_property("ub", bp::make_function(&WrenchCone::get_ub, bp::return_internal_reference<>()),
                    "inequality upper bound")
      .add_property("rot", bp::make_function(&WrenchCone::get_rot, bp::return_internal_reference<>()),
                    "rotation matrix")
      .add_property("box", bp::make_function(&WrenchCone::get_box, bp::return_internal_reference<>()),
                    "box size")
      .add_property("mu", bp::make_function(&WrenchCone::get_mu, bp::return_value_policy<bp::return_by_value>()),
                    "friction coefficient")
      .add_property("nf", bp::make_function(&WrenchCone::get_nf, bp::return_value_policy<bp::return_by_value>()),
                    "number of facets");

}

}  // namespace python
}  // namespace crocoddyl
