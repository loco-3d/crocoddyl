///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/wrench-cone.hpp"
#include "python/crocoddyl/utils/printable.hpp"

namespace crocoddyl {
namespace python {

void exposeWrenchCone() {
  bp::class_<WrenchCone>(
      "WrenchCone", "Model of the wrench cone as lb <= Af <= ub",
      bp::init<Eigen::Matrix3d, double, Eigen::Vector2d, bp::optional<std::size_t, double, double> >(
          bp::args("self", "R", "mu", "box", "nf", "min_nforce", "max_nforce"),
          "Initialize the linearize wrench cone.\n\n"
          ":param R: rotation matrix that defines the cone orientation\n"
          ":param mu: friction coefficient\n"
          ":param box: dimension of the foot surface dim = (length, width)\n"
          ":param nf: number of facets\n"
          ":param min_nforce: minimum normal force (default 0.)\n"
          ":param max_nforce: maximum normal force (default sys.float_info.max)\n"))
      .def(bp::init<>(bp::args("self"), "Default initialization of the wrench cone."))
      .def("update", &WrenchCone::update, bp::args("self", "R", "mu", "box", "min_nforce", "max_nforce"),
           "Update the linear inequality (matrix and bounds).\n\n"
           ":param R: rotation matrix that defines the cone orientation\n"
           ":param mu: friction coefficient\n"
           ":param box: dimension of the foot surface dim = (length, width)\n"
           ":param min_nforce: minimum normal force (default 0.)\n"
           ":param max_nforce: maximum normal force (default sys.float_info.max)")
      .add_property("A", bp::make_function(&WrenchCone::get_A, bp::return_internal_reference<>()), "inequality matrix")
      .add_property("lb", bp::make_function(&WrenchCone::get_lb, bp::return_internal_reference<>()),
                    "inequality lower bound")
      .add_property("ub", bp::make_function(&WrenchCone::get_ub, bp::return_internal_reference<>()),
                    "inequality upper bound")
      .add_property("R", bp::make_function(&WrenchCone::get_R, bp::return_internal_reference<>()),
                    bp::make_function(&WrenchCone::set_R), "rotation matrix")
      .add_property("box", bp::make_function(&WrenchCone::get_box, bp::return_internal_reference<>()),
                    bp::make_function(&WrenchCone::set_box), "box size used to define the sole")
      .add_property("mu", bp::make_function(&WrenchCone::get_mu, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_function(&WrenchCone::set_mu), "friction coefficient")
      .add_property("nf", bp::make_function(&WrenchCone::get_nf, bp::return_value_policy<bp::return_by_value>()),
                    "number of facets")
      .add_property("min_nforce",
                    bp::make_function(&WrenchCone::get_min_nforce, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_function(&WrenchCone::set_min_nforce), "minimum normal force")
      .add_property("max_nforce",
                    bp::make_function(&WrenchCone::get_max_nforce, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_function(&WrenchCone::set_max_nforce), "maximum normal force")
      .def(PrintableVisitor<WrenchCone>());
}

}  // namespace python
}  // namespace crocoddyl
