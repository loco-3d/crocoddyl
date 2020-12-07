///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/friction-cone.hpp"

namespace crocoddyl {
namespace python {

void exposeFrictionCone() {
  bp::class_<FrictionCone>("FrictionCone", "Model of the friction cone as lb <= Af <= ub",
                           bp::init<Eigen::Vector3d, double, bp::optional<std::size_t, bool, double, double> >(
                               bp::args("self", "normal", "mu", "nf", "inner_appr", "min_nforce", "max_nforce"),
                               "Initialize the linearize friction cone.\n\n"
                               ":param normal: normal vector that defines the cone orientation\n"
                               ":param mu: friction coefficient\n"
                               ":param nf: number of facets\n"
                               ":param inner_appr: inner or outer approximation (default True)\n"
                               ":param min_nforce: minimum normal force (default 0.)\n"
                               ":param max_nforce: maximum normal force (default sys.float_info.max)\n"))
      .def(bp::init<>(bp::args("self"), "Default initialization of the friction cone."))
      .def("update", &FrictionCone::update, bp::args("self", "normal", "mu", "inner_appr", "min_nforce", "max_nforce"),
           "Update the linear inequality (matrix and bounds).\n\n"
           ":param normal: normal vector that defines the cone orientation\n"
           ":param mu: friction coefficient\n"
           ":param inner_appr: inner or outer approximation (default True)\n"
           ":param min_nforce: minimum normal force (default 0.)\n"
           ":param max_nforce: maximum normal force (default sys.float_info.max)")
      .add_property("A", bp::make_function(&FrictionCone::get_A, bp::return_internal_reference<>()),
                    "inequality matrix")
      .add_property("lb", bp::make_function(&FrictionCone::get_lb, bp::return_internal_reference<>()),
                    "inequality lower bound")
      .add_property("ub", bp::make_function(&FrictionCone::get_ub, bp::return_internal_reference<>()),
                    "inequality upper bound")
      .add_property("nsurf", bp::make_function(&FrictionCone::get_nsurf, bp::return_internal_reference<>()),
                    bp::make_function(&FrictionCone::set_nsurf), "normal vector")
      .add_property("mu", bp::make_function(&FrictionCone::get_mu),
                    bp::make_function(&FrictionCone::set_mu), "friction coefficient")
      .add_property("nf", bp::make_function(&FrictionCone::get_nf),
                    "number of facets")
      .add_property("inner_appr",
                    bp::make_function(&FrictionCone::get_inner_appr),
                    bp::make_function(&FrictionCone::set_inner_appr), "type of cone approxition")
      .add_property("min_nforce",
                    bp::make_function(&FrictionCone::get_min_nforce),
                    bp::make_function(&FrictionCone::set_min_nforce), "minimum normal force")
      .add_property("max_nforce",
                    bp::make_function(&FrictionCone::get_max_nforce),
                    bp::make_function(&FrictionCone::set_max_nforce), "maximum normal force");
}

}  // namespace python
}  // namespace crocoddyl
