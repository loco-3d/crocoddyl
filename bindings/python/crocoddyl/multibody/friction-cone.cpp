///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/friction-cone.hpp"

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/utils/printable.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"

namespace crocoddyl {
namespace python {

void exposeFrictionCone() {
  bp::register_ptr_to_python<boost::shared_ptr<FrictionCone> >();

#pragma GCC diagnostic push  // TODO: Remove once the deprecated update call has been removed in a future release
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  bp::class_<FrictionCone>(
      "FrictionCone", "Model of the friction cone as lb <= Af <= ub",
      bp::init<Eigen::Matrix3d, double, bp::optional<std::size_t, bool, double, double> >(
          bp::args("self", "R", "mu", "nf", "inner_appr", "min_nforce", "max_nforce"),
          "Initialize the linearize friction cone.\n\n"
          ":param R: rotation matrix that defines the cone orientation w.r.t. the inertial frame\n"
          ":param mu: friction coefficient\n"
          ":param nf: number of facets\n"
          ":param inner_appr: inner or outer approximation (default True)\n"
          ":param min_nforce: minimum normal force (default 0.)\n"
          ":param max_nforce: maximum normal force (default sys.float_info.max)"))
      .def(bp::init<>(bp::args("self"), "Default initialization of the friction cone."))
      .def<void (FrictionCone::*)()>("update", &FrictionCone::update, bp::args("self"),
                                     "Update the linear inequality (matrix and bounds).\n\n"
                                     "Run this function if you have changed one of the parameters.")
      .def<void (FrictionCone::*)(const Eigen::Vector3d&, const double, const bool, const double, const double)>(
          "update", &FrictionCone::update, deprecated<>("Deprecated. Use update()."),
          bp::args("self", "nsurf", "mu", "inner_appr", "min_nforce", "max_nforce"),
          "Update the linear inequality (matrix and bounds).\n\n"
          ":param nsurf: surface normal vector (it defines the cone orientation)\n"
          ":param mu: friction coefficient\n"
          ":param inner_appr: inner or outer approximation (default True)\n"
          ":param min_nforce: minimum normal force (default 0.)\n"
          ":param max_nforce: maximum normal force (default sys.float_info.max)")
      .add_property("A", bp::make_function(&FrictionCone::get_A, bp::return_internal_reference<>()),
                    "inequality matrix")
      .add_property("ub", bp::make_function(&FrictionCone::get_ub, bp::return_internal_reference<>()),
                    "inequality upper bound")
      .add_property("lb", bp::make_function(&FrictionCone::get_lb, bp::return_internal_reference<>()),
                    "inequality lower bound")
      .add_property("nf", bp::make_function(&FrictionCone::get_nf, bp::return_value_policy<bp::return_by_value>()),
                    "number of facets (run update() if you have changed the value)")
      .add_property("R", bp::make_function(&FrictionCone::get_R, bp::return_internal_reference<>()),
                    bp::make_function(&FrictionCone::set_R),
                    "rotation matrix that defines the cone orientation w.r.t. the inertial frame (run update() if you "
                    "have changed the value)")
      .add_property("mu", bp::make_function(&FrictionCone::get_mu), bp::make_function(&FrictionCone::set_mu),
                    "friction coefficient (run update() if you have changed the value)")
      .add_property("inner_appr", bp::make_function(&FrictionCone::get_inner_appr),
                    bp::make_function(&FrictionCone::set_inner_appr),
                    "type of cone approximation (run update() if you have changed the value)")
      .add_property("min_nforce", bp::make_function(&FrictionCone::get_min_nforce),
                    bp::make_function(&FrictionCone::set_min_nforce),
                    "minimum normal force (run update() if you have changed the value)")
      .add_property("max_nforce", bp::make_function(&FrictionCone::get_max_nforce),
                    bp::make_function(&FrictionCone::set_max_nforce),
                    "maximum normal force (run update() if you have changed the value)")
      .def(PrintableVisitor<FrictionCone>());

#pragma GCC diagnostic pop
}

}  // namespace python
}  // namespace crocoddyl
