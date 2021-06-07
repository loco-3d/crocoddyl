///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/cop-support.hpp"

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/utils/printable.hpp"

namespace crocoddyl {
namespace python {

void exposeCoPSupport() {
  bp::register_ptr_to_python<boost::shared_ptr<CoPSupport> >();

#pragma GCC diagnostic push  // TODO: Remove once the deprecated update call has been removed in a future release
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  bp::class_<CoPSupport>(
      "CoPSupport", "Model of the CoP support as lb <= Af <= ub",
      bp::init<Eigen::Matrix3d, Eigen::Vector2d>(bp::args("self", "R", "box"),
                                                 "Initialize the CoP support.\n\n"
                                                 ":param R: rotation matrix that defines the cone orientation\n"
                                                 ":param box: dimension of the foot surface dim = (length, width)\n"))
      .def(bp::init<>(bp::args("self"), "Default initialization of the CoP support."))
      .def("update", &CoPSupport::update, bp::args("self"),
           "Update the linear inequality (matrix and bounds).\n\n"
           "Run this function if you have changed one of the parameters.")
      .add_property("A", bp::make_function(&CoPSupport::get_A, bp::return_internal_reference<>()), "inequality matrix")
      .add_property("ub", bp::make_function(&CoPSupport::get_ub, bp::return_internal_reference<>()),
                    "inequality upper bound")
      .add_property("lb", bp::make_function(&CoPSupport::get_lb, bp::return_internal_reference<>()),
                    "inequality lower bound")
      .add_property("R", bp::make_function(&CoPSupport::get_R, bp::return_internal_reference<>()),
                    bp::make_function(&CoPSupport::set_R), "rotation matrix")
      .add_property("box", bp::make_function(&CoPSupport::get_box, bp::return_internal_reference<>()),
                    bp::make_function(&CoPSupport::set_box), "box size used to define the sole")
      .def(PrintableVisitor<CoPSupport>());

#pragma GCC diagnostic pop
}

}  // namespace python
}  // namespace crocoddyl
