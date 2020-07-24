///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/frames.hpp"
#include "python/crocoddyl/utils/printable.hpp"

namespace crocoddyl {
namespace python {

void exposeFrames() {
  bp::class_<FrameTranslation>(
      "FrameTranslation",
      "Frame translation describe using Pinocchio.\n\n"
      "It defines a frame translation (3D vector) for a given frame ID",
      bp::init<FrameIndex, Eigen::Vector3d>(bp::args("self", "frame", "oxf"),
                                            "Initialize the frame translation.\n\n"
                                            ":param frame: frame ID\n"
                                            ":param oxf: Frame translation w.r.t. the origin"))
      .def(bp::init<>(bp::args("self"), "Default initialization of the frame translation."))
      .def_readwrite("frame", &FrameTranslation::frame, "frame ID")
      .add_property("oxf", bp::make_getter(&FrameTranslation::oxf, bp::return_internal_reference<>()),
                    bp::make_setter(&FrameTranslation::oxf), "frame translation")
      .def(PrintableVisitor<FrameTranslation>());

  bp::class_<FrameRotation>("FrameRotation",
                            "Frame rotation describe using Pinocchio.\n\n"
                            "It defines a frame rotation (rotation matrix) for a given frame ID",
                            bp::init<FrameIndex, Eigen::Matrix3d>(bp::args("self", "frame", "oRf"),
                                                                  "Initialize the frame rotation.\n\n"
                                                                  ":param frame: frame ID\n"
                                                                  ":param oRf: Frame rotation w.r.t. the origin"))
      .def(bp::init<>(bp::args("self"), "Default initialization of the frame rotation."))
      .def_readwrite("frame", &FrameRotation::frame, "frame ID")
      .add_property("oRf", bp::make_getter(&FrameRotation::oRf, bp::return_internal_reference<>()),
                    bp::make_setter(&FrameRotation::oRf), "frame rotation")
      .def(PrintableVisitor<FrameRotation>());

  bp::class_<FramePlacement>("FramePlacement",
                             "Frame placement describe using Pinocchio.\n\n"
                             "It defines a frame placement (SE(3) point) for a given frame ID",
                             bp::init<FrameIndex, pinocchio::SE3>(bp::args("self", "frame", "oMf"),
                                                                  "Initialize the frame placement.\n\n"
                                                                  ":param frame: frame ID\n"
                                                                  ":param oMf: Frame placement w.r.t. the origin"))
      .def(bp::init<>(bp::args("self"), "Default initialization of the frame placement."))
      .def_readwrite("frame", &FramePlacement::frame, "frame ID")
      .add_property("oMf", bp::make_getter(&FramePlacement::oMf, bp::return_internal_reference<>()), "frame placement")
      .def(PrintableVisitor<FramePlacement>());

  bp::class_<FrameMotion>("FrameMotion",
                          "Frame motion describe using Pinocchio.\n\n"
                          "It defines a frame motion (tangent of SE(3) point) for a given frame ID",
                          bp::init<FrameIndex, pinocchio::Motion>(bp::args("self", "frame", "oMf"),
                                                                  "Initialize the frame motion.\n\n"
                                                                  ":param frame: frame ID\n"
                                                                  ":param oMf: Frame motion w.r.t. the origin"))
      .def(bp::init<>(bp::args("self"), "Default initialization of the frame motion."))
      .def_readwrite("frame", &FrameMotion::frame, "frame ID")
      .add_property("oMf", bp::make_getter(&FrameMotion::oMf, bp::return_internal_reference<>()), "frame motion")
      .def(PrintableVisitor<FrameMotion>());

  bp::class_<FrameForce>("FrameForce",
                         "Frame force describe using Pinocchio.\n\n"
                         "It defines a frame force for a given frame ID",
                         bp::init<FrameIndex, pinocchio::Force>(bp::args("self", "frame", "oFf"),
                                                                "Initialize the frame force.\n\n"
                                                                ":param frame: frame ID\n"
                                                                ":param oFf: Frame force w.r.t. the origin"))
      .def(bp::init<>(bp::args("self"), "Default initialization of the frame force."))
      .def_readwrite("frame", &FrameForce::frame, "frame ID")
      .add_property("oFf", bp::make_getter(&FrameForce::oFf, bp::return_internal_reference<>()), "frame force")
      .def(PrintableVisitor<FrameForce>());

  bp::class_<FrameFrictionCone>(
      "FrameFrictionCone",
      "Frame friction cone.\n\n"
      "It defines a friction cone for a given frame ID",
      bp::init<FrameIndex, FrictionCone>(bp::args("self", "frame", "oRf"),
                                         "Initialize the frame friction cone.\n\n"
                                         ":param frame: frame ID\n"
                                         ":param oRf: Frame friction cone w.r.t. the origin"))
      .def(bp::init<>(bp::args("self"), "Default initialization of the frame friction cone."))
      .def_readwrite("frame", &FrameFrictionCone::frame, "frame ID")
      .add_property("oRf", bp::make_getter(&FrameFrictionCone::oRf, bp::return_internal_reference<>()),
                    "frame friction cone")
      .def(PrintableVisitor<FrameFrictionCone>());

  bp::class_<FrameCoPSupport>("FrameCoPSupport",
                              "Frame foot geometry.\n\n"
                              "It defines the ID of the contact frame and the geometry of the contact surface",
                              bp::init<FrameIndex, Eigen::Vector2d>(
                                  bp::args("self", "frame", "support_region"),
                                  "Initialize the frame foot geometry.\n\n"
                                  ":param frame: ID of the contact frame \n"
                                  ":param support_region: dimension of the foot surface dim = (length, width) "))
      .def(bp::init<>(bp::args("self"), "Default initialization of the frame CoP support."))
      .def("update_A", &FrameCoPSupport::update_A, "update the matrix that defines the support region")
      .add_property("frame",
                    bp::make_function(&FrameCoPSupport::get_frame, bp::return_value_policy<bp::return_by_value>()),
                    &FrameCoPSupport::set_frame, "frame ID")
      .add_property("support_region",
                    bp::make_function(&FrameCoPSupport::get_support_region, bp::return_internal_reference<>()),
                    &FrameCoPSupport::set_support_region, "support region")
      .add_property("A", bp::make_function(&FrictionCone::get_A, bp::return_internal_reference<>()),
                    "inequality matrix")
      .def(PrintableVisitor<FrameCoPSupport>());
}

}  // namespace python
}  // namespace crocoddyl
