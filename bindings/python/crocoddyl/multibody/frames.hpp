///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_FRAMES_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_FRAMES_HPP_

#include "crocoddyl/multibody/frames.hpp"
#include "python/crocoddyl/utils/printable.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeFrames() {
  bp::class_<FrameTranslation, boost::noncopyable>(
      "FrameTranslation",
      "Frame translation describe using Pinocchio.\n\n"
      "It defines a frame translation (3D vector) for a given frame ID",
      bp::init<FrameIndex, Eigen::Vector3d>(bp::args(" self", " frame", " oxf"),
                                            "Initialize the frame translation.\n\n"
                                            ":param frame: frame ID\n"
                                            ":param oxf: Frame translation w.r.t. the origin"))
      .def_readwrite("frame", &FrameTranslation::frame, "frame ID")
      .add_property("oxf", bp::make_getter(&FrameTranslation::oxf, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&FrameTranslation::oxf), "frame translation")
      .def(PrintableVisitor<FrameTranslation>());

  bp::class_<FrameRotation, boost::noncopyable>(
      "FrameRotation",
      "Frame rotation describe using Pinocchio.\n\n"
      "It defines a frame rotation (rotation matrix) for a given frame ID",
      bp::init<FrameIndex, Eigen::Matrix3d>(bp::args(" self", " frame", " oRf"),
                                            "Initialize the frame translation.\n\n"
                                            ":param frame: frame ID\n"
                                            ":param oRf: Frame rotation w.r.t. the origin"))
      .def_readwrite("frame", &FrameRotation::frame, "frame ID")
      .add_property("oRf", bp::make_getter(&FrameRotation::oRf, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&FrameRotation::oRf), "frame rotation")
      .def(PrintableVisitor<FrameRotation>());

  bp::class_<FramePlacement, boost::noncopyable>(
      "FramePlacement",
      "Frame placement describe using Pinocchio.\n\n"
      "It defines a frame placement (SE(3) point) for a given frame ID",
      bp::init<FrameIndex, pinocchio::SE3>(bp::args(" self", " frame", " oMf"),
                                           "Initialize the frame placement.\n\n"
                                           ":param frame: frame ID\n"
                                           ":param oMf: Frame placement w.r.t. the origin"))
      .def_readwrite("frame", &FramePlacement::frame, "frame ID")
      .add_property("oMf", bp::make_getter(&FramePlacement::oMf, bp::return_internal_reference<>()), "frame placement")
      .def(PrintableVisitor<FramePlacement>());

  bp::class_<FrameMotion, boost::noncopyable>(
      "FrameMotion",
      "Frame motion describe using Pinocchio.\n\n"
      "It defines a frame motion (tangent of SE(3) point) for a given frame ID",
      bp::init<FrameIndex, pinocchio::Motion>(bp::args(" self", " frame", " oMf"),
                                              "Initialize the frame motion.\n\n"
                                              ":param frame: frame ID\n"
                                              ":param oMf: Frame motion w.r.t. the origin"))
      .def_readwrite("frame", &FrameMotion::frame, "frame ID")
      .add_property("oMf", bp::make_getter(&FrameMotion::oMf, bp::return_internal_reference<>()), "frame motion")
      .def(PrintableVisitor<FrameMotion>());

  bp::class_<FrameForce, boost::noncopyable>(
      "FrameForce",
      "Frame force describe using Pinocchio.\n\n"
      "It defines a frame motion (tangent of SE(3) point) for a given frame ID",
      bp::init<FrameIndex, pinocchio::Force>(bp::args(" self", " frame", " oFf"),
                                             "Initialize the frame motion.\n\n"
                                             ":param frame: frame ID\n"
                                             ":param oFf: Frame force w.r.t. the origin"))
      .def_readwrite("frame", &FrameForce::frame, "frame ID")
      .add_property("oFf", bp::make_getter(&FrameForce::oFf, bp::return_internal_reference<>()), "frame force")
      .def(PrintableVisitor<FrameForce>());
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_FRAMES_HPP_
