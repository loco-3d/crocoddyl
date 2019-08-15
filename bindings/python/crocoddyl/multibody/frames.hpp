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

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeFrames() {
  bp::class_<FrameTranslation, boost::noncopyable>(
      "FrameTranslation",
      "Frame translation describe using Pinocchio.\n\n"
      "It defines a frame translation (3D vector) for a given frame ID",
      bp::init<int, Eigen::Vector3d>(bp::args(" self", " frame", " oxf"),
                                     "Initialize the frame translation.\n\n"
                                     ":param frame: frame ID\n"
                                     ":param oxf: Frame translation w.r.t. the origin"))
      .def_readwrite("frame", &FrameTranslation::frame, "frame ID")
      .add_property("oxf", bp::make_getter(&FrameTranslation::oxf, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&FrameTranslation::oxf), "frame translation");

  bp::class_<FramePlacement, boost::noncopyable>(
      "FramePlacement",
      "Frame placement describe using Pinocchio.\n\n"
      "It defines a frame placement (SE(3) point) for a given frame ID",
      bp::init<int, pinocchio::SE3>(bp::args(" self", " frame", " oMf"),
                                    "Initialize the frame placement.\n\n"
                                    ":param frame: frame ID\n"
                                    ":param oMf: Frame placement w.r.t. the origin"))
      .def_readwrite("frame", &FramePlacement::frame, "frame ID")
      .add_property("oMf", bp::make_getter(&FramePlacement::oMf, bp::return_internal_reference<>()),
                    "frame placement");

  bp::class_<FrameMotion, boost::noncopyable>(
      "FrameMotion",
      "Frame motion describe using Pinocchio.\n\n"
      "It defines a frame motion (tangent of SE(3) point) for a given frame ID",
      bp::init<int, pinocchio::Motion>(bp::args(" self", " frame", " oMf"),
                                       "Initialize the frame motion.\n\n"
                                       ":param frame: frame ID\n"
                                       ":param oMf: Frame motion w.r.t. the origin"))
      .def_readwrite("frame", &FrameMotion::frame, "frame ID")
      .add_property("oMf", bp::make_getter(&FrameMotion::oMf, bp::return_internal_reference<>()), "frame motion");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_FRAMES_HPP_
