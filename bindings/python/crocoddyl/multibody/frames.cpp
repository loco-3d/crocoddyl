///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <eigenpy/memory.hpp>
#include <eigenpy/eigen-to-python.hpp>

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/frames.hpp"
#include "python/crocoddyl/utils/printable.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"

#include "pinocchio/bindings/python/utils/std-aligned-vector.hpp"

EIGENPY_DEFINE_STRUCT_ALLOCATOR_SPECIALIZATION(crocoddyl::FrameTranslation)
EIGENPY_DEFINE_STRUCT_ALLOCATOR_SPECIALIZATION(crocoddyl::FrameRotation)
EIGENPY_DEFINE_STRUCT_ALLOCATOR_SPECIALIZATION(crocoddyl::FramePlacement)
EIGENPY_DEFINE_STRUCT_ALLOCATOR_SPECIALIZATION(crocoddyl::FrameForce)
EIGENPY_DEFINE_STRUCT_ALLOCATOR_SPECIALIZATION(crocoddyl::FrameMotion)
EIGENPY_DEFINE_STRUCT_ALLOCATOR_SPECIALIZATION(crocoddyl::FrameFrictionCone)
EIGENPY_DEFINE_STRUCT_ALLOCATOR_SPECIALIZATION(crocoddyl::FrameWrenchCone)
EIGENPY_DEFINE_STRUCT_ALLOCATOR_SPECIALIZATION(crocoddyl::FrameCoPSupport)

namespace crocoddyl {
namespace python {

void exposeFrames() {
  bp::class_<FrameTranslation>(
      "FrameTranslation",
      "Frame translation describe using Pinocchio.\n\n"
      "It defines a frame translation (3D vector) for a given frame ID",
      bp::init<FrameIndex, Eigen::Vector3d>(bp::args("self", "id", "translation"),
                                            "Initialize the frame translation.\n\n"
                                            ":param id: frame ID\n"
                                            ":param translation: Frame translation w.r.t. the origin"))
      .def(bp::init<>(bp::args("self"), "Default initialization of the frame translation."))
      .def_readwrite("id", &FrameTranslation::id, "frame ID")
      .add_property("translation", bp::make_getter(&FrameTranslation::translation, bp::return_internal_reference<>()),
                    bp::make_setter(&FrameTranslation::translation), "frame translation")
      .add_property(
          "frame",
          bp::make_getter(&FrameTranslation::id,
                          deprecated<bp::return_value_policy<bp::copy_non_const_reference> >("Deprecated. Use id")),
          bp::make_setter(&FrameTranslation::id, deprecated<>("Deprecated. Use id")), "frame ID")
      .add_property("oxf",
                    bp::make_getter(&FrameTranslation::translation,
                                    deprecated<bp::return_internal_reference<> >("Deprecated. Use translation.")),
                    bp::make_setter(&FrameTranslation::translation, deprecated<>("Deprecated. Use translation.")),
                    "frame translation")
      .def(PrintableVisitor<FrameTranslation>());

  bp::class_<FrameRotation>("FrameRotation",
                            "Frame rotation describe using Pinocchio.\n\n"
                            "It defines a frame rotation (rotation matrix) for a given frame ID",
                            bp::init<FrameIndex, Eigen::Matrix3d>(bp::args("self", "id", "rotation"),
                                                                  "Initialize the frame rotation.\n\n"
                                                                  ":param id: frame ID\n"
                                                                  ":param rotation: Frame rotation w.r.t. the origin"))
      .def(bp::init<>(bp::args("self"), "Default initialization of the frame rotation."))
      .def_readwrite("id", &FrameRotation::id, "frame ID")
      .add_property("rotation", bp::make_getter(&FrameRotation::rotation, bp::return_internal_reference<>()),
                    bp::make_setter(&FrameRotation::rotation), "frame rotation")
      .add_property(
          "frame",
          bp::make_getter(&FrameRotation::id,
                          deprecated<bp::return_value_policy<bp::copy_non_const_reference> >("Deprecated. Use id")),
          bp::make_setter(&FrameRotation::id, deprecated<>("Deprecated. Use id")), "frame ID")
      .add_property("oRf",
                    bp::make_getter(&FrameRotation::rotation,
                                    deprecated<bp::return_internal_reference<> >("Deprecated. Use rotation.")),
                    bp::make_setter(&FrameRotation::rotation, deprecated<>("Deprecated. Use rotation.")),
                    "frame rotation")
      .def(PrintableVisitor<FrameRotation>());

  bp::class_<FramePlacement>(
      "FramePlacement",
      "Frame placement describe using Pinocchio.\n\n"
      "It defines a frame placement (SE(3) point) for a given frame ID",
      bp::init<FrameIndex, pinocchio::SE3>(bp::args("self", "id", "placement"),
                                           "Initialize the frame placement.\n\n"
                                           ":param id: frame ID\n"
                                           ":param placement: Frame placement w.r.t. the origin"))
      .def(bp::init<>(bp::args("self"), "Default initialization of the frame placement."))
      .def_readwrite("id", &FramePlacement::id, "frame ID")
      .add_property("placement", bp::make_getter(&FramePlacement::placement, bp::return_internal_reference<>()),
                    bp::make_setter(&FramePlacement::placement), "frame placement")
      .add_property(
          "frame",
          bp::make_getter(&FramePlacement::id,
                          deprecated<bp::return_value_policy<bp::copy_non_const_reference> >("Deprecated. Use id")),
          bp::make_setter(&FramePlacement::id, deprecated<>("Deprecated. Use id")), "frame ID")
      .add_property("oMf",
                    bp::make_getter(&FramePlacement::placement,
                                    deprecated<bp::return_internal_reference<> >("Deprecated. Use placement.")),
                    bp::make_setter(&FramePlacement::placement, deprecated<>("Deprecated. Use placement.")),
                    "frame placement")
      .def(PrintableVisitor<FramePlacement>());

  bp::class_<FrameMotion>("FrameMotion",
                          "Frame motion describe using Pinocchio.\n\n"
                          "It defines a frame motion (tangent of SE(3) point) for a given frame ID",
                          bp::init<FrameIndex, pinocchio::Motion, bp::optional<pinocchio::ReferenceFrame> >(
                              bp::args("self", "id", "motion", "reference"),
                              "Initialize the frame motion.\n\n"
                              ":param id: frame ID\n"
                              ":param motion: Frame motion w.r.t. the origin\n"
                              ":param reference: Reference frame (default LOCAL)"))
      .def(bp::init<>(bp::args("self"), "Default initialization of the frame motion."))
      .def_readwrite("id", &FrameMotion::id, "frame ID")
      .add_property("motion", bp::make_getter(&FrameMotion::motion, bp::return_internal_reference<>()),
                    bp::make_setter(&FrameMotion::motion), "frame motion")
      .def_readwrite("reference", &FrameMotion::reference, "reference frame")
      .add_property(
          "frame",
          bp::make_getter(&FrameMotion::id,
                          deprecated<bp::return_value_policy<bp::copy_non_const_reference> >("Deprecated. Use id")),
          bp::make_setter(&FrameMotion::id, deprecated<>("Deprecated. Use id")), "frame ID")
      .add_property("oMf",
                    bp::make_getter(&FrameMotion::motion,
                                    deprecated<bp::return_internal_reference<> >("Deprecated. Use motion.")),
                    bp::make_setter(&FrameMotion::motion, deprecated<>("Deprecated. Use motion.")), "frame motion")
      .def(PrintableVisitor<FrameMotion>());

  bp::class_<FrameForce>("FrameForce",
                         "Frame force describe using Pinocchio.\n\n"
                         "It defines a frame force for a given frame ID",
                         bp::init<FrameIndex, pinocchio::Force>(bp::args("self", "id", "force"),
                                                                "Initialize the frame force.\n\n"
                                                                ":param id: frame ID\n"
                                                                ":param force: Frame force w.r.t. the origin"))
      .def(bp::init<>(bp::args("self"), "Default initialization of the frame force."))
      .def_readwrite("id", &FrameForce::id, "frame ID")
      .add_property("force", bp::make_getter(&FrameForce::force, bp::return_internal_reference<>()),
                    bp::make_setter(&FrameForce::force), "frame force")
      .add_property(
          "frame",
          bp::make_getter(&FrameForce::id,
                          deprecated<bp::return_value_policy<bp::copy_non_const_reference> >("Deprecated. Use id")),
          bp::make_setter(&FrameForce::id, deprecated<>("Deprecated. Use id")), "frame ID")
      .add_property(
          "oFf",
          bp::make_getter(&FrameForce::force, deprecated<bp::return_internal_reference<> >("Deprecated. Use force.")),
          bp::make_setter(&FrameForce::force, deprecated<>("Deprecated. Use force.")), "frame force")
      .def(PrintableVisitor<FrameForce>());

  bp::class_<FrameFrictionCone>(
      "FrameFrictionCone",
      "Frame friction cone.\n\n"
      "It defines a friction cone for a given frame ID",
      bp::init<FrameIndex, FrictionCone>(bp::args("self", "id", "cone"),
                                         "Initialize the frame friction cone.\n\n"
                                         ":param id: frame ID\n"
                                         ":param cone: Frame friction cone w.r.t. the origin"))
      .def(bp::init<>(bp::args("self"), "Default initialization of the frame friction cone."))
      .def_readwrite("id", &FrameFrictionCone::id, "frame ID")
      .add_property("cone", bp::make_getter(&FrameFrictionCone::cone, bp::return_internal_reference<>()),
                    bp::make_setter(&FrameFrictionCone::cone), "frame friction cone")
      .add_property(
          "frame",
          bp::make_getter(&FrameFrictionCone::id,
                          deprecated<bp::return_value_policy<bp::copy_non_const_reference> >("Deprecated. Use id")),
          bp::make_setter(&FrameFrictionCone::id, deprecated<>("Deprecated. Use id")), "frame ID")
      .add_property("oFf",
                    bp::make_getter(&FrameFrictionCone::cone,
                                    deprecated<bp::return_internal_reference<> >("Deprecated. Use cone.")),
                    bp::make_setter(&FrameFrictionCone::cone, deprecated<>("Deprecated. Use cone.")),
                    "frame friction cone")
      .def(PrintableVisitor<FrameFrictionCone>());

  bp::class_<FrameWrenchCone>("FrameWrenchCone",
                              "Frame wrench cone.\n\n"
                              "It defines a wrench cone for a given frame ID",
                              bp::init<FrameIndex, WrenchCone>(bp::args("self", "id", "cone"),
                                                               "Initialize the frame wrench cone.\n\n"
                                                               ":param id: frame ID\n"
                                                               ":param cone: Frame wrench cone w.r.t. the origin"))
      .def(bp::init<>(bp::args("self"), "Default initialization of the frame wrench cone."))
      .def_readwrite("id", &FrameWrenchCone::id, "frame ID")
      .add_property("cone", bp::make_getter(&FrameWrenchCone::cone, bp::return_internal_reference<>()),
                    bp::make_setter(&FrameWrenchCone::cone), "frame wrench cone")
      .def(PrintableVisitor<FrameWrenchCone>());

  bp::class_<FrameCoPSupport>(
      "FrameCoPSupport",
      "Frame foot geometry.\n\n"
      "It defines the ID of the contact frame and the geometry of the contact surface",
      bp::init<FrameIndex, Eigen::Vector2d>(bp::args("self", "id", "box"),
                                            "Initialize the frame foot geometry.\n\n"
                                            ":param id: ID of the contact frame \n"
                                            ":param box: dimension of the foot surface dim = (length, width)"))
      .def(bp::init<>(bp::args("self"), "Default initialization of the frame CoP support."))
      .def("update_A", &FrameCoPSupport::update_A, "update the matrix that defines the support region")
      .add_property("id", bp::make_function(&FrameCoPSupport::get_id, bp::return_value_policy<bp::return_by_value>()),
                    &FrameCoPSupport::set_id, "frame ID")
      .add_property("box", bp::make_function(&FrameCoPSupport::get_box, bp::return_internal_reference<>()),
                    bp::make_function(&FrameCoPSupport::set_box), "box size used to define the sole")
      .add_property("support_region",
                    bp::make_function(&FrameCoPSupport::get_box,
                                      deprecated<bp::return_internal_reference<> >("Deprecated. Use box.")),
                    bp::make_function(&FrameCoPSupport::set_box, deprecated<>("Deprecated. Use box.")),
                    "box size used to define the sole")
      //   .add_property("A", bp::make_function(&FrameCoPSupport::get_A, bp::return_internal_reference<>()),
      //                 "inequality matrix") // TODO(cmastalli) we cannot expose due to a compilation error with
      //                 Matrix46
      .add_property(
          "frame",
          bp::make_function(&FrameCoPSupport::get_id,
                            deprecated<bp::return_value_policy<bp::return_by_value> >("Deprecated. Use id.")),
          bp::make_function(&FrameCoPSupport::set_id, deprecated<>("Deprecated. Use id.")), "frame ID")
      .def(PrintableVisitor<FrameCoPSupport>());

  pinocchio::python::StdAlignedVectorPythonVisitor<FramePlacement, true>::expose("StdVec_FramePlacement");

  pinocchio::python::StdAlignedVectorPythonVisitor<FrameRotation, true>::expose("StdVec_FrameRotation");

  pinocchio::python::StdAlignedVectorPythonVisitor<FrameTranslation, true>::expose("StdVec_FrameTranslation");
  pinocchio::python::StdAlignedVectorPythonVisitor<FrameForce, true>::expose("StdVec_FrameForce");

  pinocchio::python::StdAlignedVectorPythonVisitor<FrameMotion, true>::expose("StdVec_FrameMotion");
}

}  // namespace python
}  // namespace crocoddyl
