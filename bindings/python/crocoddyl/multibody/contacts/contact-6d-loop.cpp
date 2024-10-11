///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, LAAS-CNRS, University of Edinburgh
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/contacts/contact-6d-loop.hpp"

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/utils/copyable.hpp"

namespace crocoddyl {
namespace python {

void exposeContact6DLoop() {
  bp::register_ptr_to_python<boost::shared_ptr<ContactModel6DLoop> >();

  bp::class_<ContactModel6DLoop, bp::bases<ContactModelAbstract> >(
      "ContactModel6DLoop",
      "Rigid 6D contact model.\n\n"
      "It defines a rigid 6D contact models based on acceleration-based "
      "holonomic constraints.\n"
      "The calc and calcDiff functions compute the contact Jacobian and drift "
      "(holonomic constraint) or\n"
      "the derivatives of the holonomic constraint, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, int, pinocchio::SE3, int,
               pinocchio::SE3, pinocchio::ReferenceFrame, std::size_t,
               bp::optional<Eigen::Vector2d> >(
          bp::args("self", "state", "joint1_id", "joint1_placement",
                   "joint2_id", "joint2_placement", "ref", "nu", "gains"),
          "Initialize the contact model.\n\n"
          ":param state: state of the multibody system\n"
          ":param joint1_id: Parent joint id of the first contact frame\n"
          ":param joint1_placement: Placement of the first contact frame with "
          "respect to the parent joint\n"
          ":param joint2_id: Parent joint id of the second contact frames\n"
          ":param joint2_placement: Placement of the second contact frame with "
          "respect to the parent joint\n"
          ":param ref: reference frame of contact (must be pinocchio::LOCAL)\n"
          ":param nu: dimension of control vector\n"
          ":param gains: gains of the contact model (default "
          "np.matrix([0.,0.]))"))
      .def("calc", &ContactModel6DLoop::calc, bp::args("self", "data", "x"),
           "Compute the 6D loop-contact Jacobian and drift.\n\n"
           "The rigid contact model throught acceleration-base holonomic "
           "constraint\n"
           "of the contact frame placement.\n"
           ":param data: contact data\n"
           ":param x: state point (dim. state.nx)")
      .def("calcDiff", &ContactModel6DLoop::calcDiff,
           bp::args("self", "data", "x"),
           "Compute the derivatives of the 6D loop-contact holonomic "
           "constraint.\n\n"
           "The rigid contact model throught acceleration-base holonomic "
           "constraint\n"
           "of the contact frame placement.\n"
           "It assumes that calc has been run first.\n"
           ":param data: cost data\n"
           ":param x: state point (dim. state.nx)")
      .def("updateForce", &ContactModel6DLoop::updateForce,
           bp::args("self", "data", "force"),
           "Convert the Lagrangian into a stack of spatial forces.\n\n"
           ":param data: cost data\n"
           ":param force: force vector (dimension 6)")
      .def("updateForceDiff", &ContactModel6DLoop::updateForceDiff,
           bp::args("self", "data", "df_dx", "df_du"),
           "Update the force derivatives.\n\n"
           ":param data: cost data\n"
           ":param df_dx: Jacobian of the force with respect to the state\n"
           ":param df_du: Jacobian of the force with respect to the control")
      .def("createData", &ContactModel6DLoop::createData,
           bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the 6D loop-contact data.\n\n"
           "Each contact model has its own data that needs to be allocated. "
           "This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: Pinocchio data\n"
           ":return contact data.")
      .add_property(
          "joint1_id",
          bp::make_function(&ContactModel6DLoop::get_joint1_id,
                            bp::return_value_policy<bp::return_by_value>()),
          "Parent joint id of the first contact frame")
      .add_property(
          "joint1_placement",
          bp::make_function(&ContactModel6DLoop::get_joint1_placement,
                            bp::return_value_policy<bp::return_by_value>()),
          "Placement of the first contact frame with respect to the parent "
          "joint")
      .add_property(
          "joint2_id",
          bp::make_function(&ContactModel6DLoop::get_joint2_id,
                            bp::return_value_policy<bp::return_by_value>()),
          "Parent joint id of the second contact frame")
      .add_property(
          "joint2_placement",
          bp::make_function(&ContactModel6DLoop::get_joint2_placement,
                            bp::return_value_policy<bp::return_by_value>()),
          "Placement of the second contact frame with respect to the parent "
          "joint")
      .add_property(
          "gains",
          bp::make_function(&ContactModel6DLoop::get_gains,
                            bp::return_value_policy<bp::return_by_value>()),
          "Baumegarte stabilisation gains (Kp, Kd)")
      .def(CopyableVisitor<ContactModel6DLoop>());

  bp::register_ptr_to_python<boost::shared_ptr<ContactData6DLoop> >();

  bp::class_<ContactData6DLoop, bp::bases<ContactDataAbstract> >(
      "ContactData6DLoop", "Data for 6DLoop contact.\n\n",
      bp::init<ContactModel6DLoop*, pinocchio::Data*>(
          bp::args("self", "model", "data"),
          "Create 6D loop-contact data.\n\n"
          ":param model: 6D loop-contact model\n"
          ":param data: Pinocchio data")[bp::with_custodian_and_ward<
          1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property(
          "v1_partial_dq",
          bp::make_getter(&ContactData6DLoop::v1_partial_dq,
                          bp::return_internal_reference<>()),
          "Jacobian of the spatial velocity of the first contact frame wrt q")
      .add_property("a1_partial_dq",
                    bp::make_getter(&ContactData6DLoop::a1_partial_dq,
                                    bp::return_internal_reference<>()),
                    "Jacobian of the spatial acceleration of the first contact "
                    "frame wrt q")
      .add_property("a1_partial_dv",
                    bp::make_getter(&ContactData6DLoop::a1_partial_dv,
                                    bp::return_internal_reference<>()),
                    "Jacobian of the spatial acceleration of the first contact "
                    "frame wrt v")
      .add_property("a1_partial_da",
                    bp::make_getter(&ContactData6DLoop::a1_partial_da,
                                    bp::return_internal_reference<>()),
                    "Jacobian of the spatial acceleration of the first contact "
                    "frame wrt a")
      .add_property(
          "v2_partial_dq",
          bp::make_getter(&ContactData6DLoop::v2_partial_dq,
                          bp::return_internal_reference<>()),
          "Jacobian of the spatial velocity of the second contact frame wrt q")
      .add_property("a2_partial_dq",
                    bp::make_getter(&ContactData6DLoop::a2_partial_dq,
                                    bp::return_internal_reference<>()),
                    "Jacobian of the spatial acceleration of the second "
                    "contact frame wrt q")
      .add_property("a2_partial_dv",
                    bp::make_getter(&ContactData6DLoop::a2_partial_dv,
                                    bp::return_internal_reference<>()),
                    "Jacobian of the spatial acceleration of the second "
                    "contact frame wrt v")
      .add_property("a2_partial_da",
                    bp::make_getter(&ContactData6DLoop::a2_partial_da,
                                    bp::return_internal_reference<>()),
                    "Jacobian of the spatial acceleration of the second "
                    "contact frame wrt a")
      .add_property("da0_dx",
                    bp::make_getter(&ContactData6DLoop::da0_dx,
                                    bp::return_internal_reference<>()),
                    "Jacobian of the acceleration drift wrt x")
      .add_property("da0_dq_t1",
                    bp::make_getter(&ContactData6DLoop::da0_dq_t1,
                                    bp::return_internal_reference<>()),
                    "Jacobian of the acceleration drift wrt q - part 1")
      .add_property("da0_dq_t2",
                    bp::make_getter(&ContactData6DLoop::da0_dq_t2,
                                    bp::return_internal_reference<>()),
                    "Jacobian of the acceleration drift wrt q - part 2")
      .add_property("da0_dq_t3",
                    bp::make_getter(&ContactData6DLoop::da0_dq_t3,
                                    bp::return_internal_reference<>()),
                    "Jacobian of the acceleration drift wrt q - part 3")
      .add_property("j1Xf1",
                    bp::make_getter(&ContactData6DLoop::j1Xf1,
                                    bp::return_internal_reference<>()),
                    "Placement of the first contact frame in the joint frame - "
                    "Action Matrix")
      .add_property("j2Xf2",
                    bp::make_getter(&ContactData6DLoop::j2Xf2,
                                    bp::return_internal_reference<>()),
                    "Placement of the second contact frame in the joint frame "
                    "- Action Matrix")
      .add_property("f1Mf2",
                    bp::make_getter(&ContactData6DLoop::f1Mf2,
                                    bp::return_internal_reference<>()),
                    "Relative placement of the contact frames")
      .add_property("f1Xf2",
                    bp::make_getter(&ContactData6DLoop::f1Xf2,
                                    bp::return_internal_reference<>()),
                    "Relative placement of the contact frames - Action Matrix")
      .add_property("f1Jf1",
                    bp::make_getter(&ContactData6DLoop::f1Jf1,
                                    bp::return_internal_reference<>()),
                    "Jacobian of the first contact frame")
      .add_property("f2Jf2",
                    bp::make_getter(&ContactData6DLoop::f2Jf2,
                                    bp::return_internal_reference<>()),
                    "Jacobian of the second frame in the joint frame")
      .add_property("j1Jj1",
                    bp::make_getter(&ContactData6DLoop::j1Jj1,
                                    bp::return_internal_reference<>()),
                    "Jacobian of the first joint in the joint frame")
      .add_property("j2Jj2",
                    bp::make_getter(&ContactData6DLoop::j2Jj2,
                                    bp::return_internal_reference<>()),
                    "Jacobian of the second contact frame")
      .add_property("f1vf1",
                    bp::make_getter(&ContactData6DLoop::f1vf1,
                                    bp::return_internal_reference<>()),
                    "Velocity of the first contact frame")
      .add_property("f2vf2",
                    bp::make_getter(&ContactData6DLoop::f2vf2,
                                    bp::return_internal_reference<>()),
                    "Velocity of the second contact frame")
      .add_property(
          "f1vf2",
          bp::make_getter(&ContactData6DLoop::f1vf2,
                          bp::return_internal_reference<>()),
          "Velocity of the second contact frame in the first contact frame")
      .add_property("f1af1",
                    bp::make_getter(&ContactData6DLoop::f1af1,
                                    bp::return_internal_reference<>()),
                    "Acceleration of the first contact frame")
      .add_property("f2af2",
                    bp::make_getter(&ContactData6DLoop::f2af2,
                                    bp::return_internal_reference<>()),
                    "Acceleration of the second contact frame")
      .add_property(
          "f1af2",
          bp::make_getter(&ContactData6DLoop::f1af2,
                          bp::return_internal_reference<>()),
          "Acceleration of the second contact frame in the first contact frame")
      .add_property("joint1_f",
                    bp::make_getter(&ContactData6DLoop::joint1_f,
                                    bp::return_internal_reference<>()),
                    "Force at the first joint")
      .add_property("joint2_f",
                    bp::make_getter(&ContactData6DLoop::joint2_f,
                                    bp::return_internal_reference<>()),
                    "Force at the second joint")
      .def(CopyableVisitor<ContactData6DLoop>());
}

}  // namespace python
}  // namespace crocoddyl
