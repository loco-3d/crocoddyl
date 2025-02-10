///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2023, LAAS-CNRS, University of Edinburgh
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/contacts/contact-1d.hpp"

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/utils/copyable.hpp"

namespace crocoddyl {
namespace python {

void exposeContact1D() {
  bp::register_ptr_to_python<std::shared_ptr<ContactModel1D> >();

  bp::class_<ContactModel1D, bp::bases<ContactModelAbstract> >(
      "ContactModel1D",
      "Rigid 1D contact model.\n\n"
      "It defines a rigid 1D contact model (point contact) based on "
      "acceleration-based holonomic constraints, in the "
      "z "
      "direction.\n"
      "The calc and calcDiff functions compute the contact Jacobian and drift "
      "(holonomic constraint) or\n"
      "the derivatives of the holonomic constraint, respectively.",
      bp::init<std::shared_ptr<StateMultibody>, pinocchio::FrameIndex, double,
               pinocchio::ReferenceFrame, Eigen::Matrix3d, std::size_t,
               bp::optional<Eigen::Vector2d> >(
          bp::args("self", "state", "id", "xref", "type", "rotation", "nu",
                   "gains"),
          "Initialize the contact model.\n\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id of the contact\n"
          ":param xref: contact position used for the Baumgarte stabilization\n"
          ":param type: type of contact\n"
          ":param rotation: rotation of the reference frame's z axis"
          ":param nu: dimension of control vector\n"
          ":param gains: gains of the contact model (default "
          "np.matrix([0.,0.]))"))
      .def(bp::init<std::shared_ptr<StateMultibody>, pinocchio::FrameIndex,
                    double, pinocchio::ReferenceFrame,
                    bp::optional<Eigen::Vector2d> >(
          bp::args("self", "state", "id", "xref", "type", "gains"),
          "Initialize the contact model.\n\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id of the contact\n"
          ":param xref: contact position used for the Baumgarte stabilization\n"
          ":param type: type of contact\n"
          ":param gains: gains of the contact model (default "
          "np.matrix([0.,0.]))"))
      .def("calc", &ContactModel1D::calc, bp::args("self", "data", "x"),
           "Compute the 1D contact Jacobian and drift.\n\n"
           "The rigid contact model throught acceleration-base holonomic "
           "constraint\n"
           "of the contact frame placement.\n"
           ":param data: contact data\n"
           ":param x: state point (dim. state.nx)")
      .def("calcDiff", &ContactModel1D::calcDiff, bp::args("self", "data", "x"),
           "Compute the derivatives of the 1D contact holonomic constraint.\n\n"
           "The rigid contact model throught acceleration-base holonomic "
           "constraint\n"
           "of the contact frame placement.\n"
           "It assumes that calc has been run first.\n"
           ":param data: cost data\n"
           ":param x: state point (dim. state.nx)")
      .def("updateForce", &ContactModel1D::updateForce,
           bp::args("self", "data", "force"),
           "Convert the force into a stack of spatial forces.\n\n"
           ":param data: cost data\n"
           ":param force: force vector (dimension 1)")
      .def("createData", &ContactModel1D::createData,
           bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the 1D contact data.\n\n"
           "Each contact model has its own data that needs to be allocated. "
           "This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: Pinocchio data\n"
           ":return contact data.")
      .add_property(
          "reference",
          bp::make_function(&ContactModel1D::get_reference,
                            bp::return_value_policy<bp::return_by_value>()),
          &ContactModel1D::set_reference, "reference contact translation")
      .add_property("Raxis",
                    bp::make_function(&ContactModel1D::get_axis_rotation,
                                      bp::return_internal_reference<>()),
                    &ContactModel1D::set_axis_rotation,
                    "rotation of the reference frame's z axis")
      .add_property(
          "gains",
          bp::make_function(&ContactModel1D::get_gains,
                            bp::return_value_policy<bp::return_by_value>()),
          "contact gains")
      .def(CopyableVisitor<ContactModel1D>());

  bp::register_ptr_to_python<std::shared_ptr<ContactData1D> >();

  bp::class_<ContactData1D, bp::bases<ContactDataAbstract> >(
      "ContactData1D", "Data for 1D contact.\n\n",
      bp::init<ContactModel1D*, pinocchio::Data*>(
          bp::args("self", "model", "data"),
          "Create 1D contact data.\n\n"
          ":param model: 1D contact model\n"
          ":param data: Pinocchio data")[bp::with_custodian_and_ward<
          1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property(
          "v",
          bp::make_getter(&ContactData1D::v,
                          bp::return_value_policy<bp::return_by_value>()),
          "spatial velocity of the contact body")
      .add_property("a0_local",
                    bp::make_getter(&ContactData1D::a0_local,
                                    bp::return_internal_reference<>()),
                    bp::make_setter(&ContactData1D::a0_local),
                    "desired local contact acceleration")
      .add_property("a0_skew",
                    bp::make_getter(&ContactData1D::a0_skew,
                                    bp::return_internal_reference<>()),
                    bp::make_setter(&ContactData1D::a0_skew),
                    "contact acceleration skew (local)")
      .add_property("a0_world_skew",
                    bp::make_getter(&ContactData1D::a0_world_skew,
                                    bp::return_internal_reference<>()),
                    bp::make_setter(&ContactData1D::a0_world_skew),
                    "contact acceleration skew (world)")
      .add_property(
          "dp",
          bp::make_getter(&ContactData1D::dp,
                          bp::return_internal_reference<>()),
          bp::make_setter(&ContactData1D::dp),
          "Translation error computed for the Baumgarte regularization term")
      .add_property("dp_local",
                    bp::make_getter(&ContactData1D::dp_local,
                                    bp::return_internal_reference<>()),
                    bp::make_setter(&ContactData1D::dp_local),
                    "local translation error computed for the Baumgarte "
                    "regularization term")
      .add_property("f_local",
                    bp::make_getter(&ContactData1D::f_local,
                                    bp::return_internal_reference<>()),
                    bp::make_setter(&ContactData1D::f_local),
                    "spatial contact force in local coordinates")
      .add_property("da0_local_dx",
                    bp::make_getter(&ContactData1D::da0_local_dx,
                                    bp::return_internal_reference<>()),
                    bp::make_setter(&ContactData1D::da0_local_dx),
                    "Jacobian of the desired local contact acceleration")
      .add_property("fJf",
                    bp::make_getter(&ContactData1D::fJf,
                                    bp::return_internal_reference<>()),
                    "local Jacobian of the contact frame")
      .add_property("v_partial_dq",
                    bp::make_getter(&ContactData1D::v_partial_dq,
                                    bp::return_internal_reference<>()),
                    "Jacobian of the spatial body velocity")
      .add_property("a_partial_dq",
                    bp::make_getter(&ContactData1D::a_partial_dq,
                                    bp::return_internal_reference<>()),
                    "Jacobian of the spatial body acceleration")
      .add_property("a_partial_dv",
                    bp::make_getter(&ContactData1D::a_partial_dv,
                                    bp::return_internal_reference<>()),
                    "Jacobian of the spatial body acceleration")
      .add_property("a_partial_da",
                    bp::make_getter(&ContactData1D::a_partial_da,
                                    bp::return_internal_reference<>()),
                    "Jacobian of the spatial body acceleration")
      .add_property(
          "oRf",
          bp::make_getter(&ContactData1D::oRf,
                          bp::return_internal_reference<>()),
          "Rotation matrix of the contact body expressed in the world frame")
      .def(CopyableVisitor<ContactData1D>());
}

}  // namespace python
}  // namespace crocoddyl
