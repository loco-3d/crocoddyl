///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, LAAS-CNRS, University of Edinburgh
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/contacts/contact-6d.hpp"

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/utils/copyable.hpp"

namespace crocoddyl {
namespace python {

void exposeContact6D() {
  bp::register_ptr_to_python<std::shared_ptr<ContactModel6D> >();

#pragma GCC diagnostic push  // TODO: Remove once the deprecated FrameXX has
                             // been removed in a future release
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

  bp::class_<ContactModel6D, bp::bases<ContactModelAbstract> >(
      "ContactModel6D",
      "Rigid 6D contact model.\n\n"
      "It defines a rigid 6D contact models based on acceleration-based "
      "holonomic constraints.\n"
      "The calc and calcDiff functions compute the contact Jacobian and drift "
      "(holonomic constraint) or\n"
      "the derivatives of the holonomic constraint, respectively.",
      bp::init<std::shared_ptr<StateMultibody>, pinocchio::FrameIndex,
               pinocchio::SE3, pinocchio::ReferenceFrame, std::size_t,
               bp::optional<Eigen::Vector2d> >(
          bp::args("self", "state", "id", "pref", "type", "nu", "gains"),
          "Initialize the contact model.\n\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id of the contact\n"
          ":param pref: contact placement used for the Baumgarte "
          "stabilization\n"
          ":param type: type of contact\n"
          ":param nu: dimension of control vector\n"
          ":param gains: gains of the contact model (default "
          "np.matrix([0.,0.]))"))
      .def(bp::init<std::shared_ptr<StateMultibody>, pinocchio::FrameIndex,
                    pinocchio::SE3, pinocchio::ReferenceFrame,
                    bp::optional<Eigen::Vector2d> >(
          bp::args("self", "state", "id", "pref", "type", "gains"),
          "Initialize the contact model.\n\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id of the contact\n"
          ":param pref: contact placement used for the Baumgarte "
          "stabilization\n"
          ":param type: type of contact\n"
          ":param gains: gains of the contact model (default "
          "np.matrix([0.,0.]))"))
      .def("calc", &ContactModel6D::calc, bp::args("self", "data", "x"),
           "Compute the 6D contact Jacobian and drift.\n\n"
           "The rigid contact model throught acceleration-base holonomic "
           "constraint\n"
           "of the contact frame placement.\n"
           ":param data: contact data\n"
           ":param x: state point (dim. state.nx)")
      .def("calcDiff", &ContactModel6D::calcDiff, bp::args("self", "data", "x"),
           "Compute the derivatives of the 6D contact holonomic constraint.\n\n"
           "The rigid contact model throught acceleration-base holonomic "
           "constraint\n"
           "of the contact frame placement.\n"
           "It assumes that calc has been run first.\n"
           ":param data: cost data\n"
           ":param x: state point (dim. state.nx)")
      .def("updateForce", &ContactModel6D::updateForce,
           bp::args("self", "data", "force"),
           "Convert the Lagrangian into a stack of spatial forces.\n\n"
           ":param data: cost data\n"
           ":param force: force vector (dimension 6)")
      .def("createData", &ContactModel6D::createData,
           bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the 6D contact data.\n\n"
           "Each contact model has its own data that needs to be allocated. "
           "This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: Pinocchio data\n"
           ":return contact data.")
      .add_property("reference",
                    bp::make_function(&ContactModel6D::get_reference,
                                      bp::return_internal_reference<>()),
                    &ContactModel6D::set_reference,
                    "reference contact placement")
      .add_property(
          "gains",
          bp::make_function(&ContactModel6D::get_gains,
                            bp::return_value_policy<bp::return_by_value>()),
          "contact gains")
      .def(CopyableVisitor<ContactModel6D>());

#pragma GCC diagnostic pop

  bp::register_ptr_to_python<std::shared_ptr<ContactData6D> >();

  bp::class_<ContactData6D, bp::bases<ContactDataAbstract> >(
      "ContactData6D", "Data for 6D contact.\n\n",
      bp::init<ContactModel6D*, pinocchio::Data*>(
          bp::args("self", "model", "data"),
          "Create 6D contact data.\n\n"
          ":param model: 6D contact model\n"
          ":param data: Pinocchio data")[bp::with_custodian_and_ward<
          1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property(
          "rMf",
          bp::make_getter(&ContactData6D::jMf,
                          bp::return_value_policy<bp::return_by_value>()),
          "error frame placement of the contact frame")
      .add_property(
          "v",
          bp::make_getter(&ContactData6D::v,
                          bp::return_value_policy<bp::return_by_value>()),
          "spatial velocity of the contact body")
      .add_property(
          "a0_local",
          bp::make_getter(&ContactData6D::a0_local,
                          bp::return_value_policy<bp::return_by_value>()),
          "desired local contact acceleration")
      .add_property("v_partial_dq",
                    bp::make_getter(&ContactData6D::v_partial_dq,
                                    bp::return_internal_reference<>()),
                    "Jacobian of the spatial body velocity")
      .add_property("a_partial_dq",
                    bp::make_getter(&ContactData6D::a_partial_dq,
                                    bp::return_internal_reference<>()),
                    "Jacobian of the spatial body acceleration")
      .add_property("a_partial_dv",
                    bp::make_getter(&ContactData6D::a_partial_dv,
                                    bp::return_internal_reference<>()),
                    "Jacobian of the spatial body acceleration")
      .add_property("a_partial_da",
                    bp::make_getter(&ContactData6D::a_partial_da,
                                    bp::return_internal_reference<>()),
                    "Jacobian of the spatial body acceleration")
      .def(CopyableVisitor<ContactData6D>());
}

}  // namespace python
}  // namespace crocoddyl
