///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/contacts/contact-2d.hpp"

namespace crocoddyl {
namespace python {

void exposeContact2D() {
  bp::class_<ContactModel2D, bp::bases<ContactModelAbstract> >(
      "ContactModel2D",
      "Rigid 2D contact model.\n\n"
      "It defines a rigid 2D contact models (point contact) based on acceleration-based holonomic constraints, in x,z "
      "directions.\n"
      "The calc and calcDiff functions compute the contact Jacobian and drift (holonomic constraint) or\n"
      "the derivatives of the holonomic constraint, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, FrameTranslation, int, bp::optional<Eigen::Vector2d> >(
          bp::args("self", "state", "xref", "nu", "gains"),
          "Initialize the contact model.\n\n"
          ":param state: state of the multibody system\n"
          ":param xref: reference frame translation\n"
          ":param nu: dimension of control vector\n"
          ":param gains: gains of the contact model (default np.matrix([ [0.],[0.] ]))"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameTranslation, bp::optional<Eigen::Vector2d> >(
          bp::args("self", "state", "xref", "gains"),
          "Initialize the contact model.\n\n"
          ":param state: state of the multibody system\n"
          ":param Mref: reference frame translation\n"
          ":param gains: gains of the contact model (default np.matrix([ [0.],[0.] ]))"))
      .def("calc", &ContactModel2D::calc, bp::args("self", "data", "x"),
           "Compute the 2D contact Jacobian and drift.\n\n"
           "The rigid contact model throught acceleration-base holonomic constraint\n"
           "of the contact frame placement.\n"
           ":param data: contact data\n"
           ":param x: state vector")
      .def("calcDiff", &ContactModel2D::calcDiff, bp::args("self", "data", "x"),
           "Compute the derivatives of the 2D contact holonomic constraint.\n\n"
           "The rigid contact model throught acceleration-base holonomic constraint\n"
           "of the contact frame placement.\n"
           ":param data: cost data\n"
           ":param x: state vector\n")
      .def("updateForce", &ContactModel2D::updateForce, bp::args("self", "data", "force"),
           "Convert the force into a stack of spatial forces.\n\n"
           ":param data: cost data\n"
           ":param force: force vector (dimension 2)")
      .def("createData", &ContactModel2D::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the 2D contact data.\n\n"
           "Each contact model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: Pinocchio data\n"
           ":return contact data.")
      .add_property("xref", bp::make_function(&ContactModel2D::get_xref, bp::return_internal_reference<>()),
                    "reference frame translation")
      .add_property("gains",
                    bp::make_function(&ContactModel2D::get_gains, bp::return_value_policy<bp::return_by_value>()),
                    "contact gains");

  bp::register_ptr_to_python<boost::shared_ptr<ContactData2D> >();

  bp::class_<ContactData2D, bp::bases<ContactDataAbstract> >(
      "ContactData2D", "Data for 2D contact.\n\n",
      bp::init<ContactModel2D*, pinocchio::Data*>(
          bp::args("self", "model", "data"),
          "Create 2D contact data.\n\n"
          ":param model: 2D contact model\n"
          ":param data: Pinocchio data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("v", bp::make_getter(&ContactData2D::v, bp::return_value_policy<bp::return_by_value>()),
                    "spatial velocity of the contact body")
      .add_property("a", bp::make_getter(&ContactData2D::a, bp::return_value_policy<bp::return_by_value>()),
                    "spatial acceleration of the contact body")
      .add_property("fJf", bp::make_getter(&ContactData2D::fJf, bp::return_internal_reference<>()),
                    "local Jacobian of the contact frame")
      .add_property("v_partial_dq", bp::make_getter(&ContactData2D::v_partial_dq, bp::return_internal_reference<>()),
                    "Jacobian of the spatial body velocity")
      .add_property("a_partial_dq", bp::make_getter(&ContactData2D::a_partial_dq, bp::return_internal_reference<>()),
                    "Jacobian of the spatial body acceleration")
      .add_property("a_partial_dv", bp::make_getter(&ContactData2D::a_partial_dv, bp::return_internal_reference<>()),
                    "Jacobian of the spatial body acceleration")
      .add_property("a_partial_da", bp::make_getter(&ContactData2D::a_partial_da, bp::return_internal_reference<>()),
                    "Jacobian of the spatial body acceleration")
      .add_property("oRf", bp::make_getter(&ContactData2D::oRf, bp::return_internal_reference<>()),
                    "Rotation matrix of the contact body expressed in the world frame");
}

}  // namespace python
}  // namespace crocoddyl
