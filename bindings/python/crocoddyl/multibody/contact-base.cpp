///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/multibody/contact-base.hpp"

namespace crocoddyl {
namespace python {

void exposeContactAbstract() {
  bp::register_ptr_to_python<boost::shared_ptr<ContactModelAbstract> >();

  bp::class_<ContactModelAbstract_wrap, boost::noncopyable>(
      "ContactModelAbstract",
      "Abstract rigid contact model.\n\n"
      "It defines a template for rigid contact models based on acceleration-based holonomic constraints.\n"
      "The calc and calcDiff functions compute the contact Jacobian and drift (holonomic constraint) or\n"
      "the derivatives of the holonomic constraint, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, int, bp::optional<int> >(
          bp::args("self", "state", "nc", "nu"),
          "Initialize the contact model.\n\n"
          ":param state: state of the multibody system\n"
          ":param nc: dimension of contact model\n"
          ":param nu: dimension of the control vector"))
      .def("calc", pure_virtual(&ContactModelAbstract_wrap::calc), bp::args("self", "data", "x"),
           "Compute the contact Jacobian and drift.\n\n"
           "The rigid contact model throught acceleration-base holonomic constraint\n"
           "of the contact frame placement.\n"
           ":param data: contact data\n"
           ":param x: state vector")
      .def("calcDiff", pure_virtual(&ContactModelAbstract_wrap::calcDiff), bp::args("self", "data", "x"),
           "Compute the derivatives of contact holonomic constraint.\n\n"
           "The rigid contact model throught acceleration-base holonomic constraint\n"
           "of the contact frame placement.\n"
           "It assumes that calc has been run first.\n"
           ":param data: contact data\n"
           ":param x: state vector\n")
      .def("updateForce", pure_virtual(&ContactModelAbstract_wrap::updateForce), bp::args("self", "data", "force"),
           "Convert the force into a stack of spatial forces.\n\n"
           ":param data: contact data\n"
           ":param force: force vector (dimension nc)")
      .def("updateForceDiff", &ContactModelAbstract_wrap::updateForceDiff, bp::args("self", "data", "df_dx", "df_du"),
           "Update the Jacobians of the force.\n\n"
           ":param data: contact data\n"
           ":param df_dx: Jacobian of the force with respect to the state\n"
           ":param df_du: Jacobian of the force with respect to the control")
      .def("setZeroForce", &ContactModelAbstract_wrap::setZeroForce, bp::args("self", "data"),
           "Set zero the spatial force.\n\n"
           ":param data: contact data")
      .def("setZeroForceDiff", &ContactModelAbstract_wrap::setZeroForceDiff, bp::args("self", "data"),
           "Set zero the derivatives of the spatial force.\n\n"
           ":param data: contact data")
      .def("createData", &ContactModelAbstract_wrap::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the contact data.\n\n"
           "Each contact model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined contact.\n"
           ":param data: Pinocchio data\n"
           ":return contact data.")
      .def("createData", &ContactModelAbstract_wrap::default_createData, bp::with_custodian_and_ward_postcall<0, 2>())
      .add_property(
          "state",
          bp::make_function(&ContactModelAbstract_wrap::get_state, bp::return_value_policy<bp::return_by_value>()),
          "state of the multibody system")
      .add_property(
          "nc", bp::make_function(&ContactModelAbstract_wrap::get_nc, bp::return_value_policy<bp::return_by_value>()),
          "dimension of contact")
      .add_property(
          "nu", bp::make_function(&ContactModelAbstract_wrap::get_nu, bp::return_value_policy<bp::return_by_value>()),
          "dimension of control");

  bp::register_ptr_to_python<boost::shared_ptr<ContactDataAbstract> >();

  bp::class_<ContactDataAbstract, boost::noncopyable>(
      "ContactDataAbstract", "Abstract class for contact datas.\n\n",
      bp::init<ContactModelAbstract*, pinocchio::Data*>(
          bp::args("self", "model", "data"),
          "Create common data shared between contact models.\n\n"
          ":param model: contact model\n"
          ":param data: Pinocchio data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("pinocchio", bp::make_getter(&ContactDataAbstract::pinocchio, bp::return_internal_reference<>()),
                    "pinocchio data")
      .add_property("jMf", bp::make_getter(&ContactDataAbstract::jMf, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ContactDataAbstract::jMf), "local frame placement of the contact frame")
      .add_property("fXj", bp::make_getter(&ContactDataAbstract::fXj, bp::return_internal_reference<>()),
                    bp::make_setter(&ContactDataAbstract::fXj), "action matrix from contact to local frames")
      .add_property("Jc", bp::make_getter(&ContactDataAbstract::Jc, bp::return_internal_reference<>()),
                    bp::make_setter(&ContactDataAbstract::Jc), "contact Jacobian")
      .add_property("a0", bp::make_getter(&ContactDataAbstract::a0, bp::return_internal_reference<>()),
                    bp::make_setter(&ContactDataAbstract::a0), "desired contact acceleration")
      .add_property("da0_dx", bp::make_getter(&ContactDataAbstract::da0_dx, bp::return_internal_reference<>()),
                    bp::make_setter(&ContactDataAbstract::da0_dx), "Jacobian of the desired contact acceleration")
      .add_property("df_dx", bp::make_getter(&ContactDataAbstract::df_dx, bp::return_internal_reference<>()),
                    bp::make_setter(&ContactDataAbstract::df_dx), "Jacobian of the contact forces")
      .add_property("df_du", bp::make_getter(&ContactDataAbstract::df_du, bp::return_internal_reference<>()),
                    bp::make_setter(&ContactDataAbstract::df_du), "Jacobian of the contact forces")
      .def_readwrite("joint", &ContactDataAbstract::joint, "joint index of the contact frame")
      .def_readwrite("frame", &ContactDataAbstract::frame, "frame index of the contact frame")
      .def_readwrite("f", &ContactDataAbstract::f,
                     "external spatial force at the parent joint level. Note that we could compute the force at the "
                     "contact frame by using jMf (i.e. data.jMf.actInv(data.f)");
}

}  // namespace python
}  // namespace crocoddyl
