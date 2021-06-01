///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <string>
#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/utils/map-converter.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "python/crocoddyl/utils/printable.hpp"

namespace crocoddyl {
namespace python {

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(ContactModelMultiple_addContact_wrap, ContactModelMultiple::addContact, 2, 3)

void exposeContactMultiple() {
  // Register custom converters between std::map and Python dict
  typedef boost::shared_ptr<ContactItem> ContactItemPtr;
  typedef boost::shared_ptr<ContactDataAbstract> ContactDataPtr;
  StdMapPythonVisitor<std::string, ContactItemPtr, std::less<std::string>,
                      std::allocator<std::pair<const std::string, ContactItemPtr> >,
                      true>::expose("StdMap_ContactItem");
  StdMapPythonVisitor<std::string, ContactDataPtr, std::less<std::string>,
                      std::allocator<std::pair<const std::string, ContactDataPtr> >,
                      true>::expose("StdMap_ContactData");

  bp::register_ptr_to_python<boost::shared_ptr<ContactItem> >();

  bp::class_<ContactItem>("ContactItem", "Describe a contact item.\n\n",
                          bp::init<std::string, boost::shared_ptr<ContactModelAbstract>, bp::optional<bool> >(
                              bp::args("self", "name", "contact", "active"),
                              "Initialize the contact item.\n\n"
                              ":param name: contact name\n"
                              ":param contact: contact model\n"
                              ":param active: True if the contact is activated (default true)"))
      .def_readwrite("name", &ContactItem::name, "contact name")
      .add_property("contact", bp::make_getter(&ContactItem::contact, bp::return_value_policy<bp::return_by_value>()),
                    "contact model")
      .def_readwrite("active", &ContactItem::active, "contact status")
      .def(PrintableVisitor<ContactItem>());

  bp::register_ptr_to_python<boost::shared_ptr<ContactModelMultiple> >();

  bp::class_<ContactModelMultiple>("ContactModelMultiple",
                                   bp::init<boost::shared_ptr<StateMultibody>, bp::optional<std::size_t> >(
                                       bp::args("self", "state", "nu"),
                                       "Initialize the multiple contact model.\n\n"
                                       ":param state: state of the multibody system\n"
                                       ":param nu: dimension of control vector"))
      .def("addContact", &ContactModelMultiple::addContact,
           ContactModelMultiple_addContact_wrap(bp::args("self", "name", "contact", "active"),
                                                "Add a contact item.\n\n"
                                                ":param name: contact name\n"
                                                ":param contact: contact model\n"
                                                ":param active: True if the contact is activated (default true)"))
      .def("removeContact", &ContactModelMultiple::removeContact, bp::args("self", "name"),
           "Remove a contact item.\n\n"
           ":param name: contact name")
      .def("changeContactStatus", &ContactModelMultiple::changeContactStatus, bp::args("self", "name", "active"),
           "Change the contact status.\n\n"
           ":param name: contact name\n"
           ":param active: contact status (true for active and false for inactive)")
      .def("calc", &ContactModelMultiple::calc, bp::args("self", "data", "x"),
           "Compute the contact Jacobian and contact acceleration.\n\n"
           "The rigid contact model throught acceleration-base holonomic constraint\n"
           "of the contact frame placement.\n"
           ":param data: contact data\n"
           ":param x: state vector")
      .def("calcDiff", &ContactModelMultiple::calcDiff, bp::args("self", "data", "x"),
           "Compute the derivatives of the contact holonomic constraint.\n\n"
           "The rigid contact model throught acceleration-base holonomic constraint\n"
           "of the contact frame placement.\n"
           "It assumes that calc has been run first.\n"
           ":param data: contact data\n"
           ":param x: state vector\n")
      .def("updateAcceleration", &ContactModelMultiple::updateAcceleration, bp::args("self", "data", "dv"),
           "Update the constrained system acceleration.\n\n"
           ":param data: contact data\n"
           ":param dv: constrained acceleration (dimension nv)")
      .def("updateForce", &ContactModelMultiple::updateForce, bp::args("self", "data", "force"),
           "Update the spatial force in frame coordinate.\n\n"
           ":param data: contact data\n"
           ":param force: force vector (dimension nc)")
      .def("updateAccelerationDiff", &ContactModelMultiple::updateAccelerationDiff, bp::args("self", "data", "ddv_dx"),
           "Update the Jacobian of the constrained system acceleration.\n\n"
           ":param data: contact data\n"
           ":param ddv_dx: Jacobian of the system acceleration in generalized coordinates (dimension nv*ndx)")
      .def("updateForceDiff", &ContactModelMultiple::updateForceDiff, bp::args("self", "data", "df_dx", "df_du"),
           "Update the Jacobians of the spatial force defined in frame coordinates.\n\n"
           ":param data: contact data\n"
           ":param df_dx: Jacobian of the force with respect to the state (dimension nc*ndx)\n"
           ":param df_du: Jacobian of the force with respect to the control (dimension nc*nu)")
      .def("createData", &ContactModelMultiple::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the total contact data.\n\n"
           ":param data: Pinocchio data\n"
           ":return total contact data.")
      .add_property(
          "contacts",
          bp::make_function(&ContactModelMultiple::get_contacts, bp::return_value_policy<bp::return_by_value>()),
          "stack of contacts")
      .add_property(
          "state", bp::make_function(&ContactModelMultiple::get_state, bp::return_value_policy<bp::return_by_value>()),
          "state of the multibody system")
      .add_property("nc", bp::make_function(&ContactModelMultiple::get_nc), "dimension of the active contact vector")
      .add_property("nc_total", bp::make_function(&ContactModelMultiple::get_nc_total),
                    "dimension of the total contact vector")
      .add_property("nu", bp::make_function(&ContactModelMultiple::get_nu), "dimension of control vector")
      .add_property(
          "active",
          bp::make_function(&ContactModelMultiple::get_active, bp::return_value_policy<bp::return_by_value>()),
          "name of active contact items")
      .add_property(
          "inactive",
          bp::make_function(&ContactModelMultiple::get_inactive, bp::return_value_policy<bp::return_by_value>()),
          "name of inactive contact items")
      .def("getContactStatus", &ContactModelMultiple::getContactStatus, bp::args("self", "name"),
           "Return the contact status of a given contact name.\n\n"
           ":param name: contact name")
      .def(PrintableVisitor<ContactModelMultiple>());

  bp::register_ptr_to_python<boost::shared_ptr<ContactDataMultiple> >();

  bp::class_<ContactDataMultiple>(
      "ContactDataMultiple", "Data class for multiple contacts.\n\n",
      bp::init<ContactModelMultiple*, pinocchio::Data*>(
          bp::args("self", "model", "data"),
          "Create multicontact data.\n\n"
          ":param model: multicontact model\n"
          ":param data: Pinocchio data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("Jc", bp::make_getter(&ContactDataMultiple::Jc, bp::return_internal_reference<>()),
                    bp::make_setter(&ContactDataMultiple::Jc),
                    "contact Jacobian in frame coordinate (memory defined for active and inactive contacts)")
      .add_property(
          "a0", bp::make_getter(&ContactDataMultiple::a0, bp::return_internal_reference<>()),
          bp::make_setter(&ContactDataMultiple::a0),
          "desired spatial contact acceleration in frame coordinate (memory defined for active and inactive contacts)")
      .add_property(
          "da0_dx", bp::make_getter(&ContactDataMultiple::da0_dx, bp::return_internal_reference<>()),
          bp::make_setter(&ContactDataMultiple::da0_dx),
          "Jacobian of the desired spatial contact acceleration in frame coordinate (memory defined for active and "
          "inactive contacts)")
      .add_property("dv", bp::make_getter(&ContactDataMultiple::dv, bp::return_internal_reference<>()),
                    bp::make_setter(&ContactDataMultiple::dv),
                    "constrained system acceleration in generalized coordinates")
      .add_property("ddv_dx", bp::make_getter(&ContactDataMultiple::ddv_dx, bp::return_internal_reference<>()),
                    bp::make_setter(&ContactDataMultiple::ddv_dx),
                    "Jacobian of the constrained system acceleration in generalized coordinates")
      .add_property("contacts",
                    bp::make_getter(&ContactDataMultiple::contacts, bp::return_value_policy<bp::return_by_value>()),
                    "stack of contacts data")
      .def_readwrite("fext", &ContactDataMultiple::fext, "external spatial forces in join coordinates");
}

}  // namespace python
}  // namespace crocoddyl
