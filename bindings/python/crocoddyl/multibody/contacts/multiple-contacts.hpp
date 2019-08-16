///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_CONTACTS_MULTIPLE_CONTACTS_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_CONTACTS_MULTIPLE_CONTACTS_HPP_

#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <string>
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeContactMultiple() {
  // Register custom converters between std::map and Python dict
  bp::to_python_converter<std::map<std::string, ContactItem, std::less<std::string>,
                                   std::allocator<std::pair<const std::string, ContactItem> > >,
                          map_to_dict<std::string, ContactItem> >();
  dict_to_map<std::string, ContactItem>().from_python();

  bp::class_<ContactItem, boost::noncopyable>(
      "ContactItem", "Describe a contact item.\n\n",
      bp::init<std::string, ContactModelAbstract*>(
          bp::args(" self", " name", " contact"),
          "Initialize the contact item.\n\n"
          ":param name: contact name\n"
          ":param contact: contact model")[bp::with_custodian_and_ward<1, 3>()])
      .def_readwrite("name", &ContactItem::name, "contact name")
      .add_property("contact", bp::make_getter(&ContactItem::contact, bp::return_internal_reference<>()),
                    "contact model");

  bp::class_<ContactModelMultiple, bp::bases<ContactModelAbstract> >(
      "ContactModelMultiple",
      bp::init<StateMultibody&>(bp::args(" self", " state"),
                                "Initialize the multiple contact model.\n\n"
                                ":param state: state of the multibody system")[bp::with_custodian_and_ward<1, 2>()])
      .def("addContact", &ContactModelMultiple::addContact, bp::with_custodian_and_ward<1, 3>(), "add contact item")
      .def("removeContact", &ContactModelMultiple::removeContact, "remove contact item")
      .def("calc", &ContactModelMultiple::calc_wrap, bp::args(" self", " data", " x"),
           "Compute the total contact Jacobian and drift.\n\n"
           "The rigid contact model throught acceleration-base holonomic constraint\n"
           "of the contact frame placement.\n"
           ":param data: contact data\n"
           ":param x: state vector")
      .def("calcDiff", &ContactModelMultiple::calcDiff_wrap,
           ContactModel_calcDiff_wraps(bp::args(" self", " data", " x", " recalc=True"),
                                       "Compute the derivatives of the total contact holonomic constraint.\n\n"
                                       "The rigid contact model throught acceleration-base holonomic constraint\n"
                                       "of the contact frame placement.\n"
                                       ":param data: cost data\n"
                                       ":param x: state vector\n"
                                       ":param recalc: If true, it updates the contact Jacobian and drift."))
      .def("updateLagrangian", &ContactModelMultiple::updateLagrangian, bp::args(" self", " data", " lambda"),
           "Convert the Lagrangian into a stack of spatial forces.\n\n"
           ":param data: cost data\n"
           ":param lambda: Lagrangian vector")
      .def("createData", &ContactModelMultiple::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args(" self", " data"),
           "Create the total contact data.\n\n"
           ":param data: Pinocchio data\n"
           ":return total contact data.")
      .add_property(
          "contacts",
          bp::make_function(&ContactModelMultiple::get_contacts, bp::return_value_policy<bp::return_by_value>()),
          "stack of contacts")
      .add_property("nc",
                    bp::make_function(&ContactModelMultiple::get_nc, bp::return_value_policy<bp::return_by_value>()),
                    "dimension of the total contact vector");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_CONTACTS_MULTIPLE_CONTACTS_HPP_
