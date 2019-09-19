///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_CONTACTS_CONTACT_6D_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_CONTACTS_CONTACT_6D_HPP_

#include "crocoddyl/multibody/contacts/contact-6d.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeContact6D() {
  bp::class_<ContactModel6D, bp::bases<ContactModelAbstract> >(
      "ContactModel6D",
      "Rigid 6D contact model.\n\n"
      "It defines a rigid 6D contact models based on acceleration-based holonomic constraints.\n"
      "The calc and calcDiff functions compute the contact Jacobian and drift (holonomic constraint) or\n"
      "the derivatives of the holonomic constraint, respectively.",
      bp::init<StateMultibody&, FramePlacement, int, bp::optional<Eigen::Vector2d> >(
          bp::args(" self", " state", " Mref", " nu=state.nv", " gains=np.matrix([ [0.],[0.] ]"),
          "Initialize the contact model.\n\n"
          ":param state: state of the multibody system\n"
          ":param Mref: reference frame placement\n"
          ":param nu: dimension of control vector\n"
          ":param gains: gains of the contact model")[bp::with_custodian_and_ward<1, 2>()])
      .def(bp::init<StateMultibody&, FramePlacement, bp::optional<Eigen::Vector2d> >(
          bp::args(" self", " state", " Mref"),
          "Initialize the state cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param Mref: reference frame placement\n"
          ":param nu = dimension of control vector")[bp::with_custodian_and_ward<1, 2>()])
      .def("calc", &ContactModel6D::calc_wrap, bp::args(" self", " data", " x"),
           "Compute the 6D contact Jacobian and drift.\n\n"
           "The rigid contact model throught acceleration-base holonomic constraint\n"
           "of the contact frame placement.\n"
           ":param data: contact data\n"
           ":param x: state vector")
      .def("calcDiff", &ContactModel6D::calcDiff_wrap,
           ContactModel_calcDiff_wraps(bp::args(" self", " data", " x", " recalc=True"),
                                       "Compute the derivatives of the 6D contact holonomic constraint.\n\n"
                                       "The rigid contact model throught acceleration-base holonomic constraint\n"
                                       "of the contact frame placement.\n"
                                       ":param data: cost data\n"
                                       ":param x: state vector\n"
                                       ":param recalc: If true, it updates the contact Jacobian and drift."))
      .def("updateForce", &ContactModel6D::updateForce, bp::args(" self", " data", " force"),
           "Convert the Lagrangian into a stack of spatial forces.\n\n"
           ":param data: cost data\n"
           ":param force: force vector (dimension 6)")
      .def("createData", &ContactModel6D::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args(" self", " data"),
           "Create the 6D contact data.\n\n"
           "Each contact model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: Pinocchio data\n"
           ":return contact data.")
      .add_property("Mref", bp::make_function(&ContactModel6D::get_Mref, bp::return_internal_reference<>()),
                    "reference frame placement")
      .add_property("gains",
                    bp::make_function(&ContactModel6D::get_gains, bp::return_value_policy<bp::return_by_value>()),
                    "contact gains");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_CONTACTS_CONTACT_6D_HPP_
