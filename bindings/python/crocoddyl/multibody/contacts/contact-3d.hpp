///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_CONTACTS_CONTACT_3D_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_CONTACTS_CONTACT_3D_HPP_

#include "crocoddyl/multibody/contacts/contact-3d.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeContact3D() {
  bp::class_<ContactModel3D, bp::bases<ContactModelAbstract> >(
      "ContactModel3D",
      "Rigid 3D contact model.\n\n"
      "It defines a rigid 3D contact models (point contact) based on acceleration-based holonomic constraints.\n"
      "The calc and calcDiff functions compute the contact Jacobian and drift (holonomic constraint) or\n"
      "the derivatives of the holonomic constraint, respectively.",
      bp::init<StateMultibody&, FrameTranslation, int, bp::optional<Eigen::Vector2d> >(
          bp::args(" self", " state", " xref", " nu=state.nv", " gains=np.matrix([ [0.],[0.] ]"),
          "Initialize the contact model.\n\n"
          ":param state: state of the multibody system\n"
          ":param xref: reference frame translation\n"
          ":param nu: dimension of control vector\n"
          ":param gains: gains of the contact model")[bp::with_custodian_and_ward<1, 2>()])
      .def(bp::init<StateMultibody&, FrameTranslation, bp::optional<Eigen::Vector2d> >(
          bp::args(" self", " state", " xref"),
          "Initialize the state cost model.\n\n"
          "For this case the default activation model is quadratic, i.e. crocoddyl.ActivationModelQuad(state.ndx).\n"
          ":param state: state of the multibody system\n"
          ":param xref: reference state")[bp::with_custodian_and_ward<1, 2>()])
      .def("calc", &ContactModel3D::calc_wrap, bp::args(" self", " data", " x"),
           "Compute the 3D contact Jacobian and drift.\n\n"
           "The rigid contact model throught acceleration-base holonomic constraint\n"
           "of the contact frame placement.\n"
           ":param data: contact data\n"
           ":param x: state vector")
      .def("calcDiff", &ContactModel3D::calcDiff_wrap,
           ContactModel_calcDiff_wraps(bp::args(" self", " data", " x", " recalc=True"),
                                       "Compute the derivatives of the 3D contact holonomic constraint.\n\n"
                                       "The rigid contact model throught acceleration-base holonomic constraint\n"
                                       "of the contact frame placement.\n"
                                       ":param data: cost data\n"
                                       ":param x: state vector\n"
                                       ":param recalc: If true, it updates the contact Jacobian and drift."))
      .def("updateLagrangian", &ContactModel3D::updateLagrangian, bp::args(" self", " data", " lambda"),
           "Convert the Lagrangian into a stack of spatial forces.\n\n"
           ":param data: cost data\n"
           ":param lambda: Lagrangian vector")
      .def("createData", &ContactModel3D::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args(" self", " data"),
           "Create the 3D contact data.\n\n"
           "Each contact model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: Pinocchio data\n"
           ":return contact data.")
      .add_property("xref", bp::make_function(&ContactModel3D::get_xref, bp::return_internal_reference<>()),
                    "reference frame translation")
      .add_property("gains",
                    bp::make_function(&ContactModel3D::get_gains, bp::return_value_policy<bp::return_by_value>()),
                    "contact gains");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_CONTACTS_CONTACT_3D_HPP_
