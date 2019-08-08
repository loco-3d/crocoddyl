///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef PYTHON_CROCODDYL_MULTIBODY_CONTACT_BASE_HPP_
#define PYTHON_CROCODDYL_MULTIBODY_CONTACT_BASE_HPP_

#include "crocoddyl/multibody/contact-base.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

class ContactModelAbstract_wrap : public ContactModelAbstract, public bp::wrapper<ContactModelAbstract> {
 public:
  ContactModelAbstract_wrap(StateMultibody& state, int nc) : ContactModelAbstract(state, nc) {}

  void calc(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x) {
    return bp::call<void>(this->get_override("calc").ptr(), data, (Eigen::VectorXd)x);
  }

  void calcDiff(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const bool& recalc = true) {
    return bp::call<void>(this->get_override("calcDiff").ptr(), data, (Eigen::VectorXd)x, recalc);
  }
};

void exposeContactAbstract() {
  bp::class_<ContactModelAbstract_wrap, boost::noncopyable>(
      "ContactModelAbstract",
      "Abstract rigid contact model.\n\n"
      "It defines a template for rigid contact models based on acceleration-based holonomic constraints.\n"
      "The calc and calcDiff functions compute the contact Jacobian and drift (holonomic constraint) or\n"
      "the derivatives of the holonomic constraint, respectively.",
      bp::init<StateMultibody&, int>(bp::args(" self", " state", " nc"),
                                     "Initialize the contact model.\n\n"
                                     ":param state: state of the multibody system\n"
                                     ":param nc: dimension of contact model")[bp::with_custodian_and_ward<1, 2>()])
      .def("calc", pure_virtual(&ContactModelAbstract_wrap::calc), bp::args(" self", " data", " x"),
           "Compute the contact Jacobian and drift.\n\n"
           "The rigid contact model throught acceleration-base holonomic constraint\n"
           "of the contact frame placement.\n"
           ":param data: contact data\n"
           ":param x: state vector")
      .def("calcDiff", pure_virtual(&ContactModelAbstract_wrap::calcDiff),
           bp::args(" self", " data", " x", " recalc=True"),
           "Compute the derivatives of contact holonomic constraint.\n\n"
           "The rigid contact model throught acceleration-base holonomic constraint\n"
           "of the contact frame placement.\n"
           ":param data: cost data\n"
           ":param x: state vector\n"
           ":param recalc: If true, it updates the contact Jacobian and drift.")
      .def("createData", &ContactModelAbstract_wrap::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args(" self", " data"),
           "Create the contact data.\n\n"
           "Each contact model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: Pinocchio data\n"
           ":return contact data.")
      .add_property("State",
                    bp::make_function(&ContactModelAbstract_wrap::get_state, bp::return_internal_reference<>()),
                    "state of the multibody system")
      .add_property(
          "nc", bp::make_function(&ContactModelAbstract_wrap::get_nc, bp::return_value_policy<bp::return_by_value>()),
          "dimension of contact");

  bp::class_<ContactDataAbstract, boost::shared_ptr<ContactDataAbstract>, boost::noncopyable>(
      "ContactDataAbstract", "Abstract class for contact datas.\n\n",
      bp::init<ContactModelAbstract*, pinocchio::Data*>(
          bp::args(" self", " model", " data"),
          "Create common data shared between contact models.\n\n"
          ":param model: cost model\n"
          ":param data: Pinocchio data")[bp::with_custodian_and_ward<1, 3>()])
      .add_property("pinocchio", bp::make_getter(&ContactDataAbstract::pinocchio, bp::return_internal_reference<>()),
                    "pinocchio data")
      .add_property("Jc", bp::make_getter(&ContactDataAbstract::Jc, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ContactDataAbstract::Jc), "contact Jacobian")
      .add_property("a0", bp::make_getter(&ContactDataAbstract::a0, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ContactDataAbstract::a0), "contact drift")
      .add_property("Ax", bp::make_getter(&ContactDataAbstract::Ax, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ContactDataAbstract::Ax), "derivatives of the contact constraint");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // PYTHON_CROCODDYL_MULTIBODY_CONTACT_BASE_HPP_