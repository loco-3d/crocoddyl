///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/residuals/contact-impulse.hpp"

namespace crocoddyl {
namespace python {

void exposeResidualContactImpulse() {
  bp::register_ptr_to_python<boost::shared_ptr<ResidualModelContactImpulse> >();

  bp::class_<ResidualModelContactImpulse, bp::bases<ResidualModelAbstract> >(
      "ResidualModelContactImpulse",
      "This residual function defines a residual vector as r = f-fref, where f,fref describe the current and "
      "reference "
      "the spatial impulses, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, pinocchio::FrameIndex, pinocchio::Force>(
          bp::args("self", "state", "id", "fref"),
          "Initialize the contact impulse residual model.\n\n"
          ":param state: state of the multibody system\n"
          ":param id: reference frame id\n"
          ":param fref: reference spatial contact impulse in the contact coordinates"))
      .def<void (ResidualModelContactImpulse::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelContactImpulse::calc, bp::args("self", "data", "x", "u"),
          "Compute the contact impulse residual.\n\n"
          ":param data: residual data\n"
          ":param x: state vector\n"
          ":param u: control input")
      .def<void (ResidualModelContactImpulse::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (ResidualModelContactImpulse::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelContactImpulse::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the Jacobians of the contact impulse residual.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: state vector\n"
          ":param u: control input\n")
      .def<void (ResidualModelContactImpulse::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &ResidualModelContactImpulse::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the contact impulse residual data.\n\n"
           "Each residual model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined residual.\n"
           ":param data: shared data\n"
           ":return residual data.")
      .add_property("id", &ResidualModelContactImpulse::get_id, &ResidualModelContactImpulse::set_id,
                    "reference frame id")
      .add_property("reference",
                    bp::make_function(&ResidualModelContactImpulse::get_reference, bp::return_internal_reference<>()),
                    &ResidualModelContactImpulse::set_reference, "reference spatial force");

  bp::register_ptr_to_python<boost::shared_ptr<ResidualDataContactImpulse> >();

  bp::class_<ResidualDataContactImpulse, bp::bases<ResidualDataAbstract> >(
      "ResidualDataContactImpulse", "Data for contact impulse residual.\n\n",
      bp::init<ResidualModelContactImpulse*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create contact impulse residual data.\n\n"
          ":param model: contact impulse residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property(
          "impulse",
          bp::make_getter(&ResidualDataContactImpulse::impulse, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&ResidualDataContactImpulse::impulse), "impulse data associated with the current residual");
}

}  // namespace python
}  // namespace crocoddyl
