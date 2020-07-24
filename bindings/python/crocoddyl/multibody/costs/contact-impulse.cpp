///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2022, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/multibody/costs/contact-impulse.hpp"

namespace crocoddyl {
namespace python {

void exposeCostContactImpulse() {
  bp::class_<CostModelContactImpulse, bp::bases<CostModelAbstract> >(
      "CostModelContactImpulse",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameForce>(
          bp::args("self", "state", "activation", "fref"),
          "Initialize the contact impulse cost model.\n\n"
          "Note that the activation.nr is lower / equals than 6.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param fref: reference spatial contact impulse in the contact coordinates"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameForce, int>(
          bp::args("self", "state", "fref", "nr"),
          "Initialize the contact impulse cost model.\n\n"
          "For this case the default activation model is quadratic, i.e.\n"
          "crocoddyl.ActivationModelQuad(nr).\n"
          "Note that the nr is lower / equals than 6.\n"
          ":param state: state of the multibody system\n"
          ":param fref: reference spatial contact impulse in the contact coordinates\n"
          ":param nr: dimension of impulse vector (>= 6)"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameForce>(
          bp::args("self", "state", "fref"),
          "Initialize the contact impulse cost model.\n\n"
          "For this case the default activation model is quadratic, i.e.\n"
          "crocoddyl.ActivationModelQuad(6).\n"
          ":param state: state of the multibody system\n"
          ":param fref: reference spatial contact impulse in the contact coordinates"))
      .def<void (CostModelContactImpulse::*)(const boost::shared_ptr<CostDataAbstract>&,
                                             const Eigen::Ref<const Eigen::VectorXd>&,
                                             const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &CostModelContactImpulse::calc, bp::args("self", "data", "x", "u"),
          "Compute the contact impulse cost.\n\n"
          ":param data: cost data\n"
          ":param x: state vector\n"
          ":param u: control input")
      .def<void (CostModelContactImpulse::*)(const boost::shared_ptr<CostDataAbstract>&,
                                             const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &CostModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (CostModelContactImpulse::*)(const boost::shared_ptr<CostDataAbstract>&,
                                             const Eigen::Ref<const Eigen::VectorXd>&,
                                             const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelContactImpulse::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the contact impulse cost.\n\n"
          ":param data: action data\n"
          ":param x: state vector\n"
          ":param u: control input\n")
      .def<void (CostModelContactImpulse::*)(const boost::shared_ptr<CostDataAbstract>&,
                                             const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &CostModelContactImpulse::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the contact impulse cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: shared data\n"
           ":return cost data.")
      .add_property("reference", bp::make_function(&CostModelContactImpulse::get_reference<FrameForce>),
                    &CostModelContactImpulse::set_reference<FrameForce>,
                    "reference spatial contact impulse in the contact coordinates")
      .add_property("fref", bp::make_function(&CostModelContactImpulse::get_reference<FrameForce>),
                    &CostModelContactImpulse::set_reference<FrameForce>,
                    "reference spatial contact impulse in the contact coordinates");

  bp::register_ptr_to_python<boost::shared_ptr<CostDataContactImpulse> >();

  bp::class_<CostDataContactImpulse, bp::bases<CostDataAbstract> >(
      "CostDataContactImpulse", "Data for contact impulse cost.\n\n",
      bp::init<CostModelContactImpulse*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create contact impulse cost data.\n\n"
          ":param model: contact impulse cost model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property(
          "impulse", bp::make_getter(&CostDataContactImpulse::impulse, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&CostDataContactImpulse::impulse), "impulse data associated with the current cost");
}

}  // namespace python
}  // namespace crocoddyl
