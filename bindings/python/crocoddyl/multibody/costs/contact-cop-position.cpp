///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, University of Duisburg-Essen, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/multibody/costs/contact-cop-position.hpp"

namespace crocoddyl {
namespace python {

void exposeCostContactCoPPosition() {
  bp::class_<CostModelContactCoPPosition, bp::bases<CostModelAbstract> >(
      "CostModelContactCoPPosition",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameCoPSupport, int>(
          bp::args("self", "state", "activation", "cop_support", "nu"),
          "Initialize the contact CoP position cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model (default ActivationModelQuadraticBarrier)\n"
          ":param cop_support: contact frame ID and cop support region\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameCoPSupport>(
          bp::args("self", "state", "activation", "cop_support"),
          "Initialize the contact CoP position cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model (default ActivationModelQuadraticBarrier)\n"
          ":param cop_support: contact frame ID and cop support region"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameCoPSupport, int>(
          bp::args("self", "state", "cop_support", "nu"),
          "Initialize the contact CoP position cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param cop_support: contact frame ID and cop support region\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameCoPSupport>(
          bp::args("self", "state", "cop_support"),
          "Initialize the contact CoP position cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param cop_support: contact frame ID and cop support region"))
      .def<void (CostModelContactCoPPosition::*)(const boost::shared_ptr<CostDataAbstract>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &CostModelContactCoPPosition::calc, bp::args("self", "data", "x", "u"),
          "Compute the contact CoP position cost.\n\n"
          ":param data: cost data\n"
          ":param x: state vector\n"
          ":param u: control input")
      .def<void (CostModelContactCoPPosition::*)(const boost::shared_ptr<CostDataAbstract>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&,
                                                 const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelContactCoPPosition::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the contact CoP position cost.\n\n"
          ":param data: action data\n"
          ":param x: state vector\n"
          ":param u: control input\n")
      .def("createData", &CostModelContactCoPPosition::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the contact CoP position cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: shared data\n"
           ":return cost data.")
      .add_property("reference",
                    bp::make_function(&CostModelContactCoPPosition::get_copSupport, bp::return_internal_reference<>()),
                    &CostModelContactCoPPosition::set_reference<FrameCoPSupport>, "reference foot geometry");

  bp::register_ptr_to_python<boost::shared_ptr<CostDataContactCoPPosition> >();

  bp::class_<CostDataContactCoPPosition, bp::bases<CostDataAbstract> >(
      "CostDataContactCoPPosition", "Data for contact CoP position cost.\n\n",
      bp::init<CostModelContactCoPPosition*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create contact CoP position cost data.\n\n"
          ":param model: contact CoP position cost model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("Arr_Ru", bp::make_getter(&CostDataContactCoPPosition::Arr_Ru, bp::return_internal_reference<>()),
                    "Intermediate product of Arr (2nd deriv of Activation) with Ru (deriv of residue)")
      .add_property(
          "contact",
          bp::make_getter(&CostDataContactCoPPosition::contact, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&CostDataContactCoPPosition::contact), "contact data associated with the current cost");
}

}  // namespace python
}  // namespace crocoddyl
