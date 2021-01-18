///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/costs/contact-force.hpp"
#include "python/crocoddyl/utils/deprecate.hpp"

namespace crocoddyl {
namespace python {

void exposeCostContactForce() {
  bp::class_<CostModelContactForce, bp::bases<CostModelAbstract> >(
      "CostModelContactForce",
      "This cost function defines a residual vector as r = f-fref, where f,fref describe the current and reference "
      "the spatial forces, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameForce, int>(
          bp::args("self", "state", "activation", "fref", "nu"),
          "Initialize the contact force cost model.\n\n"
          "Note that the activation.nr is lower / equals than 6.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param fref: reference spatial contact force in the contact coordinates\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, FrameForce>(
          bp::args("self", "state", "activation", "fref"),
          "Initialize the contact force cost model.\n\n"
          "The default nu is obtained from state.nv.\n"
          "Note that the activation.nr is lower / equals than 6.\n"
          ":param state: state of the multibody system\n"
          ":param activation: activation model\n"
          ":param fref: reference spatial contact force in the contact coordinates\n"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameForce, int, int>(
          bp::args("self", "state", "fref", "nr", "nu"),
          "Initialize the contact force cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2).\n"
          "Note that the nr is lower / equals than 6.\n"
          ":param state: state of the multibody system\n"
          ":param fref: reference spatial contact force in the contact coordinates\n"
          ":param nr: dimension of force vector (>= 6)\n"
          ":param nu: dimension of control vector"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameForce, int>(
          bp::args("self", "state", "fref", "nr"),
          "Initialize the contact force cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2), and nu is obtained from "
          "state.nv.\n"
          "Note that the nr is lower / equals than 6.\n"
          ":param state: state of the multibody system\n"
          ":param fref: reference spatial contact force in the contact coordinates\n"
          ":param nr: dimension of force vector (>= 6)"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, FrameForce>(
          bp::args("self", "state", "fref"),
          "Initialize the contact force cost model.\n\n"
          "We use ActivationModelQuad as a default activation model (i.e. a=0.5*||r||^2), and nu is obtained from "
          "state.nv.\n"
          ":param state: state of the multibody system\n"
          ":param fref: reference spatial contact force in the contact coordinates"))
      .def<void (CostModelContactForce::*)(const boost::shared_ptr<CostDataAbstract>&,
                                           const Eigen::Ref<const Eigen::VectorXd>&,
                                           const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &CostModelContactForce::calc, bp::args("self", "data", "x", "u"),
          "Compute the contact force cost.\n\n"
          ":param data: cost data\n"
          ":param x: state vector\n"
          ":param u: control input")
      .def<void (CostModelContactForce::*)(const boost::shared_ptr<CostDataAbstract>&,
                                           const Eigen::Ref<const Eigen::VectorXd>&)>("calc", &CostModelAbstract::calc,
                                                                                      bp::args("self", "data", "x"))
      .def<void (CostModelContactForce::*)(const boost::shared_ptr<CostDataAbstract>&,
                                           const Eigen::Ref<const Eigen::VectorXd>&,
                                           const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelContactForce::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the contact force cost.\n\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: state vector\n"
          ":param u: control input\n")
      .def<void (CostModelContactForce::*)(const boost::shared_ptr<CostDataAbstract>&,
                                           const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &CostModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &CostModelContactForce::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the contact force cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: shared data\n"
           ":return cost data.")
      .add_property("reference", &CostModelContactForce::get_reference<FrameForce>,
                    &CostModelContactForce::set_reference<FrameForce>,
                    "reference spatial contact force in the contact coordinates")
      .add_property("fref",
                    bp::make_function(&CostModelContactForce::get_reference<FrameForce>,
                                      deprecated<>("Deprecated. Use reference.")),
                    bp::make_function(&CostModelContactForce::set_reference<FrameForce>,
                                      deprecated<>("Deprecated. Use reference.")),
                    "reference spatial contact force in the contact coordinates");

  bp::register_ptr_to_python<boost::shared_ptr<CostDataContactForce> >();

  bp::class_<CostDataContactForce, bp::bases<CostDataAbstract> >(
      "CostDataContactForce", "Data for contact force cost.\n\n",
      bp::init<CostModelContactForce*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create contact force cost data.\n\n"
          ":param model: contact force cost model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("Arr_Ru", bp::make_getter(&CostDataContactForce::Arr_Ru, bp::return_internal_reference<>()),
                    "Intermediate product of Arr (2nd deriv of Activation) with Ru (deriv of residue)")
      .add_property("contact",
                    bp::make_getter(&CostDataContactForce::contact, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&CostDataContactForce::contact), "contact data associated with the current cost");
}

}  // namespace python
}  // namespace crocoddyl
