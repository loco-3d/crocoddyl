///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_ACTIONS_CONTACT_FWDDYN_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_ACTIONS_CONTACT_FWDDYN_HPP_

#include "crocoddyl/multibody/actions/contact-fwddyn.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeDifferentialActionContactFwdDynamics() {
  bp::class_<DifferentialActionModelContactFwdDynamics, bp::bases<DifferentialActionModelAbstract> >(
      "DifferentialActionModelContactFwdDynamics",
      "Differential action model for contact forward dynamics in multibody systems.\n\n"
      "The contact is modelled as holonomic constraits in the contact frame. There\n"
      "is also a custom implementation in case of system with armatures. If you want to\n"
      "include the armature, you need to use setArmature(). On the other hand, the\n"
      "stack of cost functions are implemented in CostModelSum().",
      bp::init<StateMultibody&, ActuationModelFloatingBase&, ContactModelMultiple&, CostModelSum&>(
          bp::args(" self", " state", " actuation", " contacts", " costs"),
          "Initialize the constrained forward-dynamics action model.\n\n"
          ":param state: multibody state\n"
          ":param actuation: floating-base actuation model\n"
          ":param contacts: multiple contact model\n"
          ":param costs: stack of cost functions")[bp::with_custodian_and_ward<1, 3>()])
      .def("calc", &DifferentialActionModelContactFwdDynamics::calc_wrap,
           DiffActionModel_calc_wraps(
               bp::args(" self", " data", " x", " u=None"),
               "Compute the next state and cost value.\n\n"
               "It describes the time-continuous evolution of the multibody system with contact. The\n"
               "contacts are modelled as holonomic constraints.\n"
               "Additionally it computes the cost value associated to this state and control pair.\n"
               ":param data: free forward-dynamics action data\n"
               ":param x: time-continuous state vector\n"
               ":param u: time-continuous control input"))
      .def<void (DifferentialActionModelContactFwdDynamics::*)(
          const boost::shared_ptr<DifferentialActionDataAbstract>&, const Eigen::VectorXd&, const Eigen::VectorXd&,
          const bool&)>("calcDiff", &DifferentialActionModelContactFwdDynamics::calcDiff_wrap,
                        bp::args(" self", " data", " x", " u=None", " recalc=True"),
                        "Compute the derivatives of the differential multibody system and its cost\n"
                        "functions.\n\n"
                        "It computes the partial derivatives of the differential multibody system and the\n"
                        "cost function. If recalc == True, it first updates the state evolution\n"
                        "and cost value. This function builds a quadratic approximation of the\n"
                        "action model (i.e. dynamical system and cost function).\n"
                        ":param data: free forward-dynamics action data\n"
                        ":param x: time-continuous state vector\n"
                        ":param u: time-continuous control input\n"
                        ":param recalc: If true, it updates the state evolution and the cost value.")
      .def<void (DifferentialActionModelContactFwdDynamics::*)(
          const boost::shared_ptr<DifferentialActionDataAbstract>&, const Eigen::VectorXd&, const Eigen::VectorXd&)>(
          "calcDiff", &DifferentialActionModelContactFwdDynamics::calcDiff_wrap,
          bp::args(" self", " data", " x", " u"))
      .def<void (DifferentialActionModelContactFwdDynamics::*)(
          const boost::shared_ptr<DifferentialActionDataAbstract>&, const Eigen::VectorXd&)>(
          "calcDiff", &DifferentialActionModelContactFwdDynamics::calcDiff_wrap, bp::args(" self", " data", " x"))
      .def<void (DifferentialActionModelContactFwdDynamics::*)(
          const boost::shared_ptr<DifferentialActionDataAbstract>&, const Eigen::VectorXd&, const bool&)>(
          "calcDiff", &DifferentialActionModelContactFwdDynamics::calcDiff_wrap,
          bp::args(" self", " data", " x", " recalc"))
      .def("createData", &DifferentialActionModelContactFwdDynamics::createData, bp::args(" self"),
           "Create the free forward dynamics differential action data.")
      .add_property("pinocchio",
                    bp::make_function(&DifferentialActionModelContactFwdDynamics::get_pinocchio,
                                      bp::return_internal_reference<>()),
                    "multibody model (i.e. pinocchio model)")
      .add_property("actuation",
                    bp::make_function(&DifferentialActionModelContactFwdDynamics::get_actuation,
                                      bp::return_internal_reference<>()),
                    "actuation model")
      .add_property("contacts",
                    bp::make_function(&DifferentialActionModelContactFwdDynamics::get_contacts,
                                      bp::return_internal_reference<>()),
                    "multiple contact model")
      .add_property(
          "costs",
          bp::make_function(&DifferentialActionModelContactFwdDynamics::get_costs, bp::return_internal_reference<>()),
          "total cost model")
      .add_property("armature",
                    bp::make_function(&DifferentialActionModelContactFwdDynamics::get_armature,
                                      bp::return_value_policy<bp::return_by_value>()),
                    bp::make_function(&DifferentialActionModelContactFwdDynamics::set_armature),
                    "set an armature mechanism in the joints");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_ACTIONS_CONTACT_FWDDYN_HPP_
