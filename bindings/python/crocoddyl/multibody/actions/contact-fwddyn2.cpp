///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, LAAS-CNRS, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/multibody/actions/contact-fwddyn2.hpp"

namespace crocoddyl {
namespace python {

void exposeDifferentialActionContactFwdDynamics2() {
  typedef pinocchio::RigidContactModelTpl<double,0> RigidContactModel;
  bp::class_<DifferentialActionModelContactFwdDynamics2, bp::bases<DifferentialActionModelAbstract> >(
      "DifferentialActionModelContactFwdDynamics2",
      "Differential action model for contact forward dynamics in multibody systems.\n\n"
      "The contact is modelled as holonomic constraits in the contact frame. There\n"
      "is also a custom implementation in case of system with armatures. If you want to\n"
      "include the armature, you need to use setArmature(). On the other hand, the\n"
      "stack of cost functions are implemented in CostModelSum().",
      bp::init<boost::shared_ptr<StateMultibody>,
      boost::shared_ptr<ActuationModelAbstract>,
      PINOCCHIO_STD_VECTOR_WITH_EIGEN_ALLOCATOR(RigidContactModel),
      boost::shared_ptr<CostModelSum>,
      double >(
               bp::args("self", "state", "actuation", "contacts", "costs", "mu"),
          "Initialize the constrained forward-dynamics action model.\n\n"
          "The damping factor is needed when the contact Jacobian is not full-rank. Otherwise,\n"
          "a good damping factor could be 1e-12. In addition, if you have cost based on forces,\n"
          "you need to enable the computation of the force Jacobians (i.e. enable_force=True)."
          ":param state: multibody state\n"
          ":param actuation: abstract actuation model\n"
          ":param contacts: multiple contact model\n"
          ":param costs: stack of cost functions\n"
          ":param mu: damping parameter of type double\n"))
      .def<void (DifferentialActionModelContactFwdDynamics2::*)(
          const boost::shared_ptr<DifferentialActionDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &DifferentialActionModelContactFwdDynamics2::calc, bp::args("self", "data", "x", "u"),
          "Compute the next state and cost value.\n\n"
          "It describes the time-continuous evolution of the multibody system with contact. The\n"
          "contacts are modelled as holonomic constraints.\n"
          "Additionally it computes the cost value associated to this state and control pair.\n"
          ":param data: contact forward-dynamics action data\n"
          ":param x: time-continuous state vector\n"
          ":param u: time-continuous control input")
      .def<void (DifferentialActionModelContactFwdDynamics2::*)(
          const boost::shared_ptr<DifferentialActionDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &DifferentialActionModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (DifferentialActionModelContactFwdDynamics2::*)(
          const boost::shared_ptr<DifferentialActionDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &DifferentialActionModelContactFwdDynamics2::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the differential multibody system and its cost\n"
          "functions.\n\n"
          "It computes the partial derivatives of the differential multibody system and the\n"
          "cost function. It assumes that calc has been run first.\n"
          "This function builds a quadratic approximation of the\n"
          "action model (i.e. dynamical system and cost function).\n"
          ":param data: contact forward-dynamics action data\n"
          ":param x: time-continuous state vector\n"
          ":param u: time-continuous control input\n")
      .def<void (DifferentialActionModelContactFwdDynamics2::*)(
          const boost::shared_ptr<DifferentialActionDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &DifferentialActionModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &DifferentialActionModelContactFwdDynamics2::createData, bp::args("self"),
           "Create the contact forward dynamics differential action data.")
      .add_property("pinocchio",
                    bp::make_function(&DifferentialActionModelContactFwdDynamics2::get_pinocchio,
                                      bp::return_internal_reference<>()),
                    "multibody model (i.e. pinocchio model)")
      .add_property("actuation",
                    bp::make_function(&DifferentialActionModelContactFwdDynamics2::get_actuation,
                                      bp::return_value_policy<bp::return_by_value>()),
                    "actuation model")
      .add_property("contacts",
                    bp::make_function(&DifferentialActionModelContactFwdDynamics2::get_contacts,
                                      bp::return_value_policy<bp::return_by_value>()),
                    "multiple contact model")
      .add_property("costs",
                    bp::make_function(&DifferentialActionModelContactFwdDynamics2::get_costs,
                                      bp::return_value_policy<bp::return_by_value>()),
                    "total cost model")
      .add_property("armature",
                    bp::make_function(&DifferentialActionModelContactFwdDynamics2::get_armature,
                                      bp::return_value_policy<bp::return_by_value>()),
                    bp::make_function(&DifferentialActionModelContactFwdDynamics2::set_armature),
                    "set an armature mechanism in the joints")
      .add_property("mu",
                    bp::make_function(&DifferentialActionModelContactFwdDynamics2::get_mu,
                                      bp::return_value_policy<bp::return_by_value>()),
                    bp::make_function(&DifferentialActionModelContactFwdDynamics2::set_mu),
                    "set damping parmeter mu");

  bp::register_ptr_to_python<boost::shared_ptr<DifferentialActionDataContactFwdDynamics2> >();

  bp::class_<DifferentialActionDataContactFwdDynamics2, bp::bases<DifferentialActionDataAbstract> >(
      "DifferentialActionDataContactFwdDynamics2", "Action data for the contact forward dynamics system.",
      bp::init<DifferentialActionModelContactFwdDynamics2*>(bp::args("self", "model"),
                                                           "Create contact forward-dynamics action data.\n\n"
                                                           ":param model: contact forward-dynamics action model"))
      .add_property(
          "pinocchio",
          bp::make_getter(&DifferentialActionDataContactFwdDynamics2::pinocchio, bp::return_internal_reference<>()),
          "pinocchio data")
      .add_property(
          "multibody",
          bp::make_getter(&DifferentialActionDataContactFwdDynamics2::multibody, bp::return_internal_reference<>()),
          "multibody data")
      .add_property("costs",
                    bp::make_getter(&DifferentialActionDataContactFwdDynamics2::costs,
                                    bp::return_value_policy<bp::return_by_value>()),
                    "total cost data");
}

}  // namespace python
}  // namespace crocoddyl
