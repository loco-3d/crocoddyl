///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/actions/free-fwddyn.hpp"
#include "python/crocoddyl/core/diff-action-base.hpp"
#include "python/crocoddyl/multibody/multibody.hpp"

namespace crocoddyl {
namespace python {

void exposeDifferentialActionFreeFwdDynamics() {
  bp::class_<DifferentialActionModelFreeFwdDynamics,
             bp::bases<DifferentialActionModelAbstract>>(
      "DifferentialActionModelFreeFwdDynamics",
      "Differential action model for free forward dynamics in multibody "
      "systems.\n\n"
      "This class implements a the dynamics using Articulate Body Algorithm "
      "(ABA),\n"
      "or a custom implementation in case of system with armatures. If you "
      "want to\n"
      "include the armature, you need to use setArmature(). On the other hand, "
      "the\n"
      "stack of cost functions are implemented in CostModelSum().",
      bp::init<boost::shared_ptr<StateMultibody>,
               boost::shared_ptr<ActuationModelAbstract>,
               boost::shared_ptr<CostModelSum>>(
          bp::args("self", "state", "actuation", "costs"),
          "Initialize the free forward-dynamics action model.\n\n"
          ":param state: multibody state\n"
          ":param actuation: abstract actuation model\n"
          ":param costs: stack of cost functions"))
      .def<void (DifferentialActionModelFreeFwdDynamics::*)(
          const boost::shared_ptr<DifferentialActionDataAbstract> &,
          const Eigen::Ref<const Eigen::VectorXd> &,
          const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calc", &DifferentialActionModelFreeFwdDynamics::calc,
          bp::args("self", "data", "x", "u"),
          "Compute the next state and cost value.\n\n"
          "It describes the time-continuous evolution of the multibody system "
          "without any contact.\n"
          "Additionally it computes the cost value associated to this state "
          "and control pair.\n"
          ":param data: free forward-dynamics action data\n"
          ":param x: time-continuous state vector\n"
          ":param u: time-continuous control input")
      .def<void (DifferentialActionModelFreeFwdDynamics::*)(
          const boost::shared_ptr<DifferentialActionDataAbstract> &,
          const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calc", &DifferentialActionModelAbstract::calc,
          bp::args("self", "data", "x"))
      .def<void (DifferentialActionModelFreeFwdDynamics::*)(
          const boost::shared_ptr<DifferentialActionDataAbstract> &,
          const Eigen::Ref<const Eigen::VectorXd> &,
          const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calcDiff", &DifferentialActionModelFreeFwdDynamics::calcDiff,
          bp::args("self", "data", "x", "u"),
          "Compute the derivatives of the differential multibody system (free "
          "of contact) and\n"
          "its cost functions.\n\n"
          "It computes the partial derivatives of the differential multibody "
          "system and the\n"
          "cost function. It assumes that calc has been run first.\n"
          "This function builds a quadratic approximation of the\n"
          "action model (i.e. dynamical system and cost function).\n"
          ":param data: free forward-dynamics action data\n"
          ":param x: time-continuous state vector\n"
          ":param u: time-continuous control input\n")
      .def<void (DifferentialActionModelFreeFwdDynamics::*)(
          const boost::shared_ptr<DifferentialActionDataAbstract> &,
          const Eigen::Ref<const Eigen::VectorXd> &)>(
          "calcDiff", &DifferentialActionModelAbstract::calcDiff,
          bp::args("self", "data", "x"))
      .def("createData", &DifferentialActionModelFreeFwdDynamics::createData,
           bp::args("self"),
           "Create the free forward dynamics differential action data.")
      .add_property("pinocchio",
                    bp::make_function(
                        &DifferentialActionModelFreeFwdDynamics::get_pinocchio,
                        bp::return_internal_reference<>()),
                    "multibody model (i.e. pinocchio model)")
      .add_property("actuation",
                    bp::make_function(
                        &DifferentialActionModelFreeFwdDynamics::get_actuation,
                        bp::return_value_policy<bp::return_by_value>()),
                    "actuation model")
      .add_property(
          "costs",
          bp::make_function(&DifferentialActionModelFreeFwdDynamics::get_costs,
                            bp::return_value_policy<bp::return_by_value>()),
          "total cost model")
      .add_property("armature",
                    bp::make_function(
                        &DifferentialActionModelFreeFwdDynamics::get_armature,
                        bp::return_internal_reference<>()),
                    bp::make_function(
                        &DifferentialActionModelFreeFwdDynamics::set_armature),
                    "set an armature mechanism in the joints");

  bp::register_ptr_to_python<
      boost::shared_ptr<DifferentialActionDataFreeFwdDynamics>>();

  bp::class_<DifferentialActionDataFreeFwdDynamics,
             bp::bases<DifferentialActionDataAbstract>>(
      "DifferentialActionDataFreeFwdDynamics",
      "Action data for the free forward dynamics system.",
      bp::init<DifferentialActionModelFreeFwdDynamics *>(
          bp::args("self", "model"),
          "Create free forward-dynamics action data.\n\n"
          ":param model: free forward-dynamics action model"))
      .add_property(
          "pinocchio",
          bp::make_getter(&DifferentialActionDataFreeFwdDynamics::pinocchio,
                          bp::return_internal_reference<>()),
          "pinocchio data")
      .add_property(
          "multibody",
          bp::make_getter(&DifferentialActionDataFreeFwdDynamics::multibody,
                          bp::return_internal_reference<>()),
          "multibody data")
      .add_property(
          "costs",
          bp::make_getter(&DifferentialActionDataFreeFwdDynamics::costs,
                          bp::return_value_policy<bp::return_by_value>()),
          "total cost data")
      .add_property(
          "Minv",
          bp::make_getter(&DifferentialActionDataFreeFwdDynamics::Minv,
                          bp::return_internal_reference<>()),
          "inverse of the joint-space inertia matrix")
      .add_property(
          "u_drift",
          bp::make_getter(&DifferentialActionDataFreeFwdDynamics::u_drift,
                          bp::return_internal_reference<>()),
          "force-bias vector that accounts for control, Coriolis and "
          "gravitational effects");
}

} // namespace python
} // namespace crocoddyl
