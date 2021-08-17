///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/multibody/actions/free-invdyn.hpp"

namespace crocoddyl {
namespace python {

void exposeDifferentialActionFreeInvDynamics() {
  bp::scope().attr("yes") = 1;
  bp::scope().attr("no") = 0;
  {
    bp::register_ptr_to_python<boost::shared_ptr<DifferentialActionModelFreeInvDynamics> >();
    bp::scope model_outer =
        bp::class_<DifferentialActionModelFreeInvDynamics, bp::bases<DifferentialActionModelAbstract> >(
            "DifferentialActionModelFreeInvDynamics",
            "Differential action model for free inverse dynamics in multibody systems.\n\n"
            "This class implements  the dynamics using Recursive Newton Euler Algorithm (RNEA) as an equality "
            "constraint.\n"
            "The stack of cost and constraint functions are implemented in\n"
            "ConstraintModelManager() and CostModelSum(), respectively.",
            bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActuationModelAbstract>,
                     boost::shared_ptr<CostModelSum>, bp::optional<boost::shared_ptr<ConstraintModelManager> > >(
                bp::args("self", "state", "actuation", "costs", "constraints"),
                "Initialize the free inverse-dynamics action model.\n\n"
                "It describes the kinematic evolution of the multibody system without any contact,\n"
                "and imposes an inverse-dynamics (equality) constraint. Additionally, it computes\n"
                "the cost and extra constraint values associated to this state and control pair.\n"
                "Note that the name `rnea` in the ConstraintModelManager is reserved to store\n"
                "the inverse-dynamics constraint\n."
                ":param state: multibody state\n"
                ":param actuation: abstract actuation model\n"
                ":param costs: stack of cost functions\n"
                ":param constraints: stack of constraint functions"))
            .def<void (DifferentialActionModelFreeInvDynamics::*)(
                const boost::shared_ptr<DifferentialActionDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&,
                const Eigen::Ref<const Eigen::VectorXd>&)>("calc", &DifferentialActionModelFreeInvDynamics::calc,
                                                           bp::args("self", "data", "x", "u"),
                                                           "Compute the next state and cost value.\n\n"
                                                           ":param data: free inverse-dynamics action data\n"
                                                           ":param x: time-continuous state vector\n"
                                                           ":param u: time-continuous control input")
            .def<void (DifferentialActionModelFreeInvDynamics::*)(
                const boost::shared_ptr<DifferentialActionDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&)>(
                "calc", &DifferentialActionModelAbstract::calc, bp::args("self", "data", "x"))
            .def<void (DifferentialActionModelFreeInvDynamics::*)(
                const boost::shared_ptr<DifferentialActionDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&,
                const Eigen::Ref<const Eigen::VectorXd>&)>(
                "calcDiff", &DifferentialActionModelFreeInvDynamics::calcDiff, bp::args("self", "data", "x", "u"),
                "Compute the derivatives of the differential multibody system (free of contact) and\n"
                "its cost functions.\n\n"
                "It computes the partial derivatives of the differential multibody system and the\n"
                "cost function. It assumes that calc has been run first.\n"
                "This function builds a quadratic approximation of the\n"
                "action model (i.e. dynamical system and cost function).\n"
                ":param data: free inverse-dynamics action data\n"
                ":param x: time-continuous state vector\n"
                ":param u: time-continuous control input")
            .def<void (DifferentialActionModelFreeInvDynamics::*)(
                const boost::shared_ptr<DifferentialActionDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&)>(
                "calcDiff", &DifferentialActionModelAbstract::calcDiff, bp::args("self", "data", "x"))
            .def("createData", &DifferentialActionModelFreeInvDynamics::createData, bp::args("self"),
                 "Create the free inverse-dynamics differential action data.")
            .add_property("actuation",
                          bp::make_function(&DifferentialActionModelFreeInvDynamics::get_actuation,
                                            bp::return_value_policy<bp::return_by_value>()),
                          "actuation model")
            .add_property("costs",
                          bp::make_function(&DifferentialActionModelFreeInvDynamics::get_costs,
                                            bp::return_value_policy<bp::return_by_value>()),
                          "total cost model")
            .add_property("constraints",
                          bp::make_function(&DifferentialActionModelFreeInvDynamics::get_constraints,
                                            bp::return_value_policy<bp::return_by_value>()),
                          "entire constraint model");

    bp::register_ptr_to_python<boost::shared_ptr<DifferentialActionModelFreeInvDynamics::ResidualModelRnea> >();

    bp::class_<DifferentialActionModelFreeInvDynamics::ResidualModelRnea, bp::bases<ResidualModelAbstract> >(
        "ResidualModelRnea",
        "This residual function is defined as r = tau - RNEA, where\n"
        "tau is extracted from the control vector and RNEA evaluates the joint torque using\n"
        "q, q_dot, acc values.",
        bp::init<boost::shared_ptr<StateMultibody>, std::size_t>(bp::args("self", "state", "nu"),
                                                                 "Initialize the RNEA residual model.\n\n"
                                                                 ":param nu: dimension of control vector"))
        .def<void (DifferentialActionModelFreeInvDynamics::ResidualModelRnea::*)(
            const boost::shared_ptr<ResidualDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&,
            const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calc", &DifferentialActionModelFreeInvDynamics::ResidualModelRnea::calc,
            bp::args("self", "data", "x", "u"),
            "Compute the RNEA residual.\n\n"
            ":param data: residual data\n"
            ":param x: state vector\n"
            ":param u: control input")
        .def<void (DifferentialActionModelFreeInvDynamics::ResidualModelRnea::*)(
            const boost::shared_ptr<ResidualDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
        .def<void (DifferentialActionModelFreeInvDynamics::ResidualModelRnea::*)(
            const boost::shared_ptr<ResidualDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&,
            const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calcDiff", &DifferentialActionModelFreeInvDynamics::ResidualModelRnea::calcDiff,
            bp::args("self", "data", "x", "u"),
            "Compute the Jacobians of the RNEA residual.\n\n"
            "It assumes that calc has been run first.\n"
            ":param data: action data\n"
            ":param x: state vector\n"
            ":param u: control input\n")
        .def<void (DifferentialActionModelFreeInvDynamics::ResidualModelRnea::*)(
            const boost::shared_ptr<ResidualDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calcDiff", &ResidualModelAbstract::calcDiff, bp::args("self", "data", "x"))
        .def("createData", &DifferentialActionModelFreeInvDynamics::ResidualModelRnea::createData,
             bp::with_custodian_and_ward_postcall<0, 2>(), bp::args("self", "data"),
             "Create the RNEA residual data.\n\n"
             "Each residual model has its own data that needs to be allocated. This function\n"
             "returns the allocated data for the RNEA residual.\n"
             ":param data: shared data\n"
             ":return residual data.");
  }

  bp::register_ptr_to_python<boost::shared_ptr<DifferentialActionDataFreeInvDynamics> >();

  bp::scope data_outer =
      bp::class_<DifferentialActionDataFreeInvDynamics, bp::bases<DifferentialActionDataAbstract> >(
          "DifferentialActionDataFreeInvDynamics", "Action data for the free inverse-dynamics system.",
          bp::init<DifferentialActionModelFreeInvDynamics*>(bp::args("self", "model"),
                                                            "Create free inverse-dynamics action data.\n\n"
                                                            ":param model: free inverse-dynamics action model"))
          .add_property(
              "pinocchio",
              bp::make_getter(&DifferentialActionDataFreeInvDynamics::pinocchio, bp::return_internal_reference<>()),
              "pinocchio data")
          .add_property(
              "multibody",
              bp::make_getter(&DifferentialActionDataFreeInvDynamics::multibody, bp::return_internal_reference<>()),
              "multibody data")
          .add_property("costs",
                        bp::make_getter(&DifferentialActionDataFreeInvDynamics::costs,
                                        bp::return_value_policy<bp::return_by_value>()),
                        "total cost data")
          .add_property("constraints",
                        bp::make_getter(&DifferentialActionDataFreeInvDynamics::constraints,
                                        bp::return_value_policy<bp::return_by_value>()),
                        "constraint data");

  bp::register_ptr_to_python<boost::shared_ptr<DifferentialActionDataFreeInvDynamics::ResidualDataRnea> >();

  bp::class_<DifferentialActionDataFreeInvDynamics::ResidualDataRnea, bp::bases<ResidualDataAbstract> >(
      "ResidualDataRnea", "Data for RNEA residual.\n\n",
      bp::init<DifferentialActionModelFreeInvDynamics::ResidualModelRnea*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create RNEA residual data.\n\n"
          ":param model: RNEA residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()]);
}

}  // namespace python
}  // namespace crocoddyl
