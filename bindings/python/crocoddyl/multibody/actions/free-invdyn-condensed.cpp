///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2022, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/multibody/actions/free-invdyn-condensed.hpp"

namespace crocoddyl {
namespace python {

void exposeDifferentialActionFreeInvDynamicsCondensed() {
  bp::scope().attr("yes") = 1;
  bp::scope().attr("no") = 0;
  {
    bp::register_ptr_to_python<boost::shared_ptr<DifferentialActionModelFreeInvDynamicsCondensed>>();
    bp::scope model_outer =
        bp::class_<DifferentialActionModelFreeInvDynamicsCondensed, bp::bases<DifferentialActionModelAbstract>>(
            "DifferentialActionModelFreeInvDynamicsCondensed",
            "Differential action model for free inverse dynamics in multibody systems.\n\n"
            "This class implements the dynamics using Recursive Newton Euler Algorithm (RNEA) as an equality "
            "constraint.\n"
            "The stack of cost and constraint functions are implemented in\n"
            "ConstraintModelManager() and CostModelSum(), respectively.",
            bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActuationModelAbstract>,
                     boost::shared_ptr<CostModelSum>, bp::optional<boost::shared_ptr<ConstraintModelManager>>>(
                bp::args("self", "state", "actuation", "costs", "constraints"),
                "Initialize the free inverse-dynamics action model.\n\n"
                "It describes the kinematic evolution of the multibody system and computes the\n"
                "needed torques using inverse dynamics.\n"
                ":param state: multibody state\n"
                ":param actuation: abstract actuation model\n"
                ":param costs: stack of cost functions\n"
                ":param constraints: stack of constraint functions"))
            .def<void (DifferentialActionModelFreeInvDynamicsCondensed::*)(
                const boost::shared_ptr<DifferentialActionDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&,
                const Eigen::Ref<const Eigen::VectorXd>&)>(
                "calc", &DifferentialActionModelFreeInvDynamicsCondensed::calc, bp::args("self", "data", "x", "u"),
                "Compute the next state and cost value.\n\n"
                ":param data: free inverse-dynamics action data\n"
                ":param x: state point (dim. state.nx)\n"
                ":param u: control input (dim. nu)")
            .def<void (DifferentialActionModelFreeInvDynamicsCondensed::*)(
                const boost::shared_ptr<DifferentialActionDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&)>(
                "calc", &DifferentialActionModelAbstract::calc, bp::args("self", "data", "x"))
            .def<void (DifferentialActionModelFreeInvDynamicsCondensed::*)(
                const boost::shared_ptr<DifferentialActionDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&,
                const Eigen::Ref<const Eigen::VectorXd>&)>(
                "calcDiff", &DifferentialActionModelFreeInvDynamicsCondensed::calcDiff,
                bp::args("self", "data", "x", "u"),
                "Compute the derivatives of the differential multibody system (free of contact) and\n"
                "its cost functions.\n\n"
                "It computes the partial derivatives of the differential multibody system and the\n"
                "cost function. It assumes that calc has been run first.\n"
                "This function builds a quadratic approximation of the\n"
                "action model (i.e. dynamical system and cost function).\n"
                ":param data: free inverse-dynamics action data\n"
                ":param x: state point (dim. state.nx)\n"
                ":param u: control input (dim. nu)")
            .def<void (DifferentialActionModelFreeInvDynamicsCondensed::*)(
                const boost::shared_ptr<DifferentialActionDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&)>(
                "calcDiff", &DifferentialActionModelAbstract::calcDiff, bp::args("self", "data", "x"))
            .def("createData", &DifferentialActionModelFreeInvDynamicsCondensed::createData, bp::args("self"),
                 "Create the free inverse-dynamics differential action data.")
            .add_property("actuation",
                          bp::make_function(&DifferentialActionModelFreeInvDynamicsCondensed::get_actuation,
                                            bp::return_value_policy<bp::return_by_value>()),
                          "actuation model")
            .add_property("costs",
                          bp::make_function(&DifferentialActionModelFreeInvDynamicsCondensed::get_costs,
                                            bp::return_value_policy<bp::return_by_value>()),
                          "total cost model")
            .add_property("constraints",
                          bp::make_function(&DifferentialActionModelFreeInvDynamicsCondensed::get_constraints,
                                            bp::return_value_policy<bp::return_by_value>()),
                          "entire constraint model");

    bp::register_ptr_to_python<
        boost::shared_ptr<DifferentialActionModelFreeInvDynamicsCondensed::ResidualModelActuation>>();

    bp::class_<DifferentialActionModelFreeInvDynamicsCondensed::ResidualModelActuation,
               bp::bases<ResidualModelAbstract>>(
        "ResidualModelActuation",
        "This residual function enforces the torques of under-actuated joints (e.g., floating-base\n"
        "joints) to be zero. We compute these torques and their derivatives using RNEA inside \n"
        "DifferentialActionModelFreeInvDynamicsCondensed.",
        bp::init<boost::shared_ptr<StateMultibody>, std::size_t>(bp::args("self", "state", "nu"),
                                                                 "Initialize the actuation residual model.\n\n"
                                                                 ":param state: state description\n"
                                                                 ":param nu: dimension of the joint torques"))
        .def<void (DifferentialActionModelFreeInvDynamicsCondensed::ResidualModelActuation::*)(
            const boost::shared_ptr<ResidualDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&,
            const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calc", &DifferentialActionModelFreeInvDynamicsCondensed::ResidualModelActuation::calc,
            bp::args("self", "data", "x", "u"),
            "Compute the actuation residual.\n\n"
            ":param data: residual data\n"
            ":param x: state point (dim. state.nx)\n"
            ":param u: control input (dim. nu)")
        .def<void (DifferentialActionModelFreeInvDynamicsCondensed::ResidualModelActuation::*)(
            const boost::shared_ptr<ResidualDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
        .def<void (DifferentialActionModelFreeInvDynamicsCondensed::ResidualModelActuation::*)(
            const boost::shared_ptr<ResidualDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&,
            const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calcDiff", &DifferentialActionModelFreeInvDynamicsCondensed::ResidualModelActuation::calcDiff,
            bp::args("self", "data", "x", "u"),
            "Compute the Jacobians of the actuation residual.\n\n"
            "It assumes that calc has been run first.\n"
            ":param data: action data\n"
            ":param x: state point (dim. state.nx)\n"
            ":param u: control input (dim. nu)\n")
        .def<void (DifferentialActionModelFreeInvDynamicsCondensed::ResidualModelActuation::*)(
            const boost::shared_ptr<ResidualDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calcDiff", &ResidualModelAbstract::calcDiff, bp::args("self", "data", "x"))
        .def("createData", &DifferentialActionModelFreeInvDynamicsCondensed::ResidualModelActuation::createData,
             bp::with_custodian_and_ward_postcall<0, 2>(), bp::args("self", "data"),
             "Create the actuation residual data.\n\n"
             "Each residual model has its own data that needs to be allocated. This function\n"
             "returns the allocated data for the actuation residual.\n"
             ":param data: shared data\n"
             ":return residual data.");
  }

  bp::register_ptr_to_python<boost::shared_ptr<DifferentialActionDataFreeInvDynamicsCondensed>>();

  bp::scope data_outer =
      bp::class_<DifferentialActionDataFreeInvDynamicsCondensed, bp::bases<DifferentialActionDataAbstract>>(
          "DifferentialActionDataFreeInvDynamicsCondensed", "Action data for the free inverse-dynamics system.",
          bp::init<DifferentialActionModelFreeInvDynamicsCondensed*>(
              bp::args("self", "model"),
              "Create free inverse-dynamics action data.\n\n"
              ":param model: free inverse-dynamics action model"))
          .add_property("pinocchio",
                        bp::make_getter(&DifferentialActionDataFreeInvDynamicsCondensed::pinocchio,
                                        bp::return_internal_reference<>()),
                        "pinocchio data")
          .add_property("multibody",
                        bp::make_getter(&DifferentialActionDataFreeInvDynamicsCondensed::multibody,
                                        bp::return_internal_reference<>()),
                        "multibody data")
          .add_property("costs",
                        bp::make_getter(&DifferentialActionDataFreeInvDynamicsCondensed::costs,
                                        bp::return_value_policy<bp::return_by_value>()),
                        "total cost data")
          .add_property("constraints",
                        bp::make_getter(&DifferentialActionDataFreeInvDynamicsCondensed::constraints,
                                        bp::return_value_policy<bp::return_by_value>()),
                        "constraint data");

  bp::register_ptr_to_python<
      boost::shared_ptr<DifferentialActionDataFreeInvDynamicsCondensed::ResidualDataActuation>>();

  bp::class_<DifferentialActionDataFreeInvDynamicsCondensed::ResidualDataActuation, bp::bases<ResidualDataAbstract>>(
      "ResidualDataActuation", "Data for actuation residual.\n\n",
      bp::init<DifferentialActionModelFreeInvDynamicsCondensed::ResidualModelActuation*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create actuation residual data.\n\n"
          ":param model: actuation residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3>>()]);
}

}  // namespace python
}  // namespace crocoddyl
