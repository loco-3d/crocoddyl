///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh, University of Pisa
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/multibody/actions/contact-invdyn.hpp"

namespace crocoddyl {
namespace python {

void exposeDifferentialActionContactInvDynamics() {
  bp::scope().attr("yes") = 1;
  bp::scope().attr("no") = 0;
  {
    bp::register_ptr_to_python<boost::shared_ptr<DifferentialActionModelContactInvDynamics> >();
    bp::scope model_outer =
        bp::class_<DifferentialActionModelContactInvDynamics, bp::bases<DifferentialActionModelAbstract> >(
            "DifferentialActionModelContactInvDynamics",
            "Differential action model for inverse dynamics in multibody systems with contacts.\n\n"
            "This class implements a the dynamics using Recursive Newton Euler Algorithm (RNEA),\n"
            "On the other hand, the stack of cost and constraint functions are implemented in\n"
            "ConstraintModelManager() and CostModelSum(), respectively.",
            bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActuationModelAbstract>,
                    boost::shared_ptr<ContactModelMultiple>,boost::shared_ptr<CostModelSum>, 
                    bp::optional<boost::shared_ptr<ConstraintModelManager> > >(
                bp::args("self", "state", "actuation", "costs", "constraints"),
                "Initialize the inverse-dynamics action model for system with contact.\n\n"
                "It describes the kinematic evolution of the multibody system with any contact,\n"
                "and imposes an inverse-dynamics (equality) constraint. Additionally, it computes\n"
                "the cost and extra constraint values associated to this state and control pair.\n"
                "Note that the name `rnea` in the ConstraintModelManager is reserved to store\n"
                "the inverse-dynamics constraint\n."
                ":param state: multibody state\n"
                ":param actuation: abstract actuation model\n"
                ":param contacts: stack of contact model\n"
                ":param costs: stack of cost functions\n"
                ":param constraints: stack of constraint functions"))
            .def<void (DifferentialActionModelContactInvDynamics::*)(
                const boost::shared_ptr<DifferentialActionDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&,
                const Eigen::Ref<const Eigen::VectorXd>&)>("calc", &DifferentialActionModelContactInvDynamics::calc,
                                                           bp::args("self", "data", "x", "u"),
                                                           "Compute the next state and cost value.\n\n"
                                                           ":param data: inverse-dynamics action data\n"
                                                           ":param x: time-continuous state vector\n"
                                                           ":param u: time-continuous control input")
            .def<void (DifferentialActionModelContactInvDynamics::*)(
                const boost::shared_ptr<DifferentialActionDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&)>(
                "calc", &DifferentialActionModelAbstract::calc, bp::args("self", "data", "x"))
            .def<void (DifferentialActionModelContactInvDynamics::*)(
                const boost::shared_ptr<DifferentialActionDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&,
                const Eigen::Ref<const Eigen::VectorXd>&)>(
                "calcDiff", &DifferentialActionModelContactInvDynamics::calcDiff, bp::args("self", "data", "x", "u"),
                "Compute the derivatives of the differential multibody system and\n"
                "its cost functions.\n\n"
                "It computes the partial derivatives of the differential multibody system and the\n"
                "cost function. It assumes that calc has been run first.\n"
                "This function builds a quadratic approximation of the\n"
                "action model (i.e. dynamical system and cost function).\n"
                ":param data: inverse-dynamics action data\n"
                ":param x: time-continuous state vector\n"
                ":param u: time-continuous control input")
            .def<void (DifferentialActionModelContactInvDynamics::*)(
                const boost::shared_ptr<DifferentialActionDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&)>(
                "calcDiff", &DifferentialActionModelAbstract::calcDiff, bp::args("self", "data", "x"))
            .def("createData", &DifferentialActionModelContactInvDynamics::createData, bp::args("self"),
                 "Create the contact inverse-dynamics differential action data.")
            .add_property("actuation",
                          bp::make_function(&DifferentialActionModelContactInvDynamics::get_actuation,
                                            bp::return_value_policy<bp::return_by_value>()),
                          "actuation model")
            .add_property("costs",
                          bp::make_function(&DifferentialActionModelContactInvDynamics::get_costs,
                                            bp::return_value_policy<bp::return_by_value>()),
                          "total cost model")
            .add_property("constraints",
                          bp::make_function(&DifferentialActionModelContactInvDynamics::get_constraints,
                                            bp::return_value_policy<bp::return_by_value>()),
                          "entire constraint model");

    bp::register_ptr_to_python<boost::shared_ptr<DifferentialActionModelContactInvDynamics::ResidualModelRnea> >();

    bp::class_<DifferentialActionModelContactInvDynamics::ResidualModelRnea, bp::bases<ResidualModelAbstract> >(
        "ResidualModelRnea",
        "This residual function is defined as r = tau - RNEA, where\n"
        "tau is extracted from the control vector and RNEA evaluates the joint torque using\n"
        "q, q_dot, acc, f_ext values.",
        bp::init<boost::shared_ptr<StateMultibody>, std::size_t>(bp::args("self", "state", "nu"),
                                                                 "Initialize the RNEA residual model.\n\n"
                                                                 ":param nu: dimension of control vector"))
        .def<void (DifferentialActionModelContactInvDynamics::ResidualModelRnea::*)(
            const boost::shared_ptr<ResidualDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&,
            const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calc", &DifferentialActionModelContactInvDynamics::ResidualModelRnea::calc,
            bp::args("self", "data", "x", "u"),
            "Compute the RNEA residual.\n\n"
            ":param data: residual data\n"
            ":param x: state vector\n"
            ":param u: control input")
        .def<void (DifferentialActionModelContactInvDynamics::ResidualModelRnea::*)(
            const boost::shared_ptr<ResidualDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
        .def<void (DifferentialActionModelContactInvDynamics::ResidualModelRnea::*)(
            const boost::shared_ptr<ResidualDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&,
            const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calcDiff", &DifferentialActionModelContactInvDynamics::ResidualModelRnea::calcDiff,
            bp::args("self", "data", "x", "u"),
            "Compute the Jacobians of the RNEA residual.\n\n"
            "It assumes that calc has been run first.\n"
            ":param data: action data\n"
            ":param x: state vector\n"
            ":param u: control input\n")
        .def<void (DifferentialActionModelContactInvDynamics::ResidualModelRnea::*)(
            const boost::shared_ptr<ResidualDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calcDiff", &ResidualModelAbstract::calcDiff, bp::args("self", "data", "x"))
        .def("createData", &DifferentialActionModelContactInvDynamics::ResidualModelRnea::createData,
             bp::with_custodian_and_ward_postcall<0, 2>(), bp::args("self", "data"),
             "Create the RNEA residual data.\n\n"
             "Each residual model has its own data that needs to be allocated. This function\n"
             "returns the allocated data for the RNEA residual.\n"
             ":param data: shared data\n"
             ":return residual data.");
  
    bp::register_ptr_to_python<boost::shared_ptr<DifferentialActionModelContactInvDynamics::ResidualModelContact> >();

    bp::class_<DifferentialActionModelContactInvDynamics::ResidualModelContact, bp::bases<ResidualModelAbstract> >(
        "ResidualModelContact",
        "This residual function penalises contact acceleration and is defined as r = acc, where\n"
        "acc is contact acceleration.",
        boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id, const std::size_t nr,
                         const std::size_t nc, const std::size_t nu
        bp::init<boost::shared_ptr<StateMultibody>, std::size_t>(bp::args("self", "state", "id", "nr", "nc", "nu"),
                                                                 "Initialize the contact acceleration residual model.\n\n"
                                                                 ":param nu: dimension of control vector"))
        .def<void (DifferentialActionModelContactInvDynamics::ResidualModelContact::*)(
            const boost::shared_ptr<ResidualDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&,
            const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calc", &DifferentialActionModelContactInvDynamics::ResidualModelRnea::calc,
            bp::args("self", "data", "x", "u"),
            "Compute the contact acceleration residual.\n\n"
            ":param data: residual data\n"
            ":param x: state vector\n"
            ":param u: control input")
        .def<void (DifferentialActionModelContactInvDynamics::ResidualModelContact::*)(
            const boost::shared_ptr<ResidualDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
        .def<void (DifferentialActionModelContactInvDynamics::ResidualModelContact::*)(
            const boost::shared_ptr<ResidualDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&,
            const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calcDiff", &DifferentialActionModelContactInvDynamics::ResidualModelContact::calcDiff,
            bp::args("self", "data", "x", "u"),
            "Compute the Jacobians of the Contact acceleration residual.\n\n"
            "It assumes that calc has been run first.\n"
            ":param data: action data\n"
            ":param x: state vector\n"
            ":param u: control input\n")
        .def<void (DifferentialActionModelContactInvDynamics::ResidualModelContact::*)(
            const boost::shared_ptr<ResidualDataAbstract>&, const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calcDiff", &ResidualModelAbstract::calcDiff, bp::args("self", "data", "x"))
        .def("createData", &DifferentialActionModelContactInvDynamics::ResidualModelContact::createData,
             bp::with_custodian_and_ward_postcall<0, 2>(), bp::args("self", "data"),
             "Create the contact acceleration residual data.\n\n"
             "Each residual model has its own data that needs to be allocated. This function\n"
             "returns the allocated data for the contact acceleration residual.\n"
             ":param data: shared data\n"
             ":return residual data.");
  }

  bp::register_ptr_to_python<boost::shared_ptr<DifferentialActionDataContactInvDynamics> >();

  bp::scope data_outer =
      bp::class_<DifferentialActionDataContactInvDynamics, bp::bases<DifferentialActionDataAbstract> >(
          "DifferentialActionDataContactInvDynamics", "Action data for the inverse-dynamics for system with contact.",
          bp::init<DifferentialActionModelContactInvDynamics*>(bp::args("self", "model"),
                                                            "Create inverse-dynamics action data for system with contacts.\n\n"
                                                            ":param model: contact inverse-dynamics action model"))
          .add_property(
              "pinocchio",
              bp::make_getter(&DifferentialActionDataContactInvDynamics::pinocchio, bp::return_internal_reference<>()),
              "pinocchio data")
          .add_property(
              "multibody",
              bp::make_getter(&DifferentialActionDataContactInvDynamics::multibody, bp::return_internal_reference<>()),
              "multibody data")
          .add_property("costs",
                        bp::make_getter(&DifferentialActionDataContactInvDynamics::costs,
                                        bp::return_value_policy<bp::return_by_value>()),
                        "total cost data")
          .add_property("constraints",
                        bp::make_getter(&DifferentialActionDataContactInvDynamics::constraints,
                                        bp::return_value_policy<bp::return_by_value>()),
                        "constraint data");

  bp::register_ptr_to_python<boost::shared_ptr<DifferentialActionDataContactInvDynamics::ResidualDataRnea> >();

  bp::class_<DifferentialActionDataContactInvDynamics::ResidualDataRnea, bp::bases<ResidualDataAbstract> >(
      "ResidualDataRnea", "Data for RNEA residual.\n\n",
      bp::init<DifferentialActionModelContactInvDynamics::ResidualModelRnea*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create RNEA residual data.\n\n"
          ":param model: RNEA residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()]);

  bp::register_ptr_to_python<boost::shared_ptr<DifferentialActionDataContactInvDynamics::ResidualDataContact> >();

  bp::class_<DifferentialActionDataContactInvDynamics::ResidualDataContact, bp::bases<ResidualDataAbstract> >(
      "ResidualDataContact", "Data for contact acceleration residual.\n\n",
      bp::init<DifferentialActionModelContactInvDynamics::ResidualModelContact*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create contact acceleration residual data.\n\n"
          ":param model: contact residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()]);
}

}  // namespace python
}  // namespace crocoddyl
