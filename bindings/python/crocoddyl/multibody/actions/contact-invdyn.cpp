///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2023, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/multibody/actions/contact-invdyn.hpp"

#include "python/crocoddyl/core/diff-action-base.hpp"
#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/utils/copyable.hpp"

namespace crocoddyl {
namespace python {

void exposeDifferentialActionContactInvDynamics() {
  bp::scope().attr("yes") = 1;
  bp::scope().attr("no") = 0;
  {
    bp::register_ptr_to_python<
        std::shared_ptr<DifferentialActionModelContactInvDynamics> >();
    bp::scope model_outer =
        bp::class_<DifferentialActionModelContactInvDynamics,
                   bp::bases<DifferentialActionModelAbstract> >(
            "DifferentialActionModelContactInvDynamics",
            "Differential action model for inverse dynamics in multibody "
            "systems with contacts.\n\n"
            "This class implements forward kinematic with holonomic contact "
            "constraints (defined at the acceleration\n"
            "level) and inverse-dynamics computation using the Recursive "
            "Newton Euler Algorithm (RNEA)\n"
            "On the other hand, the stack of cost and constraint functions are "
            "implemented in\n"
            "ConstraintModelManager() and CostModelSum(), respectively.",
            bp::init<std::shared_ptr<StateMultibody>,
                     std::shared_ptr<ActuationModelAbstract>,
                     std::shared_ptr<ContactModelMultiple>,
                     std::shared_ptr<CostModelSum>,
                     bp::optional<std::shared_ptr<ConstraintModelManager> > >(
                bp::args("self", "state", "actuation", "contacts", "costs",
                         "constraints"),
                "Initialize the inverse-dynamics action model for system with "
                "contact.\n\n"
                "It describes the kinematic evolution of the multibody system "
                "with contacts,\n"
                "and computes the needed torques using inverse-dynamics.\n."
                ":param state: multibody state\n"
                ":param actuation: abstract actuation model\n"
                ":param contacts: stack of contact model\n"
                ":param costs: stack of cost functions\n"
                ":param constraints: stack of constraint functions"))
            .def<void (DifferentialActionModelContactInvDynamics::*)(
                const std::shared_ptr<DifferentialActionDataAbstract>&,
                const Eigen::Ref<const Eigen::VectorXd>&,
                const Eigen::Ref<const Eigen::VectorXd>&)>(
                "calc", &DifferentialActionModelContactInvDynamics::calc,
                bp::args("self", "data", "x", "u"),
                "Compute the next state, cost value and constraints.\n\n"
                ":param data: contact inverse-dynamics data\n"
                ":param x: state point (dim. state.nx)\n"
                ":param u: control input (dim. nu)")
            .def<void (DifferentialActionModelContactInvDynamics::*)(
                const std::shared_ptr<DifferentialActionDataAbstract>&,
                const Eigen::Ref<const Eigen::VectorXd>&)>(
                "calc", &DifferentialActionModelAbstract::calc,
                bp::args("self", "data", "x"))
            .def<void (DifferentialActionModelContactInvDynamics::*)(
                const std::shared_ptr<DifferentialActionDataAbstract>&,
                const Eigen::Ref<const Eigen::VectorXd>&,
                const Eigen::Ref<const Eigen::VectorXd>&)>(
                "calcDiff",
                &DifferentialActionModelContactInvDynamics::calcDiff,
                bp::args("self", "data", "x", "u"),
                "Compute the derivatives of the differential multibody system, "
                "and its cost and constraint "
                "functions.\n\n"
                "It computes the partial derivatives of the differential "
                "multibody system, the cost and constraint\n"
                "functions. It assumes that calc has been run first. This "
                "function builds a quadratic approximation\n"
                "of the action model (i.e., dynamical system, cost and "
                "constraint functions).\n"
                ":param data: contact inverse-dynamics data\n"
                ":param x: state point (dim. state.nx)\n"
                ":param u: control input (dim. nu)")
            .def<void (DifferentialActionModelContactInvDynamics::*)(
                const std::shared_ptr<DifferentialActionDataAbstract>&,
                const Eigen::Ref<const Eigen::VectorXd>&)>(
                "calcDiff", &DifferentialActionModelAbstract::calcDiff,
                bp::args("self", "data", "x"))
            .def(
                "createData",
                &DifferentialActionModelContactInvDynamics::createData,
                bp::args("self"),
                "Create the contact inverse-dynamics differential action data.")
            .add_property(
                "actuation",
                bp::make_function(
                    &DifferentialActionModelContactInvDynamics::get_actuation,
                    bp::return_value_policy<bp::return_by_value>()),
                "actuation model")
            .add_property(
                "contacts",
                bp::make_function(
                    &DifferentialActionModelContactInvDynamics::get_contacts,
                    bp::return_value_policy<bp::return_by_value>()),
                "multiple contact model")
            .add_property(
                "costs",
                bp::make_function(
                    &DifferentialActionModelContactInvDynamics::get_costs,
                    bp::return_value_policy<bp::return_by_value>()),
                "total cost model")
            .add_property(
                "constraints",
                bp::make_function(
                    &DifferentialActionModelContactInvDynamics::get_constraints,
                    bp::return_value_policy<bp::return_by_value>()),
                "constraint model manager")
            .def(CopyableVisitor<DifferentialActionModelContactInvDynamics>());

    bp::register_ptr_to_python<std::shared_ptr<
        DifferentialActionModelContactInvDynamics::ResidualModelActuation> >();

    bp::class_<
        DifferentialActionModelContactInvDynamics::ResidualModelActuation,
        bp::bases<ResidualModelAbstract> >(
        "ResidualModelActuation",
        "This residual function enforces the torques of under-actuated joints "
        "(e.g., floating-base\n"
        "joints) to be zero. We compute these torques and their derivatives "
        "using RNEA inside \n"
        "DifferentialActionModelContactInvDynamics.",
        bp::init<std::shared_ptr<StateMultibody>, std::size_t, std::size_t>(
            bp::args("self", "state", "nu", "nc"),
            "Initialize the actuation residual model.\n\n"
            ":param nu: dimension of control vector\n"
            ":param nc: number of the contacts"))
        .def<void (DifferentialActionModelContactInvDynamics::
                       ResidualModelActuation::*)(
            const std::shared_ptr<ResidualDataAbstract>&,
            const Eigen::Ref<const Eigen::VectorXd>&,
            const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calc",
            &DifferentialActionModelContactInvDynamics::ResidualModelActuation::
                calc,
            bp::args("self", "data", "x", "u"),
            "Compute the actuation residual.\n\n"
            ":param data: residual data\n"
            ":param x: state point (dim. state.nx)\n"
            ":param u: control input (dim. nu)")
        .def<void (DifferentialActionModelContactInvDynamics::
                       ResidualModelActuation::*)(
            const std::shared_ptr<ResidualDataAbstract>&,
            const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
        .def<void (DifferentialActionModelContactInvDynamics::
                       ResidualModelActuation::*)(
            const std::shared_ptr<ResidualDataAbstract>&,
            const Eigen::Ref<const Eigen::VectorXd>&,
            const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calcDiff",
            &DifferentialActionModelContactInvDynamics::ResidualModelActuation::
                calcDiff,
            bp::args("self", "data", "x", "u"),
            "Compute the Jacobians of the actuation residual.\n\n"
            "It assumes that calc has been run first.\n"
            ":param data: action data\n"
            ":param x: state point (dim. state.nx)\n"
            ":param u: control input (dim. nu)\n")
        .def<void (DifferentialActionModelContactInvDynamics::
                       ResidualModelActuation::*)(
            const std::shared_ptr<ResidualDataAbstract>&,
            const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calcDiff", &ResidualModelAbstract::calcDiff,
            bp::args("self", "data", "x"))
        .def("createData",
             &DifferentialActionModelContactInvDynamics::
                 ResidualModelActuation::createData,
             bp::with_custodian_and_ward_postcall<0, 2>(),
             bp::args("self", "data"),
             "Create the actuation residual data.\n\n"
             "Each residual model has its own data that needs to be allocated. "
             "This function\n"
             "returns the allocated data for the actuation residual.\n"
             ":param data: shared data\n"
             ":return residual data.")
        .def(CopyableVisitor<DifferentialActionModelContactInvDynamics::
                                 ResidualModelActuation>());

    bp::register_ptr_to_python<std::shared_ptr<
        DifferentialActionModelContactInvDynamics::ResidualModelContact> >();

    bp::class_<DifferentialActionModelContactInvDynamics::ResidualModelContact,
               bp::bases<ResidualModelAbstract> >(
        "ResidualModelContact",
        "This residual function for the contact acceleration, i.e., r = a0, "
        "where a0 is the desired\n"
        "contact acceleration which also considers the Baumgarte "
        "stabilization.",
        bp::init<std::shared_ptr<StateMultibody>, pinocchio::FrameIndex,
                 std::size_t, std::size_t>(
            bp::args("self", "state", "id", "nr", "nc"),
            "Initialize the contact-acceleration residual model.\n\n"
            ":param id: contact id\n"
            ":param nr: dimension of contact residual\n"
            ":param nc: dimension of contact vector"))
        .def<void (
            DifferentialActionModelContactInvDynamics::ResidualModelContact::*)(
            const std::shared_ptr<ResidualDataAbstract>&,
            const Eigen::Ref<const Eigen::VectorXd>&,
            const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calc",
            &DifferentialActionModelContactInvDynamics::ResidualModelContact::
                calc,
            bp::args("self", "data", "x", "u"),
            "Compute the contact-acceleration residual.\n\n"
            ":param data: residual data\n"
            ":param x: state vector\n"
            ":param u: control input")
        .def<void (
            DifferentialActionModelContactInvDynamics::ResidualModelContact::*)(
            const std::shared_ptr<ResidualDataAbstract>&,
            const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
        .def<void (
            DifferentialActionModelContactInvDynamics::ResidualModelContact::*)(
            const std::shared_ptr<ResidualDataAbstract>&,
            const Eigen::Ref<const Eigen::VectorXd>&,
            const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calcDiff",
            &DifferentialActionModelContactInvDynamics::ResidualModelContact::
                calcDiff,
            bp::args("self", "data", "x", "u"),
            "Compute the Jacobians of the contact-acceleration residual.\n\n"
            "It assumes that calc has been run first.\n"
            ":param data: action data\n"
            ":param x: state vector\n"
            ":param u: control input\n")
        .def<void (
            DifferentialActionModelContactInvDynamics::ResidualModelContact::*)(
            const std::shared_ptr<ResidualDataAbstract>&,
            const Eigen::Ref<const Eigen::VectorXd>&)>(
            "calcDiff", &ResidualModelAbstract::calcDiff,
            bp::args("self", "data", "x"))
        .def("createData",
             &DifferentialActionModelContactInvDynamics::ResidualModelContact::
                 createData,
             bp::with_custodian_and_ward_postcall<0, 2>(),
             bp::args("self", "data"),
             "Create the contact-acceleration residual data.\n\n"
             "Each residual model has its own data that needs to be allocated. "
             "This function\n"
             "returns the allocated data for the contact-acceleration "
             "residual.\n"
             ":param data: shared data\n"
             ":return residual data.")
        .def(CopyableVisitor<DifferentialActionModelContactInvDynamics::
                                 ResidualModelContact>());
  }

  bp::register_ptr_to_python<
      std::shared_ptr<DifferentialActionDataContactInvDynamics> >();

  bp::scope data_outer =
      bp::class_<DifferentialActionDataContactInvDynamics,
                 bp::bases<DifferentialActionDataAbstract> >(
          "DifferentialActionDataContactInvDynamics",
          "Differential action data for the inverse-dynamics for system with "
          "contact.",
          bp::init<DifferentialActionModelContactInvDynamics*>(
              bp::args("self", "model"),
              "Create inverse-dynamics action data for system with "
              "contacts.\n\n"
              ":param model: contact inverse-dynamics action model"))
          .add_property(
              "pinocchio",
              bp::make_getter(
                  &DifferentialActionDataContactInvDynamics::pinocchio,
                  bp::return_internal_reference<>()),
              "pinocchio data")
          .add_property(
              "multibody",
              bp::make_getter(
                  &DifferentialActionDataContactInvDynamics::multibody,
                  bp::return_internal_reference<>()),
              "multibody data")
          .add_property(
              "costs",
              bp::make_getter(&DifferentialActionDataContactInvDynamics::costs,
                              bp::return_value_policy<bp::return_by_value>()),
              "total cost data")
          .add_property(
              "constraints",
              bp::make_getter(
                  &DifferentialActionDataContactInvDynamics::constraints,
                  bp::return_value_policy<bp::return_by_value>()),
              "constraint data")
          .def(CopyableVisitor<DifferentialActionDataContactInvDynamics>());

  bp::register_ptr_to_python<std::shared_ptr<
      DifferentialActionDataContactInvDynamics::ResidualDataActuation> >();

  bp::class_<DifferentialActionDataContactInvDynamics::ResidualDataActuation,
             bp::bases<ResidualDataAbstract> >(
      "ResidualDataActuation", "Data for actuation residual.\n\n",
      bp::init<
          DifferentialActionModelContactInvDynamics::ResidualModelActuation*,
          DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create actuation residual data.\n\n"
          ":param model: actuation residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<
          1, 2, bp::with_custodian_and_ward<1, 3> >()]);

  bp::register_ptr_to_python<std::shared_ptr<
      DifferentialActionDataContactInvDynamics::ResidualDataContact> >();

  bp::class_<DifferentialActionDataContactInvDynamics::ResidualDataContact,
             bp::bases<ResidualDataAbstract> >(
      "ResidualDataContact", "Data for contact acceleration residual.\n\n",
      bp::init<DifferentialActionModelContactInvDynamics::ResidualModelContact*,
               DataCollectorAbstract*, std::size_t>(
          bp::args("self", "model", "data", "id"),
          "Create contact-acceleration residual data.\n\n"
          ":param model: contact-acceleration residual model\n"
          ":param data: shared data\n"
          ":param id: contact id")[bp::with_custodian_and_ward<
          1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property(
          "contact",
          bp::make_getter(&DifferentialActionDataContactInvDynamics::
                              ResidualDataContact::contact,
                          bp::return_value_policy<bp::return_by_value>()),
          "contact data associated with the current residual")
      .def(CopyableVisitor<
           DifferentialActionDataContactInvDynamics::ResidualDataContact>());
}

}  // namespace python
}  // namespace crocoddyl
