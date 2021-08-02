///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh, University of Oxford,
//                     University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/integ-action-base.hpp"
#include "python/crocoddyl/utils/printable.hpp"

namespace crocoddyl {
namespace python {

void exposeIntegratedActionAbstract() {
  bp::register_ptr_to_python<boost::shared_ptr<IntegratedActionModelAbstract> >();

  bp::class_<IntegratedActionModelAbstract_wrap, boost::noncopyable, bp::bases<ActionModelAbstract> >(
      "IntegratedActionModelAbstract",
      "Abstract class for integrated action models.\n\n"
      "In Crocoddyl, an integrated action model transforms a differential action model in a (discrete) action "
      "model.\n",
      bp::init<boost::shared_ptr<DifferentialActionModelAbstract>, bp::optional<double, bool> >(
          bp::args("self", "diffModel", "timeStep", "withCostResidual"),
          "Initialize the integrated-action model.\n\n"
          "You can also integrate autonomous systems (i.e., when diffModel.nu is equals to 0).\n"
          ":param diffModel: differential action model\n"
          ":param timestep: integration time step (default 1e-3)\n"
          ":param withCostResidual: includes the cost residuals and derivatives (default True)."))
      .def(bp::init<boost::shared_ptr<DifferentialActionModelAbstract>,
                    boost::shared_ptr<ControlParametrizationModelAbstract>, bp::optional<double, bool> >(
          bp::args("self", "diffModel", "control", "stepTime", "withCostResidual"),
          "Initialize the integrated-action integrator.\n\n"
          "You can also integrate autonomous systems (i.e., when diffModel.nu is equals to 0).\n"
          ":param model: differential action model\n"
          ":param control: the control parametrization\n"
          ":param stepTime: step time (default 1e-3)\n"
          ":param withCostResidual: includes the cost residuals and derivatives (default True)."))
      .def("calc", pure_virtual(&IntegratedActionModelAbstract_wrap::calc), bp::args("self", "data", "x", "u"),
           "Compute the next state and cost value.\n\n"
           "It describes the time-discrete evolution of our dynamical system\n"
           "in which we obtain the next state. Additionally it computes the\n"
           "cost value associated to this discrete state and control pair.\n"
           ":param data: integrated-action data\n"
           ":param x: time-discrete state vector\n"
           ":param u: time-discrete control input")
      .def<void (IntegratedActionModelAbstract::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                                   const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &IntegratedActionModelAbstract::calc, bp::args("self", "data", "x"))
      .def("calcDiff", pure_virtual(&IntegratedActionModelAbstract_wrap::calcDiff), bp::args("self", "data", "x", "u"),
           "Compute the derivatives of the dynamics and cost functions.\n\n"
           "It computes the partial derivatives of the dynamical system and the\n"
           "cost function. It assumes that calc has been run first.\n"
           "This function builds a quadratic approximation of the\n"
           "action model (i.e. linear dynamics and quadratic cost).\n"
           ":param data: integrated-action data\n"
           ":param x: time-discrete state vector\n"
           ":param u: time-discrete control input\n")
      .def<void (IntegratedActionModelAbstract::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                                   const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &IntegratedActionModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &IntegratedActionModelAbstract_wrap::createData,
           &IntegratedActionModelAbstract_wrap::default_createData, bp::args("self"),
           "Create the integrated-action data.\n\n"
           "Each integrated-action model (IAM) has its own data that needs to be allocated.\n"
           "This function returns the allocated data for a predefined IAM.\n"
           ":return integrated-action data.")
      .add_property("differential",
                    bp::make_function(&IntegratedActionModelAbstract_wrap::get_differential,
                                      bp::return_value_policy<bp::return_by_value>()),
                    "differential action model")
      .add_property("control",
                    bp::make_function(&IntegratedActionModelAbstract_wrap::get_control,
                                      bp::return_value_policy<bp::return_by_value>()),
                    "control parametrization model")
      .add_property(
          "dt",
          bp::make_function(&IntegratedActionModelAbstract::get_dt, bp::return_value_policy<bp::return_by_value>()),
          &IntegratedActionModelAbstract::set_dt, "step time")
      .def(PrintableVisitor<IntegratedActionModelAbstract>());

  bp::register_ptr_to_python<boost::shared_ptr<IntegratedActionDataAbstract> >();

  bp::class_<IntegratedActionDataAbstract, bp::bases<ActionDataAbstract> >(
      "IntegratedActionDataAbstract",
      "Abstract class for integrated-action data.\n\n"
      "In Crocoddyl, an action data contains all the required information for processing an\n"
      "user-defined action model. The action data typically is allocated onces by running\n"
      "model.createData() and contains the first- and second- order derivatives of the dynamics\n"
      "and cost function, respectively.",
      bp::init<IntegratedActionModelAbstract*>(
          bp::args("self", "model"),
          "Create common data shared between integrated-action models.\n\n"
          "The integrated-action data uses its model in order to first process it.\n"
          ":param model: integrated-action model"))
      .add_property(
          "control",
          bp::make_getter(&IntegratedActionDataAbstract::control, bp::return_value_policy<bp::return_by_value>()),
          "Data of the control parametrization model");
}

}  // namespace python
}  // namespace crocoddyl
