///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh, IRI: CSIC-UPC
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/integrator/rk4.hpp"

namespace crocoddyl {
namespace python {

void exposeIntegratedActionRK4() {
  bp::class_<IntegratedActionModelRK4, bp::bases<ActionModelAbstract> >(
      "IntegratedActionModelRK4",
      "RK4 integrator for differential action models.\n\n"
      "This class implements an RK4 integrator\n"
      "given a differential action model, i.e.:\n"
      "  [q+, v+] = State.integrate([q, v], dt / 6 (k0 + k1 + k2 + k3)) with \n"
      "k0 = f(x, u) \n"
      "k1 = f(x + dt / 2 * k1, u) \n"
      "k2 = f(x + dt / 2 * k2, u) \n"
      "k3 = f(x + dt * k2, u) \n",
      bp::init<boost::shared_ptr<DifferentialActionModelAbstract>, bp::optional<double, bool> >(
          bp::args("self", "diffModel", "stepTime", "withCostResidual"),
          "Initialize the RK4 integrator.\n\n"
          ":param diffModel: differential action model\n"
          ":param stepTime: step time\n"
          ":param withCostResidual: includes the cost residuals and derivatives."))
      .def<void (IntegratedActionModelRK4::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                              const Eigen::Ref<const Eigen::VectorXd>&,
                                              const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &IntegratedActionModelRK4::calc, bp::args("self", "data", "x", "u"),
          "Compute the time-discrete evolution of a differential action model.\n\n"
          "It describes the time-discrete evolution of action model.\n"
          ":param data: action data\n"
          ":param x: state vector\n"
          ":param u: control input")
      .def<void (IntegratedActionModelRK4::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                              const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ActionModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (IntegratedActionModelRK4::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                              const Eigen::Ref<const Eigen::VectorXd>&,
                                              const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &IntegratedActionModelRK4::calcDiff, bp::args("self", "data", "x", "u"),
          "Compute the time-discrete derivatives of the integrated action model.\n\n"
          "It computes the time-discrete partial derivatives of the integration action.\n"
          "It assumes that calc has been run first.\n"
          "This function builds a quadratic approximation of the\n"
          "action model (i.e. dynamical system and cost function).\n"
          ":param data: action data\n"
          ":param x: state vector\n"
          ":param u: control input\n")
      .def<void (IntegratedActionModelRK4::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                              const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ActionModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &IntegratedActionModelRK4::createData, bp::args("self"), "Create the RK4 integrator data.")
      .add_property("differential",
                    bp::make_function(&IntegratedActionModelRK4::get_differential,
                                      bp::return_value_policy<bp::return_by_value>()),
                    &IntegratedActionModelRK4::set_differential, "differential action model")
      .add_property(
          "dt", bp::make_function(&IntegratedActionModelRK4::get_dt, bp::return_value_policy<bp::return_by_value>()),
          &IntegratedActionModelRK4::set_dt, "step time");

  bp::register_ptr_to_python<boost::shared_ptr<IntegratedActionDataRK4> >();

  bp::class_<IntegratedActionDataRK4, bp::bases<ActionDataAbstract> >(
      "IntegratedActionDataRK4", "RK4 integrator data.",
      bp::init<IntegratedActionModelRK4*>(bp::args("self", "model"),
                                          "Create RK4 integrator data.\n\n"
                                          ":param model: RK4 integrator model"))
      .add_property(
          "differential",
          bp::make_getter(&IntegratedActionDataRK4::differential, bp::return_value_policy<bp::return_by_value>()),
          "differential action data")
      .add_property("dx", bp::make_getter(&IntegratedActionDataRK4::dx, bp::return_internal_reference<>()),
                    "state rate.");
}

}  // namespace python
}  // namespace crocoddyl