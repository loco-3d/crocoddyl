///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, IRI: CSIC-UPC
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/integrator/rk4.hpp"

namespace crocoddyl {
namespace python {

void exposeIntegratedActionRK4() {
  bp::register_ptr_to_python<boost::shared_ptr<IntegratedActionModelRK4> >();

  bp::class_<IntegratedActionModelRK4, bp::bases<ActionModelAbstract> >(
      "IntegratedActionModelRK4",
      "RK4 integrator for differential action models.\n\n"
      "This class implements an RK4 integrator\n"
      "given a differential action model, i.e.:\n"
      "  [q+, v+] = State.integrate([q, v], dt / 6 (k0 + k1 + k2 + k3)) with \n"
      "k0 = f(x, u) \n"
      "k1 = f(x + dt / 2 * k0, u) \n"
      "k2 = f(x + dt / 2 * k1, u) \n"
      "k3 = f(x + dt * k2, u) \n",
      bp::init<boost::shared_ptr<DifferentialActionModelAbstract>, bp::optional<double, bool> >(
          bp::args("self", "diffModel", "stepTime", "withCostResidual"),
          "Initialize the RK4 integrator.\n\n"
          ":param diffModel: differential action model\n"
          ":param stepTime: step time (default 1e-3)\n"
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
          "Computes the derivatives of the integrated action model wrt state and control. \n\n"
          "This function builds a quadratic approximation of the\n"
          "action model (i.e. dynamical system and cost function).\n"
          "It assumes that calc has been run first.\n"
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
      .add_property("dt", bp::make_function(&IntegratedActionModelRK4::get_dt), &IntegratedActionModelRK4::set_dt,
                    "step time");

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
      .add_property(
          "integral",
          bp::make_getter(&IntegratedActionDataRK4::integral, bp::return_value_policy<bp::return_by_value>()),
          "List with the RK4 terms related to the cost")
      .add_property("ki", bp::make_getter(&IntegratedActionDataRK4::ki, bp::return_internal_reference<>()),
                    "List with the RK4 terms related to system dynamics")
      .add_property("y", bp::make_getter(&IntegratedActionDataRK4::y, bp::return_internal_reference<>()),
                    "List with the states where f is evaluated in the RK4 integration scheme")
      .add_property("dx", bp::make_getter(&IntegratedActionDataRK4::dx, bp::return_internal_reference<>()),
                    "state rate.")
      .add_property("dki_dx", bp::make_getter(&IntegratedActionDataRK4::dki_dx, bp::return_internal_reference<>()),
                    "List with the partial derivatives of dynamics with respect to to the state of the RK4 "
                    "integration method. d(x+dx)/dx")
      .add_property("dki_du", bp::make_getter(&IntegratedActionDataRK4::dki_du, bp::return_internal_reference<>()),
                    "List with the partial derivatives of dynamics with respect to to the control of the RK4 "
                    "integration method. d(x+dx)/du")
      .add_property("dki_dp", bp::make_getter(&IntegratedActionDataRK4::dki_dp, bp::return_internal_reference<>()),
                    "List with the partial derivatives of dynamics with respect to to the control parameters of the RK4 "
                    "integration method. d(x+dx)/dp")
      .add_property("dli_dx", bp::make_getter(&IntegratedActionDataRK4::dli_dx, bp::return_internal_reference<>()),
                    "List with the partial derivatives of the cost with respect to to the state of the RK4 "
                    "integration method.")
      .add_property("dli_du", bp::make_getter(&IntegratedActionDataRK4::dli_du, bp::return_internal_reference<>()),
                    "List with the partial derivatives of the cost with respect to to the control of the RK4 "
                    "integration method.")
      .add_property("ddli_ddx", bp::make_getter(&IntegratedActionDataRK4::ddli_ddx, bp::return_internal_reference<>()),
                    "List with the second partial derivatives of the cost with respect to to the state of the RK4 "
                    "integration method.")
      .add_property("ddli_ddu", bp::make_getter(&IntegratedActionDataRK4::ddli_ddu, bp::return_internal_reference<>()),
                    "List with the second partial derivatives of the cost with respect to to the control of the RK4 "
                    "integration method.")
      .add_property("ddli_dxdu",
                    bp::make_getter(&IntegratedActionDataRK4::ddli_dxdu, bp::return_internal_reference<>()),
                    "List with the second partial derivatives of the cost with respect to to the state and the "
                    "control of the RK4 "
                    "integration method.");
}

}  // namespace python
}  // namespace crocoddyl
