///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/integ-action-base.hpp"
#include "crocoddyl/core/integrator/rk3.hpp"

namespace crocoddyl {
namespace python {

void exposeIntegratedActionRK3() {
  bp::register_ptr_to_python<boost::shared_ptr<IntegratedActionModelRK3> >();

  bp::class_<IntegratedActionModelRK3, bp::bases<IntegratedActionModelAbstract, ActionModelAbstract> >(
      "IntegratedActionModelRK3",
      "RK3 integrator for differential action models.\n\n"
      "This class implements an RK3 integrator\n"
      "given a differential action model, i.e.:\n"
      "  [q+, v+] = State.integrate([q, v], dt / 4 (k0 + k1 + k2)) with \n"
      "k0 = f(x, u)\n"
      "k1 = f(x + dt / 3 * k0, u)\n"
      "k2 = f(x + 2 * dt / 3 * k1, u)",
      bp::init<boost::shared_ptr<DifferentialActionModelAbstract>, bp::optional<double, bool> >(
          bp::args("self", "diffModel", "stepTime", "withCostResidual"),
          "Initialize the RK3 integrator.\n\n"
          ":param diffModel: differential action model\n"
          ":param stepTime: step time (default 1e-3)\n"
          ":param withCostResidual: includes the cost residuals and derivatives."))
      .def(bp::init<boost::shared_ptr<DifferentialActionModelAbstract>,
                    boost::shared_ptr<ControlParametrizationModelAbstract>, bp::optional<double, bool> >(
          bp::args("self", "diffModel", "control", "stepTime", "withCostResidual"),
          "Initialize the RK3 integrator.\n\n"
          ":param diffModel: differential action model\n"
          ":param control: the control parametrization\n"
          ":param stepTime: step time (default 1e-3)\n"
          ":param withCostResidual: includes the cost residuals and derivatives."))
      .def<void (IntegratedActionModelRK3::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                              const Eigen::Ref<const Eigen::VectorXd>&,
                                              const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &IntegratedActionModelRK3::calc, bp::args("self", "data", "x", "u"),
          "Compute the time-discrete evolution of a differential action model.\n\n"
          "It describes the time-discrete evolution of action model.\n"
          ":param data: action data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (IntegratedActionModelRK3::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                              const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ActionModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (IntegratedActionModelRK3::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                              const Eigen::Ref<const Eigen::VectorXd>&,
                                              const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &IntegratedActionModelRK3::calcDiff, bp::args("self", "data", "x", "u"),
          "Computes the derivatives of the integrated action model wrt state and control. \n\n"
          "This function builds a quadratic approximation of the\n"
          "action model (i.e. dynamical system and cost function).\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)\n")
      .def<void (IntegratedActionModelRK3::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                              const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ActionModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &IntegratedActionModelRK3::createData, bp::args("self"), "Create the RK3 integrator data.");

  bp::register_ptr_to_python<boost::shared_ptr<IntegratedActionDataRK3> >();

  bp::class_<IntegratedActionDataRK3, bp::bases<IntegratedActionDataAbstract> >(
      "IntegratedActionDataRK3", "RK3 integrator data.",
      bp::init<IntegratedActionModelRK3*>(bp::args("self", "model"),
                                          "Create RK3 integrator data.\n\n"
                                          ":param model: RK3 integrator model"))
      .add_property(
          "differential",
          bp::make_getter(&IntegratedActionDataRK3::differential, bp::return_value_policy<bp::return_by_value>()),
          "differential action data")
      .add_property("control",
                    bp::make_getter(&IntegratedActionDataRK3::control, bp::return_value_policy<bp::return_by_value>()),
                    "list of control parametrization data")
      .add_property(
          "integral",
          bp::make_getter(&IntegratedActionDataRK3::integral, bp::return_value_policy<bp::return_by_value>()),
          "list of RK3 terms related to the cost")
      .add_property("dx", bp::make_getter(&IntegratedActionDataRK3::dx, bp::return_internal_reference<>()),
                    "state rate.")
      .add_property("ki", bp::make_getter(&IntegratedActionDataRK3::ki, bp::return_internal_reference<>()),
                    "list of RK3 terms related to system dynamics")
      .add_property("y", bp::make_getter(&IntegratedActionDataRK3::y, bp::return_internal_reference<>()),
                    "list of states where f is evaluated in the RK3 integration")
      .add_property("ws", bp::make_getter(&IntegratedActionDataRK3::ws, bp::return_internal_reference<>()),
                    "control inputs evaluated in the RK3 integration")
      .add_property("dki_dx", bp::make_getter(&IntegratedActionDataRK3::dki_dx, bp::return_internal_reference<>()),
                    "List with the partial derivatives of dynamics with respect to to the state of the RK3 "
                    "integration method. dki/dx")
      .add_property("dki_du", bp::make_getter(&IntegratedActionDataRK3::dki_du, bp::return_internal_reference<>()),
                    "list of partial derivatives of RK3 nodes with respect to the control parameters of the RK3 "
                    "integration. dki/du")
      .add_property("dki_du", bp::make_getter(&IntegratedActionDataRK3::dki_du, bp::return_internal_reference<>()),
                    "list of partial derivatives of RK3 nodes with respect to the control parameters of the RK3 "
                    "integration. dki/du")
      .add_property(
          "dyi_dx", bp::make_getter(&IntegratedActionDataRK3::dyi_dx, bp::return_internal_reference<>()),
          "list of partial derivatives of RK3 dynamics with respect to the state of the RK3 integrator. dyi/dx")
      .add_property("dyi_du", bp::make_getter(&IntegratedActionDataRK3::dyi_du, bp::return_internal_reference<>()),
                    "list of partial derivatives of RK3 dynamics with respect to the control parameters of the "
                    "RK3 integrator. dyi/du")
      .add_property("dli_dx", bp::make_getter(&IntegratedActionDataRK3::dli_dx, bp::return_internal_reference<>()),
                    "list of partial derivatives of the cost with respect to the state of the RK3 "
                    "integration. dli_dx")
      .add_property("dli_du", bp::make_getter(&IntegratedActionDataRK3::dli_du, bp::return_internal_reference<>()),
                    "list of partial derivatives of the cost with respect to the control input of the RK3 "
                    "integration. dli_du")
      .add_property("ddli_ddx", bp::make_getter(&IntegratedActionDataRK3::ddli_ddx, bp::return_internal_reference<>()),
                    "list of second partial derivatives of the cost with respect to the state of the RK3 "
                    "integration. ddli_ddx")
      .add_property("ddli_ddw", bp::make_getter(&IntegratedActionDataRK3::ddli_ddw, bp::return_internal_reference<>()),
                    "list of second partial derivatives of the cost with respect to the control parameters of "
                    "the RK3 integration. ddli_ddw")
      .add_property("ddli_ddu", bp::make_getter(&IntegratedActionDataRK3::ddli_ddu, bp::return_internal_reference<>()),
                    "list of second partial derivatives of the cost with respect to the control input of the RK3 "
                    "integration. ddli_ddu")
      .add_property(
          "ddli_dxdw", bp::make_getter(&IntegratedActionDataRK3::ddli_dxdw, bp::return_internal_reference<>()),
          "list of second partial derivatives of the cost with respect to the state and control parameters of "
          "the RK3 integration. ddli_dxdw")
      .add_property("ddli_dxdu",
                    bp::make_getter(&IntegratedActionDataRK3::ddli_dxdu, bp::return_internal_reference<>()),
                    "list of second partial derivatives of the cost with respect to the state and control input "
                    "of the RK3 integration. ddli_dxdu")
      .add_property("ddli_dwdu",
                    bp::make_getter(&IntegratedActionDataRK3::ddli_dwdu, bp::return_internal_reference<>()),
                    "list of second partial derivatives of the cost with respect to the control parameters and "
                    "inputs control of the RK3 integration. ddli_dxdu");
}

}  // namespace python
}  // namespace crocoddyl
