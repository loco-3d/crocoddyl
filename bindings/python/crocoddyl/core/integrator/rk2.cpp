///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh, University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/integ-action-base.hpp"
#include "crocoddyl/core/integrator/rk2.hpp"

namespace crocoddyl {
namespace python {

void exposeIntegratedActionRK2() {
  bp::register_ptr_to_python<boost::shared_ptr<IntegratedActionModelRK2> >();

  bp::class_<IntegratedActionModelRK2, bp::bases<IntegratedActionModelAbstract, ActionModelAbstract> >(
      "IntegratedActionModelRK2",
      "RK2 integrator for differential action models.\n\n"
      "This class implements an RK2 integrator\n"
      "given a differential action model, i.e.:\n"
      "  [q+, v+] = State.integrate([q, v], dt k1) with\n"
      "k0 = f(x, u)\n"
      "k1 = f(x + dt / 2 * k0, u)",
      bp::init<boost::shared_ptr<DifferentialActionModelAbstract>, bp::optional<double, bool> >(
          bp::args("self", "diffModel", "stepTime", "withCostResidual"),
          "Initialize the RK2 integrator.\n\n"
          ":param diffModel: differential action model\n"
          ":param stepTime: step time (default 1e-3)\n"
          ":param withCostResidual: includes the cost residuals and derivatives."))
      .def(bp::init<boost::shared_ptr<DifferentialActionModelAbstract>,
                    boost::shared_ptr<ControlParametrizationModelAbstract>, bp::optional<double, bool> >(
          bp::args("self", "diffModel", "control", "stepTime", "withCostResidual"),
          "Initialize the RK2 integrator.\n\n"
          ":param diffModel: differential action model\n"
          ":param control: the control parametrization\n"
          ":param stepTime: step time (default 1e-3)\n"
          ":param withCostResidual: includes the cost residuals and derivatives."))
      .def<void (IntegratedActionModelRK2::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                              const Eigen::Ref<const Eigen::VectorXd>&,
                                              const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &IntegratedActionModelRK2::calc, bp::args("self", "data", "x", "u"),
          "Compute the time-discrete evolution of a differential action model.\n\n"
          "It describes the time-discrete evolution of action model.\n"
          ":param data: action data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (IntegratedActionModelRK2::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                              const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ActionModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (IntegratedActionModelRK2::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                              const Eigen::Ref<const Eigen::VectorXd>&,
                                              const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &IntegratedActionModelRK2::calcDiff, bp::args("self", "data", "x", "u"),
          "Computes the derivatives of the integrated action model wrt state and control. \n\n"
          "This function builds a quadratic approximation of the\n"
          "action model (i.e. dynamical system and cost function).\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)\n")
      .def<void (IntegratedActionModelRK2::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                              const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ActionModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &IntegratedActionModelRK2::createData, bp::args("self"), "Create the RK2 integrator data.");

  bp::register_ptr_to_python<boost::shared_ptr<IntegratedActionDataRK2> >();

  bp::class_<IntegratedActionDataRK2, bp::bases<IntegratedActionDataAbstract> >(
      "IntegratedActionDataRK2", "RK2 integrator data.",
      bp::init<IntegratedActionModelRK2*>(bp::args("self", "model"),
                                          "Create RK2 integrator data.\n\n"
                                          ":param model: RK2 integrator model"))
      .add_property(
          "differential",
          bp::make_getter(&IntegratedActionDataRK2::differential, bp::return_value_policy<bp::return_by_value>()),
          "list of differential action data")
      .add_property(
          "integral",
          bp::make_getter(&IntegratedActionDataRK2::integral, bp::return_value_policy<bp::return_by_value>()),
          "list with the RK2 terms related to the cost")
      .add_property("ki", bp::make_getter(&IntegratedActionDataRK2::ki, bp::return_internal_reference<>()),
                    "list with the RK2 terms related to system dynamics")
      .add_property("y", bp::make_getter(&IntegratedActionDataRK2::y, bp::return_internal_reference<>()),
                    "list of states where f is evaluated in the RK2 integration")
      .add_property("ws", bp::make_getter(&IntegratedActionDataRK2::ws, bp::return_internal_reference<>()),
                    "control inputs evaluated in the RK2 integration")
      .add_property("dki_dx", bp::make_getter(&IntegratedActionDataRK2::dki_dx, bp::return_internal_reference<>()),
                    "list of partial derivatives of RK2 nodes with respect to the state of the RK2 "
                    "integration. dki/dx")
      .add_property("dki_du", bp::make_getter(&IntegratedActionDataRK2::dki_du, bp::return_internal_reference<>()),
                    "list of partial derivatives of RK2 nodes with respect to the control parameters of the RK2 "
                    "integration. dki/du")
      .add_property(
          "dyi_dx", bp::make_getter(&IntegratedActionDataRK2::dyi_dx, bp::return_internal_reference<>()),
          "list of partial derivatives of RK2 dynamics with respect to the state of the RK2 integrator. dyi/dx")
      .add_property("dyi_du", bp::make_getter(&IntegratedActionDataRK2::dyi_du, bp::return_internal_reference<>()),
                    "list of partial derivatives of RK2 dynamics with respect to the control parameters of the "
                    "RK2 integrator. dyi/du")
      .add_property("dli_dx", bp::make_getter(&IntegratedActionDataRK2::dli_dx, bp::return_internal_reference<>()),
                    "list of partial derivatives of the cost with respect to the state of the RK2 "
                    "integration. dli_dx")
      .add_property("dli_du", bp::make_getter(&IntegratedActionDataRK2::dli_du, bp::return_internal_reference<>()),
                    "list of partial derivatives of the cost with respect to the control input of the RK2 "
                    "integration. dli_du")
      .add_property("ddli_ddx", bp::make_getter(&IntegratedActionDataRK2::ddli_ddx, bp::return_internal_reference<>()),
                    "list of second partial derivatives of the cost with respect to the state of the RK2 "
                    "integration. ddli_ddx")
      .add_property("ddli_ddw", bp::make_getter(&IntegratedActionDataRK2::ddli_ddw, bp::return_internal_reference<>()),
                    "list of second partial derivatives of the cost with respect to the control parameters of "
                    "the RK2 integration. ddli_ddw")
      .add_property("ddli_ddu", bp::make_getter(&IntegratedActionDataRK2::ddli_ddu, bp::return_internal_reference<>()),
                    "list of second partial derivatives of the cost with respect to the control input of the RK2 "
                    "integration. ddli_ddu")
      .add_property(
          "ddli_dxdw", bp::make_getter(&IntegratedActionDataRK2::ddli_dxdw, bp::return_internal_reference<>()),
          "list of second partial derivatives of the cost with respect to the state and control parameters of "
          "the RK2 integration. ddli_dxdw")
      .add_property("ddli_dxdu",
                    bp::make_getter(&IntegratedActionDataRK2::ddli_dxdu, bp::return_internal_reference<>()),
                    "list of second partial derivatives of the cost with respect to the control parameters and "
                    "inputs control of the RK2 integration. ddli_dxdu");
}

}  // namespace python
}  // namespace crocoddyl
