///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, LAAS-CNRS, University of Edinburgh, IRI: CSIC-UPC,
//                          University of Trento, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/integrator/rk4.hpp"

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/integ-action-base.hpp"
#include "python/crocoddyl/utils/copyable.hpp"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

namespace crocoddyl {
namespace python {

void exposeIntegratedActionRK4() {
  bp::register_ptr_to_python<std::shared_ptr<IntegratedActionModelRK4> >();

  bp::class_<IntegratedActionModelRK4,
             bp::bases<IntegratedActionModelAbstract, ActionModelAbstract> >(
      "IntegratedActionModelRK4",
      "RK4 integrator for differential action models.\n\n"
      "This class implements an RK4 integrator\n"
      "given a differential action model, i.e.:\n"
      "  [q+, v+] = State.integrate([q, v], dt / 6 (k0 + k1 + k2 + k3)) with\n"
      "k0 = f(x, u)\n"
      "k1 = f(x + dt / 2 * k0, u)\n"
      "k2 = f(x + dt / 2 * k1, u)\n"
      "k3 = f(x + dt * k2, u)",
      bp::init<std::shared_ptr<DifferentialActionModelAbstract>,
               bp::optional<double, bool> >(
          bp::args("self", "diffModel", "stepTime", "withCostResidual"),
          "Initialize the RK4 integrator.\n\n"
          ":param diffModel: differential action model\n"
          ":param stepTime: step time (default 1e-3)\n"
          ":param withCostResidual: includes the cost residuals and "
          "derivatives (default True)."))
      .def(bp::init<std::shared_ptr<DifferentialActionModelAbstract>,
                    std::shared_ptr<ControlParametrizationModelAbstract>,
                    bp::optional<double, bool> >(
          bp::args("self", "diffModel", "control", "stepTime",
                   "withCostResidual"),
          "Initialize the RK4 integrator.\n\n"
          ":param diffModel: differential action model\n"
          ":param control: the control parametrization\n"
          ":param stepTime: step time (default 1e-3)\n"
          ":param withCostResidual: includes the cost residuals and "
          "derivatives (default True)."))
      .def<void (IntegratedActionModelRK4::*)(
          const std::shared_ptr<ActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &IntegratedActionModelRK4::calc,
          bp::args("self", "data", "x", "u"),
          "Compute the time-discrete evolution of a differential action "
          "model.\n\n"
          "It describes the time-discrete evolution of action model.\n"
          ":param data: action data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (IntegratedActionModelRK4::*)(
          const std::shared_ptr<ActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ActionModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (IntegratedActionModelRK4::*)(
          const std::shared_ptr<ActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &IntegratedActionModelRK4::calcDiff,
          bp::args("self", "data", "x", "u"),
          "Computes the derivatives of the integrated action model wrt state "
          "and control. \n\n"
          "This function builds a quadratic approximation of the\n"
          "action model (i.e. dynamical system and cost function).\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (IntegratedActionModelRK4::*)(
          const std::shared_ptr<ActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ActionModelAbstract::calcDiff,
          bp::args("self", "data", "x"))
      .def("createData", &IntegratedActionModelRK4::createData,
           bp::args("self"), "Create the RK4 integrator data.")
      .def(CopyableVisitor<IntegratedActionModelRK4>());

  bp::register_ptr_to_python<std::shared_ptr<IntegratedActionDataRK4> >();

  bp::class_<IntegratedActionDataRK4, bp::bases<IntegratedActionDataAbstract> >(
      "IntegratedActionDataRK4", "RK4 integrator data.",
      bp::init<IntegratedActionModelRK4*>(bp::args("self", "model"),
                                          "Create RK4 integrator data.\n\n"
                                          ":param model: RK4 integrator model"))
      .add_property(
          "differential",
          bp::make_getter(&IntegratedActionDataRK4::differential,
                          bp::return_value_policy<bp::return_by_value>()),
          "list of differential action data")
      .add_property(
          "control",
          bp::make_getter(&IntegratedActionDataRK4::control,
                          bp::return_value_policy<bp::return_by_value>()),
          "list of control parametrization data")
      .add_property(
          "integral",
          bp::make_getter(&IntegratedActionDataRK4::integral,
                          bp::return_value_policy<bp::return_by_value>()),
          "list of RK4 terms related to the cost")
      .add_property("dx",
                    bp::make_getter(&IntegratedActionDataRK4::dx,
                                    bp::return_internal_reference<>()),
                    "state rate.")
      .add_property("ki",
                    bp::make_getter(&IntegratedActionDataRK4::ki,
                                    bp::return_internal_reference<>()),
                    "list of RK4 terms related to system dynamics")
      .add_property(
          "y",
          bp::make_getter(&IntegratedActionDataRK4::y,
                          bp::return_internal_reference<>()),
          "list of states where f is evaluated in the RK4 integration")
      .add_property("ws",
                    bp::make_getter(&IntegratedActionDataRK4::ws,
                                    bp::return_internal_reference<>()),
                    "control inputs evaluated in the RK4 integration")
      .add_property("dki_dx",
                    bp::make_getter(&IntegratedActionDataRK4::dki_dx,
                                    bp::return_internal_reference<>()),
                    "list of partial derivatives of RK4 nodes with respect to "
                    "the state of the RK4 "
                    "integration. dki/dx")
      .add_property("dki_du",
                    bp::make_getter(&IntegratedActionDataRK4::dki_du,
                                    bp::return_internal_reference<>()),
                    "list of partial derivatives of RK4 nodes with respect to "
                    "the control parameters of the RK4 "
                    "integration. dki/du")
      .add_property("dyi_dx",
                    bp::make_getter(&IntegratedActionDataRK4::dyi_dx,
                                    bp::return_internal_reference<>()),
                    "list of partial derivatives of RK4 dynamics with respect "
                    "to the state of the RK4 integrator. dyi/dx")
      .add_property("dyi_du",
                    bp::make_getter(&IntegratedActionDataRK4::dyi_du,
                                    bp::return_internal_reference<>()),
                    "list of partial derivatives of RK4 dynamics with respect "
                    "to the control parameters of the "
                    "RK4 integrator. dyi/du")
      .add_property("dli_dx",
                    bp::make_getter(&IntegratedActionDataRK4::dli_dx,
                                    bp::return_internal_reference<>()),
                    "list of partial derivatives of the cost with respect to "
                    "the state of the RK4 "
                    "integration. dli_dx")
      .add_property("dli_du",
                    bp::make_getter(&IntegratedActionDataRK4::dli_du,
                                    bp::return_internal_reference<>()),
                    "list of partial derivatives of the cost with respect to "
                    "the control input of the RK4 "
                    "integration. dli_du")
      .add_property("ddli_ddx",
                    bp::make_getter(&IntegratedActionDataRK4::ddli_ddx,
                                    bp::return_internal_reference<>()),
                    "list of second partial derivatives of the cost with "
                    "respect to the state of the RK4 "
                    "integration. ddli_ddx")
      .add_property("ddli_ddw",
                    bp::make_getter(&IntegratedActionDataRK4::ddli_ddw,
                                    bp::return_internal_reference<>()),
                    "list of second partial derivatives of the cost with "
                    "respect to the control parameters of "
                    "the RK4 integration. ddli_ddw")
      .add_property("ddli_ddu",
                    bp::make_getter(&IntegratedActionDataRK4::ddli_ddu,
                                    bp::return_internal_reference<>()),
                    "list of second partial derivatives of the cost with "
                    "respect to the control input of the RK4 "
                    "integration. ddli_ddu")
      .add_property("ddli_dxdw",
                    bp::make_getter(&IntegratedActionDataRK4::ddli_dxdw,
                                    bp::return_internal_reference<>()),
                    "list of second partial derivatives of the cost with "
                    "respect to the state and control parameters of "
                    "the RK4 integration. ddli_dxdw")
      .add_property("ddli_dxdu",
                    bp::make_getter(&IntegratedActionDataRK4::ddli_dxdu,
                                    bp::return_internal_reference<>()),
                    "list of second partial derivatives of the cost with "
                    "respect to the state and control input "
                    "of the RK4 integration. ddli_dxdu")
      .add_property("ddli_dwdu",
                    bp::make_getter(&IntegratedActionDataRK4::ddli_dwdu,
                                    bp::return_internal_reference<>()),
                    "list of second partial derivatives of the cost with "
                    "respect to the control parameters and "
                    "inputs control of the RK4 integration. ddli_dxdu")
      .def(CopyableVisitor<IntegratedActionDataRK4>());
}

}  // namespace python
}  // namespace crocoddyl

#pragma GCC diagnostic pop
