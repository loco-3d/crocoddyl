///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, University of Edinburgh, University of Trento,
//                          LAAS-CNRS, IRI: CSIC-UPC, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/integrator/rk.hpp"

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/integ-action-base.hpp"
#include "python/crocoddyl/utils/copyable.hpp"

namespace crocoddyl {
namespace python {

void exposeIntegratedActionRK() {
  bp::register_ptr_to_python<std::shared_ptr<IntegratedActionModelRK> >();

  bp::enum_<RKType>("RKType")
      .value("two", two)
      .value("three", three)
      .value("four", four)
      .export_values();

  bp::class_<IntegratedActionModelRK,
             bp::bases<IntegratedActionModelAbstract, ActionModelAbstract> >(
      "IntegratedActionModelRK",
      "RK integrator for differential action models.\n\n"
      "This class implements different RK integrator schemes.\n"
      "The available integrators are: RK2, RK3, and RK4.",
      bp::init<std::shared_ptr<DifferentialActionModelAbstract>, RKType,
               bp::optional<double, bool> >(
          bp::args("self", "diffModel", "rktype", "stepTime",
                   "withCostResidual"),
          "Initialize the RK integrator.\n\n"
          ":param diffModel: differential action model\n"
          ":param rktype: type of RK integrator (options are two, three, and "
          "four)\n"
          ":param stepTime: step time (default 1e-3)\n"
          ":param withCostResidual: includes the cost residuals and "
          "derivatives (default True)."))
      .def(bp::init<std::shared_ptr<DifferentialActionModelAbstract>,
                    std::shared_ptr<ControlParametrizationModelAbstract>,
                    RKType, bp::optional<double, bool> >(
          bp::args("self", "diffModel", "control", "rktype", "stepTime",
                   "withCostResidual"),
          "Initialize the RK integrator.\n\n"
          ":param diffModel: differential action model\n"
          ":param control: the control parametrization\n"
          ":param rktype: type of RK integrator (options are two, three, and "
          "four)\n"
          ":param stepTime: step time (default 1e-3)\n"
          ":param withCostResidual: includes the cost residuals and "
          "derivatives (default True)."))
      .def<void (IntegratedActionModelRK::*)(
          const std::shared_ptr<ActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &IntegratedActionModelRK::calc,
          bp::args("self", "data", "x", "u"),
          "Compute the time-discrete evolution of a differential action "
          "model.\n\n"
          "It describes the time-discrete evolution of action model.\n"
          ":param data: action data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (IntegratedActionModelRK::*)(
          const std::shared_ptr<ActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ActionModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (IntegratedActionModelRK::*)(
          const std::shared_ptr<ActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &IntegratedActionModelRK::calcDiff,
          bp::args("self", "data", "x", "u"),
          "Computes the derivatives of the integrated action model wrt state "
          "and control. \n\n"
          "This function builds a quadratic approximation of the\n"
          "action model (i.e. dynamical system and cost function).\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (IntegratedActionModelRK::*)(
          const std::shared_ptr<ActionDataAbstract>&,
          const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ActionModelAbstract::calcDiff,
          bp::args("self", "data", "x"))
      .def("createData", &IntegratedActionModelRK::createData, bp::args("self"),
           "Create the RK integrator data.")
      .add_property(
          "ni",
          bp::make_function(&IntegratedActionModelRK::get_ni,
                            bp::return_value_policy<bp::return_by_value>()),
          "number of nodes to be integrated")
      .def(CopyableVisitor<IntegratedActionModelRK>());

  bp::register_ptr_to_python<std::shared_ptr<IntegratedActionDataRK> >();

  bp::class_<IntegratedActionDataRK, bp::bases<IntegratedActionDataAbstract> >(
      "IntegratedActionDataRK", "RK integrator data.",
      bp::init<IntegratedActionModelRK*>(bp::args("self", "model"),
                                         "Create RK integrator data.\n\n"
                                         ":param model: RK integrator model"))
      .add_property(
          "differential",
          bp::make_getter(&IntegratedActionDataRK::differential,
                          bp::return_value_policy<bp::return_by_value>()),
          "list of differential action data")
      .add_property(
          "control",
          bp::make_getter(&IntegratedActionDataRK::control,
                          bp::return_value_policy<bp::return_by_value>()),
          "list of control parametrization data")
      .add_property(
          "integral",
          bp::make_getter(&IntegratedActionDataRK::integral,
                          bp::return_value_policy<bp::return_by_value>()),
          "list of RK terms related to the cost")
      .add_property("dx",
                    bp::make_getter(&IntegratedActionDataRK::dx,
                                    bp::return_internal_reference<>()),
                    "state rate.")
      .add_property("ki",
                    bp::make_getter(&IntegratedActionDataRK::ki,
                                    bp::return_internal_reference<>()),
                    "list of RK terms related to system dynamics")
      .add_property("y",
                    bp::make_getter(&IntegratedActionDataRK::y,
                                    bp::return_internal_reference<>()),
                    "list of states where f is evaluated in the RK integration")
      .add_property("ws",
                    bp::make_getter(&IntegratedActionDataRK::ws,
                                    bp::return_internal_reference<>()),
                    "control inputs evaluated in the RK integration")
      .add_property("dki_dx",
                    bp::make_getter(&IntegratedActionDataRK::dki_dx,
                                    bp::return_internal_reference<>()),
                    "list of partial derivatives of RK nodes with respect to "
                    "the state of the RK "
                    "integration. dki/dx")
      .add_property("dki_du",
                    bp::make_getter(&IntegratedActionDataRK::dki_du,
                                    bp::return_internal_reference<>()),
                    "list of partial derivatives of RK nodes with respect to "
                    "the control parameters of the RK "
                    "integration. dki/du")
      .add_property("dyi_dx",
                    bp::make_getter(&IntegratedActionDataRK::dyi_dx,
                                    bp::return_internal_reference<>()),
                    "list of partial derivatives of RK dynamics with respect "
                    "to the state of the RK integrator. dyi/dx")
      .add_property("dyi_du",
                    bp::make_getter(&IntegratedActionDataRK::dyi_du,
                                    bp::return_internal_reference<>()),
                    "list of partial derivatives of RK dynamics with respect "
                    "to the control parameters of the "
                    "RK integrator. dyi/du")
      .add_property("dli_dx",
                    bp::make_getter(&IntegratedActionDataRK::dli_dx,
                                    bp::return_internal_reference<>()),
                    "list of partial derivatives of the cost with respect to "
                    "the state of the RK "
                    "integration. dli_dx")
      .add_property("dli_du",
                    bp::make_getter(&IntegratedActionDataRK::dli_du,
                                    bp::return_internal_reference<>()),
                    "list of partial derivatives of the cost with respect to "
                    "the control input of the RK "
                    "integration. dli_du")
      .add_property("ddli_ddx",
                    bp::make_getter(&IntegratedActionDataRK::ddli_ddx,
                                    bp::return_internal_reference<>()),
                    "list of second partial derivatives of the cost with "
                    "respect to the state of the RK "
                    "integration. ddli_ddx")
      .add_property("ddli_ddw",
                    bp::make_getter(&IntegratedActionDataRK::ddli_ddw,
                                    bp::return_internal_reference<>()),
                    "list of second partial derivatives of the cost with "
                    "respect to the control of the differential"
                    "action model w. ddli_ddw")
      .add_property("ddli_ddu",
                    bp::make_getter(&IntegratedActionDataRK::ddli_ddu,
                                    bp::return_internal_reference<>()),
                    "list of second partial derivatives of the cost with "
                    "respect to the control input of the RK "
                    "integration. ddli_ddu")
      .add_property("ddli_dxdw",
                    bp::make_getter(&IntegratedActionDataRK::ddli_dxdw,
                                    bp::return_internal_reference<>()),
                    "list of second partial derivatives of the cost with "
                    "respect to the state and control of the differential"
                    "action model. ddli_dxdw")
      .add_property("ddli_dxdu",
                    bp::make_getter(&IntegratedActionDataRK::ddli_dxdu,
                                    bp::return_internal_reference<>()),
                    "list of second partial derivatives of the cost with "
                    "respect to the state and control input "
                    "of the RK integration. ddli_dxdu")
      .add_property(
          "ddli_dwdu",
          bp::make_getter(&IntegratedActionDataRK::ddli_dwdu,
                          bp::return_internal_reference<>()),
          "list of second partial derivatives of the cost with respect to the "
          "control of the differential action"
          "model and the control inputs of the RK integration. ddli_dwdu")
      .def(CopyableVisitor<IntegratedActionDataRK>());
}

}  // namespace python
}  // namespace crocoddyl
