///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, University of Oxford,
//                          University of Trento
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/integ-action-base.hpp"
#include "crocoddyl/core/integrator/euler.hpp"

namespace crocoddyl {
namespace python {

void exposeIntegratedActionEuler() {
  bp::register_ptr_to_python<boost::shared_ptr<IntegratedActionModelEuler> >();

  bp::class_<IntegratedActionModelEuler, bp::bases<IntegratedActionModelAbstract, ActionModelAbstract> >(
      "IntegratedActionModelEuler",
      "Sympletic Euler integrator for differential action models.\n\n"
      "This class implements a sympletic Euler integrator (a.k.a semi-implicit\n"
      "integrator) give a differential action model, i.e.:\n"
      "  [q+, v+] = State.integrate([q, v], [v + a * dt, a * dt] * dt).",
      bp::init<boost::shared_ptr<DifferentialActionModelAbstract>, bp::optional<double, bool> >(
          bp::args("self", "diffModel", "stepTime", "withCostResidual"),
          "Initialize the sympletic Euler integrator.\n\n"
          ":param diffModel: differential action model\n"
          ":param stepTime: step time (default 1e-3)\n"
          ":param withCostResidual: includes the cost residuals and derivatives (default True)."))
      .def(bp::init<boost::shared_ptr<DifferentialActionModelAbstract>,
                    boost::shared_ptr<ControlParametrizationModelAbstract>, bp::optional<double, bool> >(
          bp::args("self", "diffModel", "control", "stepTime", "withCostResidual"),
          "Initialize the Euler integrator.\n\n"
          ":param diffModel: differential action model\n"
          ":param control: the control parametrization\n"
          ":param stepTime: step time (default 1e-3)\n"
          ":param withCostResidual: includes the cost residuals and derivatives (default True)."))
      .def<void (IntegratedActionModelEuler::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &IntegratedActionModelEuler::calc, bp::args("self", "data", "x", "u"),
          "Compute the time-discrete evolution of a differential action model.\n\n"
          "It describes the time-discrete evolution of action model.\n"
          ":param data: action data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (IntegratedActionModelEuler::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ActionModelAbstract::calc, bp::args("self", "data", "x"))
      .def<void (IntegratedActionModelEuler::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &IntegratedActionModelEuler::calcDiff, bp::args("self", "data", "x", "u"),
          "Computes the derivatives of the integrated action model wrt state and control. \n\n"
          "This function builds a quadratic approximation of the\n"
          "action model (i.e. dynamical system and cost function).\n"
          "It assumes that calc has been run first.\n"
          ":param data: action data\n"
          ":param x: state point (dim. state.nx)\n"
          ":param u: control input (dim. nu)")
      .def<void (IntegratedActionModelEuler::*)(const boost::shared_ptr<ActionDataAbstract>&,
                                                const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ActionModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &IntegratedActionModelEuler::createData, bp::args("self"),
           "Create the Euler integrator data.")
      .def("multiplyByFu", &IntegratedActionModelEuler::multiplyByFu, bp::args("self", "data", "A"),
           "Compute the product between the given matrix A and the Jacobian of the dynamics wrt control.\n\n"
           "It assumes that calcDiff has been run first.\n"
           ":param data: action data\n"
           ":param A: matrix to multiply (dim na x state.ndx)\n"
           ":return product between A and the Jacobian of the dynamics with respect the control (dim na x nu)")
      .def("multiplyFuTransposeBy", &IntegratedActionModelEuler::multiplyFuTransposeBy, bp::args("self", "Fu", "A"),
           "Compute the product between the transpose of the Jacobian of the dynamics wrt control and the given matrix A.\n\n"
           "It assumes that calcDiff has been run first.\n"
           ":param Fu: Jacobian matrix of the dynamics with respect to the control\n"
           ":param A: matrix to multiply (dim state.ndx x na)\n"
           ":return product between the tranpose of the Jacobian of the dynamics with respect the control and A  (dim "
           "nu x na)");

  bp::register_ptr_to_python<boost::shared_ptr<IntegratedActionDataEuler> >();

  bp::class_<IntegratedActionDataEuler, bp::bases<IntegratedActionDataAbstract> >(
      "IntegratedActionDataEuler", "Sympletic Euler integrator data.",
      bp::init<IntegratedActionModelEuler*>(bp::args("self", "model"),
                                            "Create sympletic Euler integrator data.\n\n"
                                            ":param model: sympletic Euler integrator model"))
      .add_property(
          "differential",
          bp::make_getter(&IntegratedActionDataEuler::differential, bp::return_value_policy<bp::return_by_value>()),
          "differential action data")
      .add_property(
          "control",
          bp::make_getter(&IntegratedActionDataEuler::control, bp::return_value_policy<bp::return_by_value>()),
          "control parametrization data")
      .add_property("dx", bp::make_getter(&IntegratedActionDataEuler::dx, bp::return_internal_reference<>()),
                    "state rate.")
      .add_property("Lwu", bp::make_getter(&IntegratedActionDataEuler::Lwu, bp::return_internal_reference<>()),
                    "Hessian of the cost wrt the differential control (w) and the control parameters (u).");
}

}  // namespace python
}  // namespace crocoddyl
