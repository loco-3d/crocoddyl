///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_INTEGRATOR_EULER_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_INTEGRATOR_EULER_HPP_

#include "crocoddyl/core/integrator/euler.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeIntegratedActionEuler() {
  bp::class_<IntegratedActionModelEuler, bp::bases<ActionModelAbstract> >(
      "IntegratedActionModelEuler",
      "Sympletic Euler integrator for differential action models.\n\n"
      "This class implements a sympletic Euler integrator (a.k.a semi-implicit\n"
      "integrator) give a differential action model, i.e.:\n"
      "  [q+, v+] = State.integrate([q, v], [v + a * dt, a * dt] * dt).",
      bp::init<DifferentialActionModelAbstract*, bp::optional<double, bool> >(
          bp::args(" self", " diffModel", " stepTime", " withCostResidual"),
          "Initialize the sympletic Euler integrator.\n\n"
          ":param diffModel: differential action model\n"
          ":param stepTime: step time\n"
          ":param withCostResidual: includes the cost residuals and derivatives.")[bp::with_custodian_and_ward<1,
                                                                                                               2>()])
      .def<void (IntegratedActionModelEuler::*)(const boost::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&,
                                                const Eigen::VectorXd&)>(
          "calc", &IntegratedActionModelEuler::calc_wrap, bp::args(" self", " data", " x", " u=None"),
          "Compute the time-discrete evolution of a differential action model.\n\n"
          "It describes the time-discrete evolution of action model.\n"
          ":param data: action data\n"
          ":param x: state vector\n"
          ":param u: control input")
      .def<void (IntegratedActionModelEuler::*)(const boost::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&)>(
          "calc", &IntegratedActionModelEuler::calc_wrap, bp::args(" self", " data", " x"))
      .def<void (IntegratedActionModelEuler::*)(const boost::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&,
                                                const Eigen::VectorXd&, const bool&)>(
          "calcDiff", &IntegratedActionModelEuler::calcDiff_wrap,
          bp::args(" self", " data", " x", " u=None", " recalc=True"),
          "Compute the time-discrete derivatives of a differential action model.\n\n"
          "It computes the time-discrete partial derivatives of a differential\n"
          "action model. If recalc == True, it first updates the state evolution\n"
          "and cost value. This function builds a quadratic approximation of the\n"
          "action model (i.e. dynamical system and cost function).\n"
          ":param data: action data\n"
          ":param x: state vector\n"
          ":param u: control input\n"
          ":param recalc: If true, it updates the state evolution and the cost value.")
      .def<void (IntegratedActionModelEuler::*)(const boost::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&,
                                                const Eigen::VectorXd&)>(
          "calcDiff", &IntegratedActionModelEuler::calcDiff_wrap, bp::args(" self", " data", " x", " u"))
      .def<void (IntegratedActionModelEuler::*)(const boost::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&)>(
          "calcDiff", &IntegratedActionModelEuler::calcDiff_wrap, bp::args(" self", " data", " x"))
      .def<void (IntegratedActionModelEuler::*)(const boost::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&,
                                                const bool&)>("calcDiff", &IntegratedActionModelEuler::calcDiff_wrap,
                                                              bp::args(" self", " data", " x", " recalc"))
      .def("createData", &IntegratedActionModelEuler::createData, bp::args(" self"),
           "Create the Euler integrator data.")
      .add_property(
          "differential",
          bp::make_function(&IntegratedActionModelEuler::get_differential, bp::return_internal_reference<>()),
          "differential action model")
      .add_property(
          "dt", bp::make_function(&IntegratedActionModelEuler::get_dt, bp::return_value_policy<bp::return_by_value>()),
          &IntegratedActionModelEuler::set_dt, "step time");

  bp::register_ptr_to_python<boost::shared_ptr<IntegratedActionDataEuler> >();

  bp::class_<IntegratedActionDataEuler, bp::bases<ActionDataAbstract> >(
      "IntegratedActionDataEuler", "Sympletic Euler integrator data.",
      bp::init<IntegratedActionModelEuler*>(bp::args(" self", " model"),
                                            "Create sympletic Euler integrator data.\n\n"
                                            ":param model: sympletic Euler integrator model"));
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_INTEGRATOR_EULER_HPP_
