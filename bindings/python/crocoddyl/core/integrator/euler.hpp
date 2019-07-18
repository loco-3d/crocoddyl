///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef PYTHON_CROCODDYL_CORE_INTEGRATOR_EULER_HPP_
#define PYTHON_CROCODDYL_CORE_INTEGRATOR_EULER_HPP_

#include "crocoddyl/core/integrator/euler.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeIntegratedActionEuler() {
  bp::class_<IntegratedActionModelEuler, bp::bases<ActionModelAbstract>>(
      "IntegratedActionModelEuler",
      R"(Sympletic Euler integrator for differential action models.

        This class implements a sympletic Euler integrator (a.k.a semi-implicit
        integrator) give a differential action model, i.e.:
          [q+, v+] = State.integrate([q, v], [v + a * dt, a * dt] * dt).)",
      bp::init<DifferentialActionModelAbstract*, bp::optional<double, bool>>(
          bp::args(" self", " diffModel", " stepTime", " withCostResidual"),
          R"(Initialize the sympletic Euler integrator.

:param diffModel: differential action model
:param stepTime: step time
:param withCostResidual: includes the cost residuals and derivatives.)")[bp::with_custodian_and_ward<1, 2>()])
      .def<void (IntegratedActionModelEuler::*)(std::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&,
                                                const Eigen::VectorXd&)>(
          "calc", &IntegratedActionModelEuler::calc_wrap, bp::args(" self", " data", " x", " u=None"),
          R"(Compute the time-discrete evolution of a differential action model.

It describes the time-discrete evolution of action model.
:param data: action data
:param x: state vector
:param u: control input)")
      .def<void (IntegratedActionModelEuler::*)(std::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&)>(
          "calc", &IntegratedActionModelEuler::calc_wrap, bp::args(" self", " data", " x"))
      .def<void (IntegratedActionModelEuler::*)(std::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&,
                                                const Eigen::VectorXd&, const bool&)>(
          "calcDiff", &IntegratedActionModelEuler::calcDiff_wrap,
          bp::args(" self", " data", " x", " u=None", " recalc=True"),
          R"(Compute the time-discrete derivatives of a differential action model.

It computes the time-discrete partial derivatives of a differential
action model. If recalc == True, it first updates the state evolution
and cost value. This function builds a quadratic approximation of the
action model (i.e. dynamical system and cost function).
:param data: action data
:param x: state vector
:param u: control input
:param recalc: If true, it updates the state evolution and the cost value.)")
      .def<void (IntegratedActionModelEuler::*)(std::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&,
                                                const Eigen::VectorXd&)>(
          "calcDiff", &IntegratedActionModelEuler::calcDiff_wrap, bp::args(" self", " data", " x", " u"))
      .def<void (IntegratedActionModelEuler::*)(std::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&)>(
          "calcDiff", &IntegratedActionModelEuler::calcDiff_wrap, bp::args(" self", " data", " x"))
      .def<void (IntegratedActionModelEuler::*)(std::shared_ptr<ActionDataAbstract>&, const Eigen::VectorXd&,
                                                const bool&)>("calcDiff", &IntegratedActionModelEuler::calcDiff_wrap,
                                                              bp::args(" self", " data", " x", " recalc"))
      .def("createData", &IntegratedActionModelEuler::createData, bp::args(" self"),
           R"(Create the Euler integrator data.)")
      .add_property(
          "differential",
          bp::make_function(&IntegratedActionModelEuler::get_differential, bp::return_internal_reference<>()),
          "differential action model");

  bp::register_ptr_to_python<std::shared_ptr<IntegratedActionDataEuler>>();

  bp::class_<IntegratedActionDataEuler, bp::bases<ActionDataAbstract>>(
      "IntegratedActionDataEuler",
      R"(Sympletic Euler integrator data.)",
      bp::init<IntegratedActionModelEuler*>(bp::args(" self", " model"),
                                            R"(Create sympletic Euler integrator data.

:param model: sympletic Euler integrator model)"));
}

}  // namespace python
}  // namespace crocoddyl

#endif