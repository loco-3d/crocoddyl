///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_ACTIVATIONS_INEQUALITY_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_ACTIVATIONS_INEQUALITY_HPP_

#include "crocoddyl/core/activations/inequality.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeActivationInequality() {
  bp::class_<ActivationBounds>(
      "ActivationBounds",
      "Activation bounds.\n\n"
      "The activation bounds describes the lower and upper vector plus it activation range\n"
      "(between 0 and 1), its default value is 1.",
      bp::init<Eigen::VectorXd, Eigen::VectorXd, double>(bp::args(" self", " lb", " ub", " beta=1."),
                                                         "Initialize the activation model.\n\n"
                                                         ":param lb: lower bounds\n"
                                                         ":param ub: upper bounds\n"
                                                         ":param beta: range of activation (between 0 to 1)"))
      .add_property("lb", bp::make_getter(&ActivationBounds::lb, bp::return_value_policy<bp::return_by_value>()),
                    "lower bounds")
      .add_property("ub", bp::make_getter(&ActivationBounds::ub, bp::return_value_policy<bp::return_by_value>()),
                    "upper bounds")
      .add_property("beta", &ActivationBounds::beta, "beta");

  bp::class_<ActivationModelInequality, bp::bases<ActivationModelAbstract> >(
      "ActivationModelInequality",
      "Inequality activation model.\n\n"
      "The activation is zero when r is between the lower (lb) and upper (ub) bounds, beta\n"
      "determines how much of the total range is not activated (default 0.9). This is the\n"
      "activation equations:\n"
      "a(r) = 0.5 * ||r||^2 for lb < r < ub\n"
      "a(r) = 0. for lb >= r >= ub.",
      bp::init<ActivationBounds>(bp::args(" self", " bounds"),
                                 "Initialize the activation model.\n\n"
                                 ":param bounds: activation bounds\n"
                                 ":param ub: upper bounds\n"
                                 ":param beta: range of activation (between 0 to 1)"))
      .def("calc", &ActivationModelInequality::calc_wrap, bp::args(" self", " data", " r"),
           "Compute the inequality activation.\n\n"
           ":param data: activation data\n"
           ":param r: residual vector")
      .def<void (ActivationModelInequality::*)(const boost::shared_ptr<ActivationDataAbstract>&,
                                               const Eigen::VectorXd&, const bool&)>(
          "calcDiff", &ActivationModelInequality::calcDiff_wrap, bp::args(" self", " data", " r", " recalc=True"),
          "Compute the derivatives of inequality activation.\n\n"
          ":param data: activation data\n"
          "Note that the Hessian is constant, so we don't write again this value.\n"
          ":param r: residual vector \n"
          ":param recalc: If true, it updates the residual value.")
      .def<void (ActivationModelInequality::*)(const boost::shared_ptr<ActivationDataAbstract>&,
                                               const Eigen::VectorXd&)>(
          "calcDiff", &ActivationModelInequality::calcDiff_wrap, bp::args(" self", " data", " r"))
      .def("createData", &ActivationModelInequality::createData, bp::args(" self"),
           "Create the weighted quadratic action data.")
      .add_property(
          "bounds",
          bp::make_function(&ActivationModelInequality::get_bounds, bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&ActivationModelInequality::set_bounds), "bounds (beta, lower and upper bounds)");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_ACTIVATIONS_INEQUALITY_HPP_
