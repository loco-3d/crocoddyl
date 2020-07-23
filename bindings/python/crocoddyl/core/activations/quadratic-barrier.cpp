///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/activations/quadratic-barrier.hpp"

namespace crocoddyl {
namespace python {

void exposeActivationQuadraticBarrier() {
  bp::class_<ActivationBounds>("ActivationBounds",
                               "Activation bounds.\n\n"
                               "The activation bounds describe the lower and upper vector plus it activation range\n"
                               "(between 0 and 1), its default value is 1. Note that a full activation is defined by\n"
                               "1 and the activation range is equally distributed between the lower and upper bounds.",
                               bp::init<Eigen::VectorXd, Eigen::VectorXd, bp::optional<double> >(
                                   bp::args("self", "lb", "ub", "beta"),
                                   "Initialize the activation bounds.\n\n"
                                   ":param lb: lower bounds\n"
                                   ":param ub: upper bounds\n"
                                   ":param beta: range of activation (between 0 to 1, default 1)"))
      .add_property("lb", bp::make_getter(&ActivationBounds::lb, bp::return_internal_reference<>()), "lower bounds")
      .add_property("ub", bp::make_getter(&ActivationBounds::ub, bp::return_internal_reference<>()), "upper bounds")
      .add_property("beta", &ActivationBounds::beta, "beta");

  bp::class_<ActivationModelQuadraticBarrier, bp::bases<ActivationModelAbstract> >(
      "ActivationModelQuadraticBarrier",
      "Inequality activation model.\n\n"
      "The activation is zero when r is between the lower (lb) and upper (ub) bounds, beta\n"
      "determines how much of the total range is not activated. This is the activation\n"
      "equations:\n"
      "a(r) = 0.5 * ||r||^2 for lb > r > ub\n"
      "a(r) = 0. for lb <= r <= ub.",
      bp::init<ActivationBounds>(bp::args("self", "bounds"),
                                 "Initialize the activation model.\n\n"
                                 ":param bounds: activation bounds"))
      .def("calc", &ActivationModelQuadraticBarrier::calc, bp::args("self", "data", "r"),
           "Compute the inequality activation.\n\n"
           ":param data: activation data\n"
           ":param r: residual vector")
      .def("calcDiff", &ActivationModelQuadraticBarrier::calcDiff, bp::args("self", "data", "r"),
           "Compute the derivatives of inequality activation.\n\n"
           ":param data: activation data\n"
           "Note that the Hessian is constant, so we don't write again this value.\n"
           ":param r: residual vector \n")
      .def("createData", &ActivationModelQuadraticBarrier::createData, bp::args("self"),
           "Create the weighted quadratic action data.")
      .add_property("bounds",
                    bp::make_function(&ActivationModelQuadraticBarrier::get_bounds,
                                      bp::return_value_policy<bp::return_by_value>()),
                    bp::make_function(&ActivationModelQuadraticBarrier::set_bounds),
                    "bounds (beta, lower and upper bounds)");
}

}  // namespace python
}  // namespace crocoddyl
