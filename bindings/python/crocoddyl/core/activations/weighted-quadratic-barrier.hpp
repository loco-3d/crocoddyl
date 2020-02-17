///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_ACTIVATIONS_WEIGHTED_QUADRATIC_BARRIER_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_ACTIVATIONS_WEIGHTED_QUADRATIC_BARRIER_HPP_

#include "crocoddyl/core/activations/weighted-quadratic-barrier.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeActivationWeightedQuadraticBarrier() {
  bp::class_<ActivationModelWeightedQuadraticBarrier, bp::bases<ActivationModelAbstract> >(
      "ActivationModelWeightedQuadraticBarrier",
      "Inequality activation model.\n\n"
      "The activation is zero when r is between the lower (lb) and upper (ub) bounds, beta\n"
      "determines how much of the total range is not activated. This is the activation\n"
      "equations:\n"
      "a(r) = 0.5 * ||r||_w^2 for lb < r < ub\n"
      "a(r) = 0. for lb >= r >= ub,"
      "where w is the vector of weights",
      bp::init<ActivationBounds, Eigen::VectorXd>(bp::args("self", "bounds", "weights"),
                                                  "Initialize the activation model.\n\n"
                                                  ":param bounds: activation bounds\n"
                                                  ":param weights: weights"))
      .def("calc", &ActivationModelWeightedQuadraticBarrier::calc_wrap, bp::args("self", "data", "r"),
           "Compute the inequality activation.\n\n"
           ":param data: activation data\n"
           ":param r: residual vector")
      .def<void (ActivationModelWeightedQuadraticBarrier::*)(const boost::shared_ptr<ActivationDataAbstract>&,
                                                             const Eigen::VectorXd&)>(
          "calcDiff", &ActivationModelWeightedQuadraticBarrier::calcDiff_wrap, bp::args("self", "data", "r"),
          "Compute the derivatives of inequality activation.\n\n"
          ":param data: activation data\n"
          "Note that the Hessian is constant, so we don't write again this value.\n"
          ":param r: residual vector \n")
      .def("createData", &ActivationModelWeightedQuadraticBarrier::createData, bp::args("self"),
           "Create the weighted quadratic action data.")
      .add_property("bounds",
                    bp::make_function(&ActivationModelWeightedQuadraticBarrier::get_bounds,
                                      bp::return_value_policy<bp::return_by_value>()),
                    bp::make_function(&ActivationModelWeightedQuadraticBarrier::set_bounds),
                    "bounds (beta, lower and upper bounds)")
      .add_property("weights",
                    bp::make_function(&ActivationModelWeightedQuadraticBarrier::get_weights,
                                      bp::return_value_policy<bp::return_by_value>()),
                    bp::make_function(&ActivationModelWeightedQuadraticBarrier::set_weights), "vector of weights");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_ACTIVATIONS_WEIGHTED_QUADRATIC_BARRIER_HPP_
