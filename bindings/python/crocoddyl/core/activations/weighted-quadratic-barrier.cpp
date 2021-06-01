///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/activations/weighted-quadratic-barrier.hpp"
#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/activation-base.hpp"

namespace crocoddyl {
namespace python {

void exposeActivationWeightedQuadraticBarrier() {
  boost::python::register_ptr_to_python<boost::shared_ptr<ActivationModelWeightedQuadraticBarrier> >();

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
      .def("calc", &ActivationModelWeightedQuadraticBarrier::calc, bp::args("self", "data", "r"),
           "Compute the inequality activation.\n\n"
           ":param data: activation data\n"
           ":param r: residual vector")
      .def("calcDiff", &ActivationModelWeightedQuadraticBarrier::calcDiff, bp::args("self", "data", "r"),
           "Compute the derivatives of inequality activation.\n\n"
           ":param data: activation data\n"
           "Note that the Hessian is constant, so we don't write again this value.\n"
           "It assumes that calc has been run first.\n"
           ":param r: residual vector \n")
      .def("createData", &ActivationModelWeightedQuadraticBarrier::createData, bp::args("self"),
           "Create the weighted quadratic action data.")
      .add_property("bounds",
                    bp::make_function(&ActivationModelWeightedQuadraticBarrier::get_bounds,
                                      bp::return_value_policy<bp::return_by_value>()),
                    bp::make_function(&ActivationModelWeightedQuadraticBarrier::set_bounds),
                    "bounds (beta, lower and upper bounds)")
      .add_property(
          "weights",
          bp::make_function(&ActivationModelWeightedQuadraticBarrier::get_weights, bp::return_internal_reference<>()),
          bp::make_function(&ActivationModelWeightedQuadraticBarrier::set_weights), "vector of weights");
}

}  // namespace python
}  // namespace crocoddyl
