///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "crocoddyl/core/activations/weighted-quadratic.hpp"
#include "python/crocoddyl/core/activation-base.hpp"

namespace crocoddyl {
namespace python {

void exposeActivationWeightedQuad() {
  bp::class_<ActivationModelWeightedQuad, bp::bases<ActivationModelAbstract> >(
      "ActivationModelWeightedQuad",
      "Weighted quadratic activation model.\n\n"
      "A weighted quadratic action describes a quadratic function that depends on the residual,\n"
      "i.e. 0.5 *||r||_w^2.",
      bp::init<Eigen::VectorXd>(bp::args("self", "weights"),
                                "Initialize the activation model.\n\n"
                                ":param weights: weights vector, note that nr=weights.size()"))
      .def("calc", &ActivationModelWeightedQuad::calc_wrap, bp::args("self", "data", "r"),
           "Compute the 0.5 * ||r||_w^2.\n\n"
           ":param data: activation data\n"
           ":param r: residual vector")
      .def<void (ActivationModelWeightedQuad::*)(const boost::shared_ptr<ActivationDataAbstract>&,
                                                 const Eigen::VectorXd&)>(
          "calcDiff", &ActivationModelWeightedQuad::calcDiff_wrap, bp::args("self", "data", "r"),
          "Compute the derivatives of a quadratic function.\n\n"
          ":param data: activation data\n"
          "Note that the Hessian is constant, so we don't write again this value.\n"
          ":param r: residual vector \n")
      .def("createData", &ActivationModelWeightedQuad::createData, bp::args("self"),
           "Create the weighted quadratic action data.")
      .add_property(
          "weights",
          bp::make_function(&ActivationModelWeightedQuad::get_weights, bp::return_value_policy<bp::return_by_value>()),
          &ActivationModelWeightedQuad::set_weights, "weights of the quadratic term");
}

}  // namespace python
}  // namespace crocoddyl
