///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, LAAS-CNRS, University of Edinburgh
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/activations/weighted-quadratic.hpp"

#include "python/crocoddyl/core/activation-base.hpp"
#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/utils/copyable.hpp"

namespace crocoddyl {
namespace python {

void exposeActivationWeightedQuad() {
  boost::python::register_ptr_to_python<
      std::shared_ptr<ActivationModelWeightedQuad> >();

  bp::class_<ActivationModelWeightedQuad, bp::bases<ActivationModelAbstract> >(
      "ActivationModelWeightedQuad",
      "Weighted quadratic activation model.\n\n"
      "A weighted quadratic action describes a quadratic function that depends "
      "on the residual,\n"
      "i.e. 0.5 *||r||_w^2.",
      bp::init<Eigen::VectorXd>(
          bp::args("self", "weights"),
          "Initialize the activation model.\n\n"
          ":param weights: weights vector, note that nr=weights.size()"))
      .def("calc", &ActivationModelWeightedQuad::calc,
           bp::args("self", "data", "r"),
           "Compute the 0.5 * ||r||_w^2.\n\n"
           ":param data: activation data\n"
           ":param r: residual vector")
      .def("calcDiff", &ActivationModelWeightedQuad::calcDiff,
           bp::args("self", "data", "r"),
           "Compute the derivatives of a quadratic function.\n\n"
           ":param data: activation data\n"
           "Note that the Hessian is constant, so we don't write again this "
           "value.\n"
           "It assumes that calc has been run first.\n"
           ":param r: residual vector \n")
      .def("createData", &ActivationModelWeightedQuad::createData,
           bp::args("self"), "Create the weighted quadratic action data.")
      .add_property("weights",
                    bp::make_function(&ActivationModelWeightedQuad::get_weights,
                                      bp::return_internal_reference<>()),
                    &ActivationModelWeightedQuad::set_weights,
                    "weights of the quadratic term")
      .def(CopyableVisitor<ActivationModelWeightedQuad>());
}

}  // namespace python
}  // namespace crocoddyl
