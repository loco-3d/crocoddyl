///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, Airbus
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/activations/2norm-barrier.hpp"

namespace crocoddyl {
namespace python {

void exposeActivation2NormBarrier() {
  bp::class_<ActivationModel2NormBarrier, bp::bases<ActivationModelAbstract> >(
      "ActivationModel2NormBarrier",
      "An 2-norm activation model for a defined barrier alpha\n\n"
      "If the residual is over an alpha threshold, this function imposes a quadratic term. \n"
      "In short, the activation value is 0 if the residual is major to alpha, otherwise, it is \n"
      "equals to 0.5 *(||r|| - alpha)^2",
      bp::init<std::size_t, bp::optional<double, bool> >(bp::args("self", "nr", "alpha", "true_hessian"),
                                                         "Initialize the activation model.\n\n"
                                                         ":param nr: dimension of the cost-residual vector\n"
                                                         ":param alpha: activation threshold (default 0.1)\n"
                                                         ":param true_hessian: use true hessian in calcDiff if true, "
                                                         "else Gauss-Newton approximation (default false)"))
      .def("calc", &ActivationModel2NormBarrier::calc, bp::args("self", "data", "r"),
           "Compute the activation value.\n\n"
           ":param data: activation data\n"
           ":param r: residual vector")
      .def("calcDiff", &ActivationModel2NormBarrier::calcDiff, bp::args("self", "data", "r"),
           "Compute the derivatives of the collision function.\n\n"
           ":param data: activation data\n"
           ":param r: residual vector \n")
      .def("createData", &ActivationModel2NormBarrier::createData, bp::args("self"),
           "Create the collision activation data.\n\n")
      .add_property(
          "alpha",
          bp::make_function(&ActivationModel2NormBarrier::get_alpha, bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&ActivationModel2NormBarrier::set_alpha), "alpha");
}

}  // namespace python
}  // namespace crocoddyl
