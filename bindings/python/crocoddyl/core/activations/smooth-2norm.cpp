///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/activations/smooth-2norm.hpp"
#include "python/crocoddyl/core/activation-base.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

void exposeActivationSmooth2Norm() {
  bp::class_<ActivationModelSmooth2Norm, bp::bases<ActivationModelAbstract>>(
      "ActivationModelSmooth2Norm",
      "Smooth-2Norm activation model.\n\n"
      "It describes a smooth representation of a 2-norm, i.e.\n"
      "sqrt{eps + sum^nr_{i=0} ||ri||^2}, where ri is the scalar residual for "
      "the i constraints,\n."
      "and nr is the dimension of the residual vector.",
      bp::init<int, bp::optional<double>>(
          bp::args("self", "nr", "eps"),
          "Initialize the activation model.\n\n"
          ":param nr: dimension of the residual vector\n"
          ":param eps: smoothing factor (default: 1.)"))
      .def("calc", &ActivationModelSmooth2Norm::calc,
           bp::args("self", "data", "r"),
           "Compute the smooth-2norm function.\n\n"
           ":param data: activation data\n"
           ":param r: residual vector")
      .def("calcDiff", &ActivationModelSmooth2Norm::calcDiff,
           bp::args("self", "data", "r"),
           "Compute the derivatives of a smooth-2norm function.\n\n"
           "It assumes that calc has been run first.\n"
           ":param data: activation data\n"
           ":param r: residual vector \n")
      .def("createData", &ActivationModelSmooth2Norm::createData,
           bp::args("self"), "Create the smooth-2norm activation data.\n\n");
}

} // namespace python
} // namespace crocoddyl
