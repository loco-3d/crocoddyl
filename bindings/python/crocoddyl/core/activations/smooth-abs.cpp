///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/activations/smooth-abs.hpp"

namespace crocoddyl {
namespace python {

void exposeActivationSmoothAbs() {
  bp::class_<ActivationModelSmoothAbs, bp::bases<ActivationModelAbstract> >(
      "ActivationModelSmoothAbs",
      "Smooth-absolute activation model.\n\n"
      "It describes a smooth representation of an absolute activation (1-norm), i.e.\n"
      "sqrt{1 + ||r||^2}.",
      bp::init<int>(bp::args("self", "nr"),
                    "Initialize the activation model.\n\n"
                    ":param nr: dimension of the cost-residual vector"))
      .def("calc", &ActivationModelSmoothAbs::calc, bp::args("self", "data", "r"),
           "Compute the sqrt{1 + ||r||^2}.\n\n"
           ":param data: activation data\n"
           ":param r: residual vector")
      .def("calcDiff", &ActivationModelSmoothAbs::calcDiff, bp::args("self", "data", "r"),
           "Compute the derivatives of a smooth-abs function.\n\n"
           ":param data: activation data\n"
           ":param r: residual vector \n")

      .def("createData", &ActivationModelSmoothAbs::createData, bp::args("self"),
           "Create the smooth-abs activation data.\n\n");
}

}  // namespace python
}  // namespace crocoddyl
