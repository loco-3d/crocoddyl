///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/activations/quadratic-flat-exp.hpp"

namespace crocoddyl {
namespace python {

void exposeActivationQuadFlatExp() {
  bp::class_<ActivationModelQuadFlatExp, bp::bases<ActivationModelAbstract>>(
      "ActivationModelQuadFlatExp",
      "Quadratic flat activation model.\n\n"
      "A quadratic flat action describes a quadratic flat function that "
      "depends on the residual, i.e.\n"
      "1 - exp(-||r||^2 / alpha).",
      bp::init<int, double>(bp::args("self", "nr", "alpha"),
                            "Initialize the activation model.\n\n"
                            ":param nr: dimension of the cost-residual vector"
                            "param alpha: width of quadratic basin near zero"))
      .def("calc", &ActivationModelQuadFlatExp::calc,
           bp::args("self", "data", "r"),
           "Compute the 1 - exp(-||r||^2 / alpha).\n\n"
           ":param data: activation data\n"
           ":param r: residual vector")
      .def("calcDiff", &ActivationModelQuadFlatExp::calcDiff,
           bp::args("self", "data", "r"),
           "Compute the derivatives of a quadratic flat function.\n\n"
           "Note that the Hessian is constant, so we don't write again this "
           "value.\n"
           ":param data: activation data\n"
           ":param r: residual vector \n")
      .def("createData", &ActivationModelQuadFlatExp::createData, bp::args("self"),
           "Create the quadratic flat activation data.\n\n")
      .add_property(
          "alpha",
          bp::make_function(&ActivationModelQuadFlatExp::get_alpha,
                            bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&ActivationModelQuadFlatExp::set_alpha), "alpha");
}

}  // namespace python
}  // namespace crocoddyl
