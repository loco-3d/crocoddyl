///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/activations/quadratic-log.hpp"

namespace crocoddyl {
namespace python {

void exposeActivationQuadLog() {
  bp::class_<ActivationModelQuadLog, bp::bases<ActivationModelAbstract> >(
      "ActivationModelQuadLog",
      "Quadratic flat activation model.\n"
      "A quadratic flat action describes a quadratic flat function that depends on the residual, i.e.\n"
      "log(1 + ||r||^2 / sigma2).",
      bp::init<int,double>(bp::args("self", "nr","sigma2"),
                    "Initialize the activation model.\n\n"
                    ":param nr: dimension of the cost-residual vector"
                    "param sigma2: width of basin"))
      .def("calc", &ActivationModelQuadLog::calc, bp::args("self", "data", "r"),
           "Compute the log(1 + ||r||^2 / sigma2).\n"
           ":param data: activation data\n"
           ":param r: residual vector")
      .def("calcDiff", &ActivationModelQuadLog::calcDiff, bp::args("self", "data", "r"),
           "Compute the derivatives of a quadratic flat function.\n"
           "Note that the Hessian is constant, so we don't write again this value.\n"
           ":param data: activation data\n"
           ":param r: residual vector \n")
      .def("createData", &ActivationModelQuadLog::createData, bp::args("self"),
           "Create the quadratic flat activation data.\n")
      .add_property("sigma2",
                    bp::make_function(&ActivationModelQuadLog::get_sigma2,
                                      bp::return_value_policy<bp::return_by_value>()),
                    bp::make_function(&ActivationModelQuadLog::set_sigma2),
                    "threshold");
}

}  // namespace python
}  // namespace crocoddyl
