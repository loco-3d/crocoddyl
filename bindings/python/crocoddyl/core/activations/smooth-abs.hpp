///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_ACTIVATIONS_SMOOTH_ABS_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_ACTIVATIONS_SMOOTH_ABS_HPP_

#include "crocoddyl/core/activations/smooth-abs.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeActivationSmoothAbs() {
  bp::class_<ActivationModelSmoothAbs, bp::bases<ActivationModelAbstract> >(
      "ActivationModelSmoothAbs",
      "Smooth-absolute activation model.\n\n"
      "It describes a smooth representation of an absolute activation (1-norm), i.e.\n"
      "sqrt{1 + ||r||^2}.",
      bp::init<int>(bp::args("self", "nr"),
                    "Initialize the activation model.\n\n"
                    ":param nr: dimension of the cost-residual vector"))
      .def("calc", &ActivationModelSmoothAbs::calc_wrap, bp::args("self", "data", "r"),
           "Compute the sqrt{1 + ||r||^2}.\n\n"
           ":param data: activation data\n"
           ":param r: residual vector")
      .def<void (ActivationModelSmoothAbs::*)(const boost::shared_ptr<ActivationDataAbstract>&, const Eigen::VectorXd&)>("calcDiff", &ActivationModelSmoothAbs::calcDiff_wrap,
                                                            bp::args("self", "data", "r"),
                                                            "Compute the derivatives of a smoot-abs function.\n\n"
                                                            ":param data: activation data\n"
                                                            ":param r: residual vector \n")

      .def("createData", &ActivationModelSmoothAbs::createData, bp::args("self"),
           "Create the smooth-abs activation data.\n\n");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_ACTIVATIONS_SMOOTH_ABS_HPP_
