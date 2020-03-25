///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, University of Edinburgh, IRI: CSIC-UPC
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/actuation/squashing-base.hpp"
#include "crocoddyl/core/actuation/squashing/smooth-sat.hpp"

namespace crocoddyl {
namespace python {

void exposeSquashingSmoothSat() {
  bp::class_<SquashingModelSmoothSat, bp::bases<SquashingModelAbstract> >(
    "SquashingModelSmoothSat",
    "Smooth Sat squashing model",
    bp::init<Eigen::VectorXd, 
             Eigen::VectorXd,
             int>(bp::args("self", "out_lb", "out_ub", "ns"),
                                    "Initialize the squashing model. \n\n"
                                    ":param out_lb: lower bound"
                                    ":param out_lb: lower bound"
                                    ":param ns: dimension of the input vector"))
    .def("calc", &SquashingModelSmoothSat::calc_wrap, bp::args("self", "data", "u"),
      "Compute the squashing value for a given value of u, component-wise. \n\n"
      ":param data: squashing data\n"
      ":param u: control input")
    .def("calcDiff", &SquashingModelSmoothSat::calcDiff_wrap,
      SquashingModel_calcDiff_wraps(
      bp::args("self", "data", "u"),
      "Compute the derivative of the squashing function.\n\n"
      ":param data: squashing data\n"
      ":param u: control input".))
    .def("createData", &SquashingModelSmoothSat::createData, bp::args("self"),
      "Create the squashing data.\n\n")
    .add_property("smooth",
                    bp::make_function(&SquashingModelSmoothSat::get_smooth, bp::return_value_policy<bp::copy_const_reference>()),
                    bp::make_function(&SquashingModelSmoothSat::set_smooth),
                    "Smoothness parameter of the smooth sat. function");
}

} // namespace python
} // namespace crocoddyl