///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, University of Edinburgh, IRI: CSIC-UPC
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/actuation/squashing-base.hpp"
#include "crocoddyl/core/actuation/squashing/smooth-sat.hpp"

namespace crocoddyl {
namespace python {

void exposeSquashingSmoothSat() {
  bp::register_ptr_to_python<boost::shared_ptr<SquashingModelSmoothSat> >();

  bp::class_<SquashingModelSmoothSat, bp::bases<SquashingModelAbstract> >(
      "SquashingModelSmoothSat", "Smooth Sat squashing model",
      bp::init<Eigen::VectorXd, Eigen::VectorXd, int>(bp::args("self", "u_lb", "u_ub", "ns"),
                                                      "Initialize the squashing model. \n\n"
                                                      ":param u_lb: output lower bound"
                                                      ":param u_ub: output upper bound"
                                                      ":param ns: dimension of the input vector"))
      .def("calc", &SquashingModelSmoothSat::calc, bp::args("self", "data", "s"),
           "Compute the squashing value for a given value of s, component-wise. \n\n"
           ":param data: squashing data\n"
           ":param s: control input")
      .def("calcDiff", &SquashingModelSmoothSat::calcDiff, bp::args("self", "data", "s"),
           "Compute the derivative of the squashing function.\n\n"
           ":param data: squashing data\n"
           ":param s: squashing input.")
      .def("createData", &SquashingModelSmoothSat::createData, bp::args("self"), "Create the squashing data.\n\n")
      .add_property("smooth", bp::make_function(&SquashingModelSmoothSat::get_smooth),
                    bp::make_function(&SquashingModelSmoothSat::set_smooth),
                    "Smoothness parameter of the smooth sat. function");
}

}  // namespace python
}  // namespace crocoddyl