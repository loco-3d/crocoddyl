///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "colmpc/activation-distance-quadratic.hpp"

#include "colmpc/python.hpp"

namespace colmpc {
namespace python {

namespace bp = boost::python;

void exposeActivationModelDistanceQuad() {
  bp::register_ptr_to_python<std::shared_ptr<ActivationModelDistanceQuad> >();

  bp::class_<ActivationModelDistanceQuad,
             bp::bases<crocoddyl::ActivationModelAbstract> >(
      "ActivationModelDistanceQuad",
      "Quadratic activation model.\n\n"
      "A quadratic  action describes a quadratic  function that "
      "depends on the residual, i.e.\n"
      "exp(-||r||^2 / alpha).",
      bp::init<int, double>(bp::args("self", "nr", "d0"),
                            "Initialize the activation model.\n\n"
                            ":param nr: dimension of the cost-residual vector"
                            "param d0: activation distance"))
      .def("calc", &ActivationModelDistanceQuad::calc,
           bp::args("self", "data", "r"),
           "Compute the exp(-||r||^2 / alpha).\n\n"
           ":param data: activation data\n"
           ":param r: residual vector")
      .def("calcDiff", &ActivationModelDistanceQuad::calcDiff,
           bp::args("self", "data", "r"),
           "Compute the derivatives of a quadratic  function.\n\n"
           "Note that the Hessian is constant, so we don't write again this "
           "value.\n"
           ":param data: activation data\n"
           ":param r: residual vector \n")
      .def("createData", &ActivationModelDistanceQuad::createData,
           bp::args("self"), "Create the quadratic  activation data.\n\n")
      .add_property(
          "d0", bp::make_function(&ActivationModelDistanceQuad::get_d0),
          bp::make_function(&ActivationModelDistanceQuad::set_d0), "d0");
}

}  // namespace python
}  // namespace colmpc
