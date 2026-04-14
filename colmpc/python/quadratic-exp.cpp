///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "colmpc/quadratic-exp.hpp"

#include "colmpc/python.hpp"

namespace colmpc {
namespace python {

namespace bp = boost::python;

template <int N>
void exposeActivationModelExpTpl(const char* name) {
  typedef ActivationModelExpTpl<double, N> ActivationType;
  bp::register_ptr_to_python<std::shared_ptr<ActivationType> >();

  bp::class_<ActivationType, bp::bases<crocoddyl::ActivationModelAbstract> >(
      name,
      "Quadratic activation model.\n\n"
      "A quadratic  action describes a quadratic  function that "
      "depends on the residual, i.e.\n"
      "exp(-||r||^2 / alpha).",
      bp::init<int, double>(bp::args("self", "nr", "alpha"),
                            "Initialize the activation model.\n\n"
                            ":param nr: dimension of the cost-residual vector"
                            "param alpha: width of quadratic basin near zero"))
      .def("calc", &ActivationType::calc, bp::args("self", "data", "r"),
           "Compute the exp(-||r||^2 / alpha).\n\n"
           ":param data: activation data\n"
           ":param r: residual vector")
      .def("calcDiff", &ActivationType::calcDiff, bp::args("self", "data", "r"),
           "Compute the derivatives of a quadratic  function.\n\n"
           "Note that the Hessian is constant, so we don't write again this "
           "value.\n"
           ":param data: activation data\n"
           ":param r: residual vector \n")
      .def("createData", &ActivationType::createData, bp::args("self"),
           "Create the quadratic  activation data.\n\n")
      .add_property("alpha", bp::make_function(&ActivationType::get_alpha),
                    bp::make_function(&ActivationType::set_alpha), "alpha");
}

void exposeActivationModelQuadExp() {
  exposeActivationModelExpTpl<1>("ActivationModelExp");
  exposeActivationModelExpTpl<2>("ActivationModelQuadExp");
}

}  // namespace python
}  // namespace colmpc
