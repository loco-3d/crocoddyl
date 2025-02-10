///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, LAAS-CNRS, University of Edinburgh
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/activations/quadratic.hpp"

#include "python/crocoddyl/core/activation-base.hpp"
#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/utils/copyable.hpp"

namespace crocoddyl {
namespace python {

void exposeActivationQuad() {
  boost::python::register_ptr_to_python<
      std::shared_ptr<ActivationModelQuad> >();

  bp::class_<ActivationModelQuad, bp::bases<ActivationModelAbstract> >(
      "ActivationModelQuad",
      "Quadratic activation model.\n\n"
      "A quadratic action describes a quadratic function that depends on the "
      "residual, i.e.\n"
      "0.5 *||r||^2.",
      bp::init<std::size_t>(bp::args("self", "nr"),
                            "Initialize the activation model.\n\n"
                            ":param nr: dimension of the cost-residual vector"))
      .def("calc", &ActivationModelQuad::calc, bp::args("self", "data", "r"),
           "Compute the 0.5 * ||r||^2.\n\n"
           ":param data: activation data\n"
           ":param r: residual vector")
      .def("calcDiff", &ActivationModelQuad::calcDiff,
           bp::args("self", "data", "r"),
           "Compute the derivatives of a quadratic function.\n\n"
           "Note that the Hessian is constant, so we don't write again this "
           "value.\n"
           "It assumes that calc has been run first.\n"
           ":param data: activation data\n"
           ":param r: residual vector \n")
      .def("createData", &ActivationModelQuad::createData, bp::args("self"),
           "Create the quadratic activation data.\n\n")
      .def(CopyableVisitor<ActivationModelQuad>());
}

}  // namespace python
}  // namespace crocoddyl
