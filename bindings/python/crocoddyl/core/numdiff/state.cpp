///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2023, University of Edinburgh, University of Pisa,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/numdiff/state.hpp"

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/state-base.hpp"
#include "python/crocoddyl/utils/copyable.hpp"

namespace crocoddyl {
namespace python {

void exposeStateNumDiff() {
  bp::register_ptr_to_python<std::shared_ptr<StateNumDiff> >();

  bp::class_<StateNumDiff, bp::bases<StateAbstract> >(
      "StateNumDiff",
      "Abstract class for computing Jdiff and Jintegrate by using numerical "
      "differentiation.\n\n",
      bp::init<std::shared_ptr<StateAbstract> >(
          bp::args("self", "state"),
          "Initialize the state numdiff.\n\n"
          ":param model: state where we compute the derivatives through "
          "numerial differentiation"))
      .def("zero", &StateNumDiff::zero, bp::args("self"),
           "Return a zero reference state.\n\n"
           ":return zero reference state")
      .def("rand", &StateNumDiff::rand, bp::args("self"),
           "Return a random reference state.\n\n"
           ":return random reference state")
      .def("diff", &StateNumDiff::diff_dx, bp::args("self", "x0", "x1"),
           "Operator that differentiates the two state points.\n\n"
           "It returns the value of x1 [-] x0 operation. Due to a state vector "
           "lies in\n"
           "the Euclidean space, this operator is defined with arithmetic "
           "subtraction.\n"
           ":param x0: current state (dim state.nx()).\n"
           ":param x1: next state (dim state.nx()).\n"
           ":return x1 - x0 value (dim state.nx()).")
      .def("integrate", &StateNumDiff::integrate_x, bp::args("self", "x", "dx"),
           "Operator that integrates the current state.\n\n"
           "It returns the value of x [+] dx operation. Due to a state vector "
           "lies in\n"
           "the Euclidean space, this operator is defined with arithmetic "
           "addition.\n"
           "Futhermore there is no timestep here (i.e. dx = v*dt), note this "
           "if you're\n"
           "integrating a velocity v during an interval dt.\n"
           ":param x: current state (dim state.nx()).\n"
           ":param dx: displacement of the state (dim state.nx()).\n"
           ":return x + dx value (dim state.nx()).")
      .def(
          "Jdiff", &StateNumDiff::Jdiff_Js,
          Jdiffs(
              bp::args("self", "x0", "x1", "firstsecond"),
              "Compute the partial derivatives of arithmetic substraction.\n\n"
              "Both Jacobian matrices are represented throught an identity "
              "matrix, with the exception\n"
              "that the first partial derivatives (w.r.t. x0) has negative "
              "signed. By default, this\n"
              "function returns the derivatives of the first and second "
              "argument (i.e.\n"
              "firstsecond='both'). However we ask for a specific partial "
              "derivative by setting\n"
              "firstsecond='first' or firstsecond='second'.\n"
              ":param x0: current state (dim state.nx()).\n"
              ":param x1: next state (dim state.nx()).\n"
              ":param firstsecond: derivative w.r.t x0 or x1 or both\n"
              ":return the partial derivative(s) of the diff(x0, x1) function"))
      .def("Jintegrate", &StateNumDiff::Jintegrate_Js,
           Jintegrates(
               bp::args("self", "x", "dx", "firstsecond"),
               "Compute the partial derivatives of arithmetic addition.\n\n"
               "Both Jacobian matrices are represented throught an identity "
               "matrix. By default, this\n"
               "function returns the derivatives of the first and second "
               "argument (i.e.\n"
               "firstsecond='both'). However we ask for a specific partial "
               "derivative by setting\n"
               "firstsecond='first' or firstsecond='second'.\n"
               ":param x: current state (dim state.nx()).\n"
               ":param dx: displacement of the state (dim state.nx()).\n"
               ":param firstsecond: derivative w.r.t x or dx or both\n"
               ":return the partial derivative(s) of the integrate(x, dx) "
               "function"))
      .def("JintegrateTransport", &StateNumDiff::JintegrateTransport,
           bp::args("self", "x", "dx", "Jin", "firstsecond"),
           "Parallel transport from integrate(x, dx) to x.\n\n"
           "This function performs the parallel transportation of an input "
           "matrix whose columns\n"
           "are expressed in the tangent space at integrate(x, dx) to the "
           "tangent space at x point\n"
           ":param x: state point (dim. state.nx).\n"
           ":param dx: velocity vector (dim state.ndx).\n"
           ":param Jin: input matrix (number of rows = state.nv).\n"
           ":param firstsecond: derivative w.r.t x or dx")
      .add_property(
          "disturbance", bp::make_function(&StateNumDiff::get_disturbance),
          &StateNumDiff::set_disturbance,
          "disturbance constant used in the numerical differentiation")
      .def(CopyableVisitor<StateNumDiff>());
}

}  // namespace python
}  // namespace crocoddyl
