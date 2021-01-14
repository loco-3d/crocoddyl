///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/state-base.hpp"

namespace crocoddyl {
namespace python {

void exposeStateAbstract() {
  bp::register_ptr_to_python<boost::shared_ptr<StateAbstract> >();

  bp::enum_<Jcomponent>("Jcomponent")
      .value("both", both)
      .value("first", first)
      .export_values()
      .value("second", second);

  bp::enum_<AssignmentOp>("AssignmentOp")
      .value("setto", setto)
      .value("addto", addto)
      .value("rmfrom", rmfrom)
      .export_values();

  bp::class_<StateAbstract_wrap, boost::noncopyable>(
      "StateAbstract",
      "Abstract class for the state representation.\n\n"
      "A state is represented by its operators: difference, integrates and their derivatives.\n"
      "The difference operator returns the value of x1 [-] x0 operation. Instead the integrate\n"
      "operator returns the value of x [+] dx. These operators are used to compared two points\n"
      "on the state manifold M or to advance the state given a tangential velocity (Tx M).\n"
      "Therefore the points x, x0 and x1 belong to the manifold M; and dx or x1 [-] x0 lie\n"
      "on its tangential space.",
      bp::init<int, int>(bp::args("self", "nx", "ndx"),
                         "Initialize the state dimensions.\n\n"
                         ":param nx: dimension of state configuration tuple\n"
                         ":param ndx: dimension of state tangent vector"))
      .def("zero", pure_virtual(&StateAbstract_wrap::zero), bp::args("self"),
           "Generate a zero reference state.\n\n"
           ":return zero reference state")
      .def("rand", pure_virtual(&StateAbstract_wrap::rand), bp::args("self"),
           "Generate a random reference state.\n\n"
           ":return random reference state")
      .def("diff", pure_virtual(&StateAbstract_wrap::diff_wrap), bp::args("self", "x0", "x1"),
           "Compute the state manifold differentiation.\n\n"
           "It returns the value of x1 [-] x0 operation. Note tha x0 and x1 are points in the state\n"
           "manifold (in M). Instead the operator result lies in the tangent-space of M.\n"
           ":param x0: previous state point (dim state.nx).\n"
           ":param x1: current state point (dim state.nx).\n"
           ":return x1 [-] x0 value (dim state.ndx).")
      .def("integrate", pure_virtual(&StateAbstract_wrap::integrate), bp::args("self", "x", "dx"),
           "Compute the state manifold integration.\n\n"
           "It returns the value of x [+] dx operation. x and dx are points in the state.diff(x0,x1) (in M)\n"
           "and its tangent, respectively. Note that the operator result lies on M too.\n"
           ":param x: state point (dim state.nx).\n"
           ":param dx: velocity vector (dim state.ndx).\n"
           ":return x [+] dx value (dim state.nx).")
      .def("Jdiff", pure_virtual(&StateAbstract_wrap::Jdiff_wrap), bp::args("self", "x0", "x1", "firstsecond"),
           "Compute the partial derivatives of difference operator.\n\n"
           "The difference operator (x1 [-] x0) is defined by diff(x0, x1). Instead Jdiff\n"
           "computes its partial derivatives, i.e. \\partial{diff(x0, x1)}{x0} and\n"
           "\\partial{diff(x0, x1)}{x1}. By default, this function returns the derivatives of the\n"
           "first and second argument (i.e. firstsecond='both'). However we can also specific the\n"
           "partial derivative for the first and second variables by setting firstsecond='first'\n"
           "or firstsecond='second', respectively.\n"
           ":param x0: previous state point (dim state.nx).\n"
           ":param x1: current state point (dim state.nx).\n"
           ":param firstsecond: desired partial derivative\n"
           ":return the partial derivative(s) of the diff(x0, x1) function")
      .def("Jintegrate", pure_virtual(&StateAbstract_wrap::Jintegrate_wrap),
           bp::args("self", "x", "dx", "firstsecond"),
           "Compute the partial derivatives of integrate operator.\n\n"
           "The integrate operator (x [+] dx) is defined by integrate(x, dx). Instead Jintegrate\n"
           "computes its partial derivatives, i.e. \\partial{integrate(x, dx)}{x} and\n"
           "\\partial{integrate(x, dx)}{dx}. By default, this function returns the derivatives of\n"
           "the first and second argument (i.e. firstsecond='both').\n"
           "partial derivative by setting firstsecond='first' or firstsecond='second'.\n"
           ":param x: state point (dim state.nx).\n"
           ":param dx: velocity vector (dim state.ndx).\n"
           ":param firstsecond: desired partial derivative\n"
           ":return the partial derivative(s) of the integrate(x, dx) function")
      .add_property("nx",
                    bp::make_function(&StateAbstract_wrap::get_nx, bp::return_value_policy<bp::return_by_value>()),
                    "dimension of state tuple")
      .add_property("ndx",
                    bp::make_function(&StateAbstract_wrap::get_ndx, bp::return_value_policy<bp::return_by_value>()),
                    "dimension of the tangent space of the state manifold")
      .add_property("nq",
                    bp::make_function(&StateAbstract_wrap::get_nq, bp::return_value_policy<bp::return_by_value>()),
                    "dimension of the configuration tuple")
      .add_property("nv",
                    bp::make_function(&StateAbstract_wrap::get_nv, bp::return_value_policy<bp::return_by_value>()),
                    "dimension of tangent space of the configuration manifold")
      .add_property(
          "has_limits",
          bp::make_function(&StateAbstract_wrap::get_has_limits, bp::return_value_policy<bp::return_by_value>()),
          "indicates whether problem has finite state limits")
      .add_property("lb", bp::make_getter(&StateAbstract_wrap::lb_, bp::return_internal_reference<>()),
                    &StateAbstract_wrap::set_lb, "lower state limits")
      .add_property("ub", bp::make_getter(&StateAbstract_wrap::ub_, bp::return_internal_reference<>()),
                    &StateAbstract_wrap::set_ub, "upper state limits");
}

}  // namespace python
}  // namespace crocoddyl
