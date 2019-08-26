///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_NUMDIFF_ACTION_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_NUMDIFF_ACTION_HPP_

#include "crocoddyl/core/numdiff/action.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeActionNumDiff() {
  // bp::class_<ActionModelNumDiff, boost::noncopyable>(
  //     "ActionModelNumDiff",
  //     "Abstract class for computing calcDiff by using numerical differentiation.\n\n" bp::init<ActionModelAbstract&,
  //                                                                                              bp::optional<bool> >(
  //         bp::args(" self", " model", " gaussApprox=False"),
  //         "Initialize the action model NumDiff.\n\n"
  //         ":param model: action model to NumDiff,\n"
  //         ":param gaussApprox: compute the Hessian using Gauss approximation")[bp::with_custodian_and_ward<1, 2>()])
  //     .def("calc", &ActionModelNumDiff::calc, bp::args(" self", " data", " x", " u"),
  //          "Compute the next state and cost value.\n\n"
  //          "It describes the time-discrete evolution of our dynamical system\n"
  //          "in which we obtain the next state. Additionally it computes the\n"
  //          "cost value associated to this discrete state and control pair.\n"
  //          ":param data: action data\n"
  //          ":param x: time-discrete state vector\n"
  //          ":param u: time-discrete control input")
  //     .def("calcDiff", &ActionModelNumDiff::calcDiff, bp::args(" self", " data", " x", " u", " recalc=True"),
  //          "Compute the derivatives of the dynamics and cost functions.\n\n"
  //          "It computes the partial derivatives of the dynamical system and the\n"
  //          "cost function. If recalc == True, it first updates the state evolution\n"
  //          "and cost value. This function builds a quadratic approximation of the\n"
  //          "action model (i.e. linear dynamics and quadratic cost).\n"
  //          ":param data: action data\n"
  //          ":param x: time-discrete state vector\n"
  //          ":param u: time-discrete control input\n"
  //          ":param recalc: If true, it updates the state evolution and the cost value.")
  //     .def("createData", &ActionModelNumDiff::createData, bp::args(" self"),
  //          "Create the action data.\n\n"
  //          "Each action model (AM) has its own data that needs to be allocated.\n"
  //          "This function returns the allocated data for a predefined AM.\n"
  //          ":return AM data.")
  //     .add_property("model", bp::make_function(&ActionModelNumDiff::get_model, bp::return_internal_reference<>()),
  //                   "action model");
}  // namespace python
}  // namespace python

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_NUMDIFF_ACTION_HPP_
