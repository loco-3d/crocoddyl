///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_SOLVERS_BOX_QP_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_SOLVERS_BOX_QP_HPP_

#include "crocoddyl/core/solvers/box-qp.hpp"
#include "python/crocoddyl/utils/vector-converter.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeSolverBoxQP() {
  bp::class_<BoxQPSolution>("BoxQPSolution", "Solution data of the box QP.\n\n",
                            bp::init<Eigen::MatrixXd, Eigen::VectorXd, std::vector<size_t>, std::vector<size_t> >(
                                bp::args("self", "Hff_inv", "x", "free_idx", "clamped_idx"),
                                "Initialize the data for the box-QP solution.\n\n"
                                ":param Hff_inv: inverse of the free Hessian\n"
                                ":param x: decision variable\n"
                                ":param free_idx: free indexes\n"
                                ":param clamped_idx: clamped indexes"))
      .add_property("Hff_inv",
                    bp::make_getter(&BoxQPSolution::Hff_inv, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&BoxQPSolution::Hff_inv), "inverse of the free Hessian matrix")
      .add_property("x", bp::make_getter(&BoxQPSolution::x, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&BoxQPSolution::x), "decision variable")
      .add_property("free_idx",
                    bp::make_getter(&BoxQPSolution::free_idx, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&BoxQPSolution::free_idx), "free indexes")
      .add_property("clamped_idx",
                    bp::make_getter(&BoxQPSolution::clamped_idx, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&BoxQPSolution::clamped_idx), "clamped indexes");

  bp::class_<BoxQP>("BoxQP",
                    "Projected-Newton QP for only bound constraints.\n\n"
                    "It solves a QP problem with bound constraints of the form:\n"
                    "    x = argmin 0.5 x^T H x + q^T x\n"
                    "    subject to:   lb <= x <= ub"
                    "where nx is the number of decision variables.",
                    bp::init<std::size_t, std::size_t, double, double, double>(
                        bp::args("self", "nx", "maxiter", "th_acceptstep", "th_grad", "reg"),
                        "Initialize the Projected-Newton QP for bound constraints.\n\n"
                        ":param maxiter: maximum number of allowed iterations (default 100)\n"
                        ":param th_acceptstep: acceptance step condition (default 0.1)\n"
                        ":param th_grad: gradient tolerance condition (default 1e-9)\n"
                        ":param reg: regularization (default 1e-9)"))
      .def("solve", &BoxQP::solve, bp::return_value_policy<bp::return_by_value>(),
           bp::args("H", "q", "lb", "ub", "xinit"),
           "Compute the solution of bound-constrained QP based on Newton projection.\n\n"
           ":param H: Hessian (dimension nx * nx)\n"
           ":param q: gradient (dimension nx)\n"
           ":param lb: lower bound (dimension nx)\n"
           ":param ub: upper bound (dimension nx)\n"
           ":param xinit: initial guess");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_SOLVERS_BOX_QP_HPP_
