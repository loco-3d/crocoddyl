// BSD 3-Clause License
//
// Copyright (C) 2024, LAAS-CNRS.
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.

#include "colmpc/python.hpp"

BOOST_PYTHON_MODULE(colmpc) {
  namespace bp = boost::python;

  bp::import("pinocchio");
  bp::import("crocoddyl");
  // Enabling eigenpy support, i.e. numpy/eigen compatibility.
  eigenpy::enableEigenPy();
  eigenpy::enableEigenPySpecific<Eigen::VectorXi>();
  colmpc::python::exposeStateMultibody();
  colmpc::python::exposeDynamicFreeFwd();
  colmpc::python::exposeActivationModelQuadExp();
  colmpc::python::exposeResidualDistanceCollision();
  colmpc::python::exposeResidualDistanceCollision2();
  colmpc::python::exposeResidualVelocityAvoidance();
  colmpc::python::exposeActivationModelDistanceQuad();
}
