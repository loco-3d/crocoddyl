///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/fwd.hpp"
#include "crocoddyl/core/utils/version.hpp"
#include "python/crocoddyl/utils/vector-converter.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

BOOST_PYTHON_MODULE(libcrocoddyl_pywrap) {
  bp::scope().attr("__version__") = printVersion();

  eigenpy::enableEigenPy();

  typedef double Scalar;
  typedef Eigen::Matrix<Scalar, 6, 1> Vector6;
  // typedef Eigen::Matrix<Scalar, 4, 6> Matrix46;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 3> MatrixX3;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixX;

  eigenpy::enableEigenPySpecific<Vector6>();
  // eigenpy::enableEigenPySpecific<Matrix46>();
  eigenpy::enableEigenPySpecific<MatrixX3>();

  // Register converters between std::vector and Python list
  StdVectorPythonVisitor<VectorX, std::allocator<VectorX>, true>::expose("StdVec_VectorX");
  StdVectorPythonVisitor<MatrixX, std::allocator<MatrixX>, true>::expose("StdVec_MatrixX");

  exposeCore();
  exposeMultibody();
}

}  // namespace python
}  // namespace crocoddyl
