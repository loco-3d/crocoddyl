///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, LAAS-CNRS, University of Edinburgh, INRIA,
//                          Heriot-Watt University
// University of Oxford Copyright note valid unless otherwise stated in
// individual files. All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/version.hpp"
#include "python/crocoddyl/fwd.hpp"
#include "python/crocoddyl/utils/set-converter.hpp"
#include "python/crocoddyl/utils/vector-converter.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

BOOST_PYTHON_MODULE(libcrocoddyl_pywrap) {
  bp::scope().attr("__version__") = crocoddyl::printVersion();
  bp::scope().attr("__raw_version__") = bp::str(CROCODDYL_VERSION);

  eigenpy::enableEigenPy();

  typedef double Scalar;
  typedef Eigen::Matrix<Scalar, 4, 1> Vector4;
  typedef Eigen::Matrix<Scalar, 6, 1> Vector6;
  typedef Eigen::Matrix<Scalar, 4, 6> Matrix46;
  typedef Eigen::Matrix<Scalar, 6, Eigen::Dynamic> Matrix6x;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 3> MatrixX3;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorX;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixX;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      RowMatrixX;

  eigenpy::enableEigenPySpecific<Vector4>();
  eigenpy::enableEigenPySpecific<Vector6>();
  eigenpy::enableEigenPySpecific<Matrix46>();
  eigenpy::enableEigenPySpecific<MatrixX3>();
  eigenpy::enableEigenPySpecific<Matrix6x>();

  // Register converters between std::vector and Python list
  StdVectorPythonVisitor<std::vector<VectorX>, true>::expose("StdVec_VectorX");
  StdVectorPythonVisitor<std::vector<MatrixX>, true>::expose("StdVec_MatrixX");
  StdVectorPythonVisitor<std::vector<RowMatrixX>, true>::expose(
      "StdVec_RowMatrixX");

  // Register converters between std::set and Python set
  StdSetPythonVisitor<std::string, std::less<std::string>,
                      std::allocator<std::string>,
                      true>::expose("StdSet_String");

  exposeCore();
  exposeMultibody();
}

}  // namespace python
}  // namespace crocoddyl
