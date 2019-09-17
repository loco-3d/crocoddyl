///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_UTILS_MATH_HPP_
#define CROCODDYL_CORE_UTILS_MATH_HPP_

#include <Eigen/Dense>
#include <algorithm>
#include <limits>

template <typename MatrixType>
MatrixType pseudoInverse(const MatrixType& a, double epsilon = std::numeric_limits<double>::epsilon()) {
  Eigen::JacobiSVD<MatrixType> svd(a, Eigen::ComputeThinU | Eigen::ComputeThinV);
  double tolerance =
      epsilon * static_cast<double>(std::max(a.cols(), a.rows())) * svd.singularValues().array().abs()(0);
  return svd.matrixV() *
         (svd.singularValues().array().abs() > tolerance)
             .select(svd.singularValues().array().inverse(), 0)
             .matrix()
             .asDiagonal() *
         svd.matrixU().adjoint();
}

#endif  // CROCODDYL_CORE_UTILS_MATH_HPP_
