///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020 LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_MATHBASE_HPP_
#define CROCODDYL_CORE_MATHBASE_HPP_

#include <Eigen/Core>

namespace crocoddyl {

template <typename _Scalar>
struct MathBaseTpl {
public:

  typedef _Scalar Scalar;
  typedef Eigen::Matrix<Scalar, 2, 1> Vector2s;
  typedef Eigen::Matrix<Scalar, 2, 1> Vector3s;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;
  typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> ArrayXs;
};

}

#endif
