///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_UTILS_MATH_HPP_
#define CROCODDYL_CORE_UTILS_MATH_HPP_

#include <boost/type_traits.hpp>

#include <Eigen/Dense>
#include <algorithm>
#include <limits>

//fwd

template<typename MatrixLike, bool value = boost::is_floating_point<typename MatrixLike::Scalar>::value>
struct pseudoInverseAlgo
{
  typedef typename MatrixLike::Scalar Scalar;
  typedef typename MatrixLike::RealScalar RealScalar;
  
  static MatrixLike run(const Eigen::MatrixBase<MatrixLike>& a, const RealScalar& epsilon)
  {
    using std::max;
    Eigen::JacobiSVD<MatrixLike> svd(a, Eigen::ComputeThinU | Eigen::ComputeThinV);
    RealScalar tolerance =
      epsilon * static_cast<Scalar>(max(a.cols(), a.rows())) * svd.singularValues().array().abs()(0);
    return svd.matrixV() *
      (svd.singularValues().array().abs() > tolerance)
      .select(svd.singularValues().array().inverse(), 0)
      .matrix()
      .asDiagonal() *
      svd.matrixU().adjoint();
  }
};

template<typename MatrixLike>
struct pseudoInverseAlgo<MatrixLike,false>
{
  typedef typename MatrixLike::Scalar Scalar;
  typedef typename MatrixLike::RealScalar RealScalar;
  
  static MatrixLike run(const Eigen::MatrixBase<MatrixLike>& a,
                                           const RealScalar & epsilon)
  {
    return Eigen::MatrixBase<MatrixLike>::Zero(a.rows(), a.cols());
  }
};

template<typename MatrixLike>
MatrixLike pseudoInverse(const Eigen::MatrixBase<MatrixLike> & a,
                                            const typename MatrixLike::RealScalar& epsilon =
                                            Eigen::NumTraits<typename MatrixLike::Scalar>::dummy_precision())
{
  return pseudoInverseAlgo<MatrixLike>::run(a, epsilon);
}


#endif  // CROCODDYL_CORE_UTILS_MATH_HPP_
