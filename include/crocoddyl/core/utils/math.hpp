///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_UTILS_MATH_HPP_
#define CROCODDYL_CORE_UTILS_MATH_HPP_

#include <Eigen/Dense>
#include <algorithm>
#include <boost/type_traits.hpp>
#include <limits>

#ifdef CROCODDYL_WITH_CODEGEN

#include <cmath>
#include <cppad/cg/cg.hpp>
#include <cppad/cppad.hpp>
#include <type_traits>

namespace CppAD {
template <class Scalar>
bool isfinite(const AD<CppAD::cg::CG<Scalar>>& x) {
  return std::isfinite(static_cast<Scalar>(CppAD::Value(x).getValue()));
}
}  // namespace CppAD

namespace crocoddyl {

// Case 1: Use std::pow for floating-point types
template <typename Scalar, typename ExpScalar>
typename std::enable_if<std::is_floating_point<Scalar>::value, Scalar>::type
pow(const Scalar& base, const ExpScalar& exponent) {
  return std::pow(base, exponent);
}

// Case 2: Use CppAD::pow for CppAD types
template <typename Scalar>
typename std::enable_if<!std::is_floating_point<Scalar>::value, Scalar>::type
pow(const Scalar& base, const Scalar& exponent) {
  return CppAD::pow(base, exponent);
}

// Case 1: Use std::pow for floating-point types
template <typename Scalar>
typename std::enable_if<std::is_floating_point<Scalar>::value, Scalar>::type
sqrt(const Scalar& base) {
  return std::sqrt(base);
}

// Case 2: Use CppAD::pow for CppAD types
template <typename Scalar>
typename std::enable_if<!std::is_floating_point<Scalar>::value, Scalar>::type
sqrt(const Scalar& base) {
  return CppAD::sqrt(base);
}

// Case 1: Use std::fabs for floating-point types
template <typename Scalar>
typename std::enable_if<std::is_floating_point<Scalar>::value, Scalar>::type
fabs(const Scalar& base) {
  return std::fabs(base);
}

// Case 2: Use CppAD::fabs for CppAD types
template <typename Scalar>
typename std::enable_if<!std::is_floating_point<Scalar>::value, Scalar>::type
fabs(const Scalar& base) {
  return CppAD::fabs(base);
}
}  // namespace crocoddyl

namespace Eigen {

// Overload for Eigen::pow with CppAD-compatible types
template <typename Derived>
auto pow(const Eigen::ArrayBase<Derived>& base, double exponent) {
  return base.unaryExpr([exponent](const typename Derived::Scalar& x) {
    return CppAD::pow(x, typename Derived::Scalar(exponent));
  });
}

// Overload for Eigen::sqrt with CppAD-compatible types
template <typename Derived>
typename std::enable_if<std::is_base_of<CppAD::cg::CG<typename Derived::Scalar>,
                                        typename Derived::Scalar>::value,
                        Eigen::Array<typename Derived::Scalar, Eigen::Dynamic,
                                     Eigen::Dynamic>>::type
sqrt(const Eigen::ArrayBase<Derived>& base) {
  return base.unaryExpr(
      [](const typename Derived::Scalar& x) { return CppAD::sqrt(x); });
}

}  // namespace Eigen

#endif

// fwd

template <typename MatrixLike,
          bool value =
              (Eigen::NumTraits<typename MatrixLike::Scalar>::IsInteger == 0)>
struct pseudoInverseAlgo {
  typedef typename MatrixLike::Scalar Scalar;
  typedef typename MatrixLike::RealScalar RealScalar;

  static MatrixLike run(const Eigen::MatrixBase<MatrixLike>& a,
                        const RealScalar& epsilon) {
    using std::max;
    Eigen::JacobiSVD<MatrixLike> svd(a,
                                     Eigen::ComputeThinU | Eigen::ComputeThinV);
    RealScalar tolerance = epsilon *
                           static_cast<Scalar>(max(a.cols(), a.rows())) *
                           svd.singularValues().array().abs()(0);
    // FIX: Replace select() with a lambda function
    Eigen::Matrix<typename MatrixLike::Scalar, Eigen::Dynamic, 1>
        invSingularValues =
            svd.singularValues().unaryExpr([&](const Scalar& x) {
              return (x > tolerance) ? Scalar(1) / x : Scalar(0);
            });
    return svd.matrixV() * invSingularValues.asDiagonal() *
           svd.matrixU().adjoint();
  }
};

template <typename MatrixLike>
struct pseudoInverseAlgo<MatrixLike, false> {
  typedef typename MatrixLike::Scalar Scalar;
  typedef typename MatrixLike::RealScalar RealScalar;

  static MatrixLike run(const Eigen::MatrixBase<MatrixLike>& a,
                        const RealScalar&) {
    return Eigen::MatrixBase<MatrixLike>::Zero(a.rows(), a.cols());
  }
};

template <typename MatrixLike>
MatrixLike pseudoInverse(
    const Eigen::MatrixBase<MatrixLike>& a,
    const typename MatrixLike::RealScalar& epsilon =
        Eigen::NumTraits<typename MatrixLike::Scalar>::dummy_precision()) {
  return pseudoInverseAlgo<MatrixLike>::run(a, epsilon);
}

#endif  // CROCODDYL_CORE_UTILS_MATH_HPP_
