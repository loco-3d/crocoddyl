///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_RANDOM_GENERATOR_HPP_
#define CROCODDYL_RANDOM_GENERATOR_HPP_

#include <Eigen/Core>
#include <random>

namespace crocoddyl {
namespace unittest {

inline std::mt19937& get_random_generator() {
  static std::mt19937 rng;
  return rng;
}

template <typename IntType>
IntType random_int_in_range(IntType first = 0, IntType last = 10) {
  return std::uniform_int_distribution<IntType>(first,
                                                last)(get_random_generator());
}

template <typename RealType>
RealType random_real_in_range(RealType first = 0, RealType last = 1) {
  return std::uniform_real_distribution<RealType>(first,
                                                  last)(get_random_generator());
}

inline bool random_boolean() {
  return std::uniform_int_distribution<>(0, 1)(get_random_generator()) != 0;
}

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, 1> random_vector(
    const Eigen::Index size, const Scalar first = Scalar(-1.),
    const Scalar last = Scalar(1.)) {
  Eigen::Matrix<Scalar, Eigen::Dynamic, 1> out(size);
  for (Eigen::Index i = 0; i < size; ++i) {
    out[i] = random_real_in_range<Scalar>(first, last);
  }
  return out;
}

template <typename Scalar>
Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> random_matrix(
    const Eigen::Index rows, const Eigen::Index cols,
    const Scalar first = Scalar(-1.), const Scalar last = Scalar(1.)) {
  Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> out(rows, cols);
  for (Eigen::Index row = 0; row < rows; ++row) {
    for (Eigen::Index col = 0; col < cols; ++col) {
      out(row, col) = random_real_in_range<Scalar>(first, last);
    }
  }
  return out;
}

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_RANDOM_GENERATOR_HPP_
