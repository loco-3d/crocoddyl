///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_RANDOM_GENERATOR_HPP_
#define CROCODDYL_RANDOM_GENERATOR_HPP_

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

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_RANDOM_GENERATOR_HPP_
