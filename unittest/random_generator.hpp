///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_RANDOM_GENERATOR_HPP_
#define CROCODDYL_RANDOM_GENERATOR_HPP_

#include <functional>
#include <random>

static std::mt19937 rng;

namespace crocoddyl {
namespace unittest {

template <typename IntType>
IntType random_int_in_range(IntType first = 0, IntType last = 10) {
  return std::uniform_int_distribution<IntType>(first, last)(rng);
}

template <typename RealType>
RealType random_real_in_range(RealType first = 0, RealType last = 1) {
  return std::uniform_real_distribution<RealType>(first, last)(rng);
}

bool random_boolean() {
  static auto generator = std::bind(std::uniform_int_distribution<>(0, 1),
                                    std::default_random_engine());
  return generator();
}

}  // namespace unittest
}  // namespace crocoddyl

#endif  // CROCODDYL_RANDOM_GENERATOR_HPP_
