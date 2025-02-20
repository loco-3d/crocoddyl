///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024-2025, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_UTILS_CONVERSIONS_HPP_
#define CROCODDYL_UTILS_CONVERSIONS_HPP_

#include <vector>

#include "crocoddyl/core/mathbase.hpp"

namespace crocoddyl {

template <typename Scalar>
static Scalar scalar_cast(const double& x) {
  return static_cast<Scalar>(x);
}

template <typename Scalar>
static Scalar scalar_cast(const float& x) {
  return static_cast<Scalar>(x);
}

template <typename NewScalar, typename Scalar,
          template <typename> class ItemTpl>
std::vector<ItemTpl<NewScalar>> vector_cast(
    const std::vector<ItemTpl<Scalar>>& in) {
  std::vector<ItemTpl<NewScalar>> out;
  out.reserve(in.size());  // Optimize allocation
  for (const auto& obj : in) {
    out.push_back(obj.template cast<NewScalar>());
  }
  return out;
}

}  // namespace crocoddyl

#ifdef CROCODDYL_WITH_CODEGEN

// Specialize Eigen's internal cast_impl for your specific types
namespace Eigen {
namespace internal {

template <>
struct cast_impl<CppAD::AD<CppAD::cg::CG<double>>, float> {
  EIGEN_DEVICE_FUNC static inline float run(
      const CppAD::AD<CppAD::cg::CG<double>>& x) {
    // Perform the conversion. This example extracts the value from the AD type.
    // You might need to adjust this depending on the specific implementation of
    // CppAD::cg::CG<double>.
    return static_cast<float>(CppAD::Value(x).getValue());
  }
};

template <>
struct cast_impl<CppAD::AD<CppAD::cg::CG<double>>, double> {
  EIGEN_DEVICE_FUNC static inline double run(
      const CppAD::AD<CppAD::cg::CG<double>>& x) {
    return CppAD::Value(x).getValue();
  }
};

template <>
struct cast_impl<CppAD::AD<CppAD::cg::CG<float>>, float> {
  EIGEN_DEVICE_FUNC static inline float run(
      const CppAD::AD<CppAD::cg::CG<float>>& x) {
    return CppAD::Value(x).getValue();
  }
};

template <>
struct cast_impl<CppAD::AD<CppAD::cg::CG<float>>, double> {
  EIGEN_DEVICE_FUNC static inline double run(
      const CppAD::AD<CppAD::cg::CG<float>>& x) {
    // Perform the conversion. This example extracts the value from the AD type.
    // You might need to adjust this depending on the specific implementation of
    // CppAD::cg::CG<float>.
    return static_cast<float>(CppAD::Value(x).getValue());
  }
};

}  // namespace internal
}  // namespace Eigen

namespace crocoddyl {

template <typename Scalar>
static inline Scalar scalar_cast(const CppAD::AD<CppAD::cg::CG<double>>& x) {
  return static_cast<Scalar>(CppAD::Value(x).getValue());
}

template <typename Scalar>
static inline Scalar scalar_cast(const CppAD::AD<CppAD::cg::CG<float>>& x) {
  return static_cast<Scalar>(CppAD::Value(x).getValue());
}

}  // namespace crocoddyl

#endif

#endif  // CROCODDYL_UTILS_CONVERSIONS_HPP_
