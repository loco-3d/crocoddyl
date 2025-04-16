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

#ifdef CROCODDYL_WITH_CODEGEN
#include <cppad/cg/support/cppadcg_eigen.hpp>
#include <cppad/cppad.hpp>
#endif

#include "crocoddyl/core/mathbase.hpp"

namespace crocoddyl {

template <typename Scalar>
struct ScalarSelector {
  typedef typename std::conditional<std::is_floating_point<Scalar>::value,
                                    Scalar, double>::type type;
};

// Casting between floating-point types
template <typename NewScalar, typename Scalar>
static typename std::enable_if<std::is_floating_point<NewScalar>::value &&
                                   std::is_floating_point<Scalar>::value,
                               NewScalar>::type
scalar_cast(const Scalar& x) {
  return static_cast<NewScalar>(x);
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

template <typename NewScalar, typename Scalar,
          template <typename> class ItemTpl>
std::vector<std::shared_ptr<ItemTpl<NewScalar>>> vector_cast(
    const std::vector<std::shared_ptr<ItemTpl<Scalar>>>& in) {
  std::vector<std::shared_ptr<ItemTpl<NewScalar>>> out;
  out.reserve(in.size());  // Optimize allocation
  for (const auto& obj : in) {
    out.push_back(std::static_pointer_cast<ItemTpl<NewScalar>>(
        obj->template cast<NewScalar>()));
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

// Convert from CppAD::AD<CppAD::cg::CG<float>> to
// CppAD::AD<CppAD::cg::CG<double>>
template <>
struct cast_impl<CppAD::AD<CppAD::cg::CG<float>>,
                 CppAD::AD<CppAD::cg::CG<double>>> {
  EIGEN_DEVICE_FUNC static inline CppAD::AD<CppAD::cg::CG<double>> run(
      const CppAD::AD<CppAD::cg::CG<float>>& x) {
    return CppAD::AD<CppAD::cg::CG<double>>(
        CppAD::cg::CG<double>(CppAD::Value(x).getValue()));
  }
};

// Convert from CppAD::AD<CppAD::cg::CG<double>> to
// CppAD::AD<CppAD::cg::CG<float>>
template <>
struct cast_impl<CppAD::AD<CppAD::cg::CG<double>>,
                 CppAD::AD<CppAD::cg::CG<float>>> {
  EIGEN_DEVICE_FUNC static inline CppAD::AD<CppAD::cg::CG<float>> run(
      const CppAD::AD<CppAD::cg::CG<double>>& x) {
    return CppAD::AD<CppAD::cg::CG<float>>(
        CppAD::cg::CG<float>(static_cast<float>(CppAD::Value(x).getValue())));
  }
};

}  // namespace internal
}  // namespace Eigen

namespace crocoddyl {

// Casting to CppAD types from floating-point types
template <typename NewScalar, typename Scalar>
static typename std::enable_if<
    std::is_floating_point<Scalar>::value &&
        (std::is_same<NewScalar, CppAD::AD<CppAD::cg::CG<double>>>::value ||
         std::is_same<NewScalar, CppAD::AD<CppAD::cg::CG<float>>>::value),
    NewScalar>::type
scalar_cast(const Scalar& x) {
  return static_cast<NewScalar>(x);
}

// Casting to floating-point types from CppAD types
template <typename NewScalar, typename Scalar>
static inline typename std::enable_if<std::is_floating_point<Scalar>::value,
                                      NewScalar>::type
scalar_cast(const CppAD::AD<CppAD::cg::CG<Scalar>>& x) {
  return static_cast<NewScalar>(CppAD::Value(x).getValue());
}

}  // namespace crocoddyl

#endif

#endif  // CROCODDYL_UTILS_CONVERSIONS_HPP_
