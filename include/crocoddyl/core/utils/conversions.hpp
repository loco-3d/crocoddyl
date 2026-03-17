///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_UTILS_CONVERSIONS_HPP_
#define CROCODDYL_UTILS_CONVERSIONS_HPP_

#include <memory>
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

#ifdef CROCODDYL_WITH_CODEGEN

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

#endif  // CROCODDYL_WITH_CODEGEN

template <typename NewScalar, typename Scalar>
std::vector<NewScalar> vector_cast(const std::vector<Scalar>& in) {
  std::vector<NewScalar> out;
  out.reserve(in.size());  // Optimize allocation
  for (const auto& obj : in) {
    out.push_back(scalar_cast<NewScalar>(obj));
  }
  return out;
}

template <typename NewScalar, typename Scalar, int Rows, int Cols, int Options,
          int MaxRows, int MaxCols>
std::vector<Eigen::Matrix<NewScalar, Rows, Cols, Options, MaxRows, MaxCols>>
vector_cast(const std::vector<
            Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>>& in) {
  std::vector<Eigen::Matrix<NewScalar, Rows, Cols, Options, MaxRows, MaxCols>>
      out;
  out.reserve(in.size());  // Optimize allocation
  for (const auto& obj : in) {
    out.push_back(obj.template cast<NewScalar>());
  }
  return out;
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

template <typename Scalar, int Rows, int Cols, int Options, int MaxRows,
          int MaxCols>
Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols> matrix_random_cast(
    const Eigen::Index rows, const Eigen::Index cols) {
  using MatrixDouble =
      Eigen::Matrix<double, Rows, Cols, Options, MaxRows, MaxCols>;
  return MatrixDouble::Random(rows, cols).template cast<Scalar>();
}

template <typename Scalar, int Rows, int Options, int MaxRows>
Eigen::Matrix<Scalar, Rows, 1, Options, MaxRows, 1> vector_random_cast(
    const Eigen::Index size) {
  using VectorDouble = Eigen::Matrix<double, Rows, 1, Options, MaxRows, 1>;
  return VectorDouble::Random(size).template cast<Scalar>();
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

#endif

#endif  // CROCODDYL_UTILS_CONVERSIONS_HPP_
