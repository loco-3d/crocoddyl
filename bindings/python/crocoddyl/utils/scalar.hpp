///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2024-2025, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_UTILS_SCALAR_HPP_
#define BINDINGS_PYTHON_CROCODDYL_UTILS_SCALAR_HPP_

#include <boost/python.hpp>

#include "python/crocoddyl/utils/namespace.hpp"

#ifdef CROCODDYL_WITH_CODEGEN
#include <cppad/cg/support/cppadcg_eigen.hpp>
#include <cppad/cppad.hpp>
#endif

#ifdef CROCODDYL_WITH_CODEGEN
#include <cppad/cg/cg.hpp>
#endif

namespace crocoddyl {
namespace python {

typedef double Float64;
typedef float Float32;

enum class DType {
  Float64,
  Float32
#ifdef CROCODDYL_WITH_CODEGEN
  ,
  ADFloat64
#endif
};

#define CROCODDYL_PYTHON_FLOATINGPOINT_SCALARS(macro)               \
  {                                                                 \
    macro(Float64)                                                  \
  }                                                                 \
  {                                                                 \
    bp::scope float_scope = getOrCreatePythonNamespace("scalar32"); \
    macro(Float32)                                                  \
  }

#ifdef CROCODDYL_WITH_CODEGEN
typedef CppAD::cg::CG<Float64> CGFloat64;
typedef CppAD::AD<CGFloat64> ADFloat64;
#define CROCODDYL_PYTHON_SCALARS(macro)                             \
  {                                                                 \
    macro(Float64)                                                  \
  }                                                                 \
  {                                                                 \
    bp::scope float_scope = getOrCreatePythonNamespace("scalar32"); \
    macro(Float32)                                                  \
  }                                                                 \
  {                                                                 \
    bp::scope cg_scope = getOrCreatePythonNamespace("cgscalar");    \
    macro(ADFloat64)                                                \
  }
#else
#define CROCODDYL_PYTHON_SCALARS(macro)                             \
  {                                                                 \
    macro(Float64)                                                  \
  }                                                                 \
  {                                                                 \
    bp::scope float_scope = getOrCreatePythonNamespace("scalar32"); \
    macro(Float32)                                                  \
  }
#endif

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_UTILS_SCALAR_HPP_
