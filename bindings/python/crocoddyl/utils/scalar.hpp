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

namespace crocoddyl {
namespace python {

typedef double Float64;
typedef float Float32;
#ifdef CROCODDYL_WITH_CODEGEN
typedef CppAD::cg::CG<Float64> CGFloat64;
typedef CppAD::AD<CGFloat64> ADFloat64;
#define CROCODDYL_PYTHON_SCALARS(macro)                             \
  macro(Float64) {                                                  \
    bp::scope float_scope = getOrCreatePythonNamespace("scalar32"); \
    macro(Float32)                                                  \
  }                                                                 \
  {                                                                 \
    bp::scope cg_scope = getOrCreatePythonNamespace("cgscalar");    \
    macro(ADScalar64)                                               \
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
