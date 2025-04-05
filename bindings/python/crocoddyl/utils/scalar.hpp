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

#include "crocoddyl/core/macros.hpp"
#include "python/crocoddyl/utils/namespace.hpp"

namespace crocoddyl {
namespace python {

enum class DType {
  Float64,
  Float32
#ifdef CROCODDYL_WITH_CODEGEN
  ,
  ADFloat64
#endif
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_UTILS_SCALAR_HPP_
