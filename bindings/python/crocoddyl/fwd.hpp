///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_FWD_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_FWD_HPP_

#include <eigenpy/eigenpy.hpp>

#ifdef PNINQP_WITH_CODEGEN
#include <pycppad/cppad.hpp>
#endif

#include "python/crocoddyl/utils/cast.hpp"
#include "python/crocoddyl/utils/copyable.hpp"
#include "python/crocoddyl/utils/printable.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeCore();
void exposeMultibody();

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_FWD_HPP_
