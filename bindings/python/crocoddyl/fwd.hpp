///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_FWD_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_FWD_HPP_

#include <eigenpy/eigenpy.hpp>
#include <boost/python.hpp>
#include <boost/python/enum.hpp>

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeCore();
void exposeMultibody();

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_FWD_HPP_