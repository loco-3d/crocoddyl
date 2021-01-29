///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/data-collector-base.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

void exposeDataCollector() {
  bp::class_<DataCollectorAbstract, boost::noncopyable>(
      "DataCollectorAbstract",
      "Abstract class for common collection of data used in different objects "
      "in action model.\n\n",
      bp::init<>(bp::args("self")));
}

} // namespace python
} // namespace crocoddyl
