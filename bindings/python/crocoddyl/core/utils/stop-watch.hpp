///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_STOP_WATCH_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_STOP_WATCH_HPP_

//#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/utils/stop-watch.hpp"

namespace crocoddyl {
namespace python {

void stop_watch_report(int precision);

long double stop_watch_get_average_time(const std::string & perf_name);

/** Returns minimum execution time of a certain performance */
long double stop_watch_get_min_time(const std::string & perf_name);

/** Returns maximum execution time of a certain performance */
long double stop_watch_get_max_time(const std::string & perf_name);

long double stop_watch_get_total_time(const std::string & perf_name);

void stop_watch_reset_all();

}  // namespace python
}  // namespace crocoddyl
