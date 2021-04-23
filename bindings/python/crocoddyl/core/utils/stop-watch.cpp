//#include <boost/python.hpp>
//#include <iostream>
//#include <string>
//#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
//#include <boost/python/make_constructor.hpp>

//#include "crocoddyl/bindings/python/stop_watch.hpp"
//#include "crocoddyl/core/utils/stop-watch.hpp"

///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "crocoddyl/core/utils/stop-watch.hpp"

namespace bp = boost::python;
using namespace boost::python;

namespace crocoddyl {
namespace python {

void stop_watch_report(int precision)
{
  getProfiler().report_all(precision);
}

long double stop_watch_get_average_time(const std::string & perf_name)
{
  return getProfiler().get_average_time(perf_name);
}

/** Returns minimum execution time of a certain performance */
long double stop_watch_get_min_time(const std::string & perf_name)
{
  return getProfiler().get_min_time(perf_name);
}

/** Returns maximum execution time of a certain performance */
long double stop_watch_get_max_time(const std::string & perf_name)
{
  return getProfiler().get_max_time(perf_name);
}

long double stop_watch_get_total_time(const std::string & perf_name)
{
  return getProfiler().get_total_time(perf_name);
}

void stop_watch_reset_all()
{
  getProfiler().reset_all();
}

void exposeStopWatch()
{
    bp::def("stop_watch_report", stop_watch_report,
            "Report all the times measured by the shared stop-watch.");

    bp::def("stop_watch_get_average_time", stop_watch_get_average_time,
            "Get the average time measured by the shared stop-watch for the specified task.");

    bp::def("stop_watch_get_min_time", stop_watch_get_min_time,
            "Get the min time measured by the shared stop-watch for the specified task.");

    bp::def("stop_watch_get_max_time", stop_watch_get_max_time,
            "Get the max time measured by the shared stop-watch for the specified task.");

    bp::def("stop_watch_get_total_time", stop_watch_get_total_time,
            "Get the total time measured by the shared stop-watch for the specified task.");

    bp::def("stop_watch_reset_all", stop_watch_reset_all,
            "Reset the shared stop-watch.");
}

}  // namespace python
}  // namespace crocoddyl
