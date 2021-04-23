/*
Copyright (c) 2010-2013 Tommaso Urli

Tommaso Urli    tommaso.urli@uniud.it   University of Udine

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/


#ifndef CROCODDYL_CORE_UTILS_STOPWATCH_H_
#define CROCODDYL_CORE_UTILS_STOPWATCH_H_

#include "crocoddyl/core/utils/Stdafx.hh"

#ifndef WIN32
/* The classes below are exported */
#pragma GCC visibility push(default)
#endif

// Uncomment the following line to activate the profiler
//#define PROFILER_ACTIVE

#ifdef PROFILER_ACTIVE
  #define START_PROFILER(name) getProfiler().start(name)
  #define STOP_PROFILER(name) getProfiler().stop(name)
#else
  #define START_PROFILER(name)
  #define STOP_PROFILER(name)
#endif

#define STOP_WATCH_MAX_NAME_LENGTH 60
#define STOP_WATCH_TIME_WIDTH 10

namespace crocoddyl {

// Generic stopwatch exception class
struct StopwatchException
{
public:
  StopwatchException(std::string error) : error(error) { }
  std::string error;
};


enum StopwatchMode
{
  NONE	    = 0,  // Clock is not initialized
  CPU_TIME  = 1,  // Clock calculates time ranges using ctime and CLOCKS_PER_SEC
  REAL_TIME = 2   // Clock calculates time by asking the operating system how
                  // much real time passed
};


/**
    @brief A class representing a stopwatch.

    @code
    Stopwatch swatch();
    @endcode

    The Stopwatch class can be used to measure execution time of code,
    algorithms, etc., // TODO: he Stopwatch can be initialized in two
    time-taking modes, CPU time and real time:

    @code
    swatch.set_mode(REAL_TIME);
    @endcode

    CPU time is the time spent by the processor on a certain piece of code,
    while real time is the real amount of time taken by a certain piece of
    code to execute (i.e. in general if you are doing hard work such as
    image or video editing on a different process the measured time will
    probably increase).

    How does it work? Basically, one wraps the code to be measured with the
    following method calls:

    @code
    swatch.start("My astounding algorithm");
    // Hic est code
    swatch.stop("My astounding algorithm");
    @endcode

    A string representing the code ID is provided so that nested portions of
    code can be profiled separately:

    @code
    swatch.start("My astounding algorithm");

    swatch.start("My astounding algorithm - Super smart init");
    // Initialization
    swatch.stop("My astounding algorithm - Super smart init");

    swatch.start("My astounding algorithm - Main loop");
    // Loop
    swatch.stop("My astounding algorithm - Main loop");

    swatch.stop("My astounding algorithm");
    @endcode

    Note: ID strings can be whatever you like, in the previous example I have
    used "My astounding algorithm - *" only to enforce the fact that the
    measured code portions are part of My astounding algorithm, but there's no
    connection between the three measurements.

    If the code for a certain task is scattered through different files or
    portions of the same file one can use the start-pause-stop method:

    @code
    swatch.start("Setup");
    // First part of setup
    swatch.pause("Setup");

    swatch.start("Main logic");
    // Main logic
    swatch.stop("Main logic");

    swatch.start("Setup");
    // Cleanup (part of the setup)
    swatch.stop("Setup");
    @endcode

    Finally, to report the results of the measurements just run:

    @code
    swatch.report("Code ID");
    @endcode

    Thou can also provide an additional std::ostream& parameter to report() to
    redirect the logging on a different output. Also, you can use the
    get_total/min/max/average_time() methods to get the individual numeric data,
    without all the details of the logging. You can also extend Stopwatch to
    implement your own logging syntax.

    To report all the measurements:

    @code
    swatch.report_all();
    @endcode

    Same as above, you can redirect the output by providing a std::ostream&
    parameter.

*/
class Stopwatch {
public:

  /** @brief Constructor */
  Stopwatch(StopwatchMode _mode=NONE);

  /** @brief Destructor */
  ~Stopwatch();

  /** @brief Tells if a performance with a certain ID exists */
  bool performance_exists(std::string perf_name);

  /** @brief Initialize stopwatch to use a certain time taking mode */
  void set_mode(StopwatchMode mode);

  /** @brief Start the stopwatch related to a certain piece of code */
  void start(const std::string &perf_name);

  /** @brief Stops the stopwatch related to a certain piece of code */
  void stop(const std::string &perf_name);

  /** @brief Stops the stopwatch related to a certain piece of code */
  void pause(const std::string &perf_name);

  /** @brief Reset a certain performance record */
  void reset(const std::string &perf_name);

  /** @brief Resets all the performance records */
  void reset_all();

  /** @brief Dump the data of a certain performance record */
  void report(const std::string &perf_name, int precision=2,
              std::ostream& output = std::cout);

  /** @brief Dump the data of all the performance records */
  void report_all(int precision=2, std::ostream& output = std::cout);

  /** @brief Returns total execution time of a certain performance */
  long double get_total_time(const std::string &perf_name);

  /** @brief Returns average execution time of a certain performance */
  long double get_average_time(const std::string &perf_name);

  /** @brief Returns minimum execution time of a certain performance */
  long double get_min_time(const std::string &perf_name);

  /** @brief Returns maximum execution time of a certain performance */
  long double get_max_time(const std::string &perf_name);

  /** @brief Return last measurement of a certain performance */
  long double get_last_time(const std::string &perf_name);

  /** @brief Return the time since the start of the last measurement of a given
      performance.
  */
  long double get_time_so_far(const std::string &perf_name);

  /** @brief Turn off clock, all the Stopwatch::* methods return without doing
        anything after this method is called. */
  void turn_off();

  /** @brief Turn on clock, restore clock operativity after a turn_off(). */
  void turn_on();

  /** @brief Take time, depends on mode. */
  long double take_time();

protected:

  /** @brief Struct to hold the performance data */
  struct PerformanceData {

    PerformanceData() :
      clock_start(0),
      total_time(0),
      min_time(0),
      max_time(0),
      last_time(0),
      paused(false),
      stops(0) {
    }

    long double	clock_start;  //!< Start time
    long double	total_time;   //!< Cumulative total time
    long double	min_time;     //!< Minimum time
    long double	max_time;     //!< Maximum time
    long double last_time;    //!< Last time
    bool paused;              //!< Tells if this performance has been paused, only for internal use
    int	stops;                //!< How many cycles have been this stopwatch executed?
  };

  bool active;                //!< Flag to hold the clock's status
  StopwatchMode mode;         //!< Time taking mode
  std::map<std::string, PerformanceData >* records_of; //!< Dynamic collection of performance data

};

Stopwatch& getProfiler();

}

#ifndef WIN32
#pragma GCC visibility pop
#endif

#endif
