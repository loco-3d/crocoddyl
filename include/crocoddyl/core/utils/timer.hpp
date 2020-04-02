///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_UTILS_TIMER_H_
#define CROCODDYL_CORE_UTILS_TIMER_H_

#include <ctime>

namespace crocoddyl {

class Timer {
 public:
  Timer() { clock_gettime(CLOCK_MONOTONIC, &start_); }

  inline void reset() { clock_gettime(CLOCK_MONOTONIC, &start_); }

  inline double get_duration() {
    clock_gettime(CLOCK_MONOTONIC, &finish_);
    duration_ = static_cast<double>(finish_.tv_sec - start_.tv_sec) * 1000000;
    duration_ += static_cast<double>(finish_.tv_nsec - start_.tv_nsec) / 1000;
    return duration_ / 1000.;
  }
  
  inline double get_us_duration() {
    clock_gettime(CLOCK_MONOTONIC, &finish_);
    duration_ = static_cast<double>(finish_.tv_sec - start_.tv_sec) * 1000000;
    duration_ += static_cast<double>(finish_.tv_nsec - start_.tv_nsec) / 1000;
    return duration_;
  }
 private:
  struct timespec start_;
  struct timespec finish_;
  double duration_;
};
}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_UTILS_TIMER_H_
