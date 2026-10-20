///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_INTEGRATOR_TIME_HPP_
#define CROCODDYL_CORE_INTEGRATOR_TIME_HPP_

#include "crocoddyl/core/fwd.hpp"

namespace crocoddyl {

/**
 * @brief Time-step description shared by integrated action models
 *
 * This class stores the integration time step, its synchronized square, and
 * whether the time step is optimized as part of the problem parameters.
 */
template <typename _Scalar>
class IntegratorTimeTpl {
 public:
  typedef _Scalar Scalar;

  explicit IntegratorTimeTpl(const Scalar time_step = Scalar(1e-3),
                             const bool timeopt = false)
      : time_step_(Scalar(0.)), time_step2_(Scalar(0.)), timeopt_(timeopt) {
    set_time_step(time_step);
  }
  IntegratorTimeTpl(const IntegratorTimeTpl& other) = default;
  IntegratorTimeTpl& operator=(const IntegratorTimeTpl& other) = default;
  ~IntegratorTimeTpl() = default;

  template <typename NewScalar>
  IntegratorTimeTpl<NewScalar> cast() const {
    return IntegratorTimeTpl<NewScalar>(scalar_cast<NewScalar>(time_step_),
                                        timeopt_);
  }

  Scalar get_time_step() const { return time_step_; }
  Scalar get_time_step2() const { return time_step2_; }
  bool get_timeopt() const { return timeopt_; }

  void set_time_step(const Scalar time_step) {
    if (time_step < Scalar(0.)) {
      throw_pretty("Invalid argument: timeStep should be positive or zero");
    }
    time_step_ = time_step;
    time_step2_ = time_step * time_step;
  }

  void set_timeopt(const bool timeopt) { timeopt_ = timeopt; }

  friend std::ostream& operator<<(std::ostream& os,
                                  const IntegratorTimeTpl& integrator_time) {
    integrator_time.print(os);
    return os;
  }

  void print(std::ostream& os) const {
    os << "IntegratorTime {timeStep=" << time_step_
       << ", timeStep2=" << time_step2_
       << ", timeopt=" << (timeopt_ ? "true" : "false") << "}";
  }

 private:
  Scalar time_step_;
  Scalar time_step2_;
  bool timeopt_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_INTEGRATOR_TIME_HPP_
