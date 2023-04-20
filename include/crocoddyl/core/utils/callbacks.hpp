///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2022, LAAS-CNRS, University of Oxford
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_UTILS_CALLBACKS_HPP_
#define CROCODDYL_CORE_UTILS_CALLBACKS_HPP_

#include <iomanip>
#include <iostream>

#include "crocoddyl/core/solver-base.hpp"

namespace crocoddyl {

enum VerboseLevel { _1 = 0, _2 };
class CallbackVerbose : public CallbackAbstract {
 public:
  explicit CallbackVerbose(VerboseLevel level = _1, int precision = 5);
  ~CallbackVerbose() override;

  void operator()(SolverAbstract& solver) override;

  VerboseLevel get_level() const;
  void set_level(VerboseLevel level);

  int get_precision() const;
  void set_precision(int precision);

 private:
  VerboseLevel level_;
  int precision_;
  std::string header_;

  void update_header();
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_UTILS_CALLBACKS_HPP_
