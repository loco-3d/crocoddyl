///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_UTILS_CALLBACKS_HPP_
#define CROCODDYL_CORE_UTILS_CALLBACKS_HPP_

#include <iostream>
#include <iomanip>

#include "crocoddyl/core/solver-base.hpp"

namespace crocoddyl {

enum VerboseLevel { _1 = 0, _2 };
class CallbackVerbose : public CallbackAbstract {
 public:
  explicit CallbackVerbose(VerboseLevel level = _1);
  ~CallbackVerbose();

  virtual void operator()(SolverAbstract& solver);

 private:
  VerboseLevel level;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_UTILS_CALLBACKS_HPP_
