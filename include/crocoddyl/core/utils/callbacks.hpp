///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_UTILS_CALLBACKS_HPP_
#define CROCODDYL_CORE_UTILS_CALLBACKS_HPP_

#include "crocoddyl/core/solver-base.hpp"
#include <iostream>
#include <iomanip>

namespace crocoddyl {

enum DDPVerboseLevel { _1 = 0, _2 };
class CallbackDDPVerbose : public CallbackAbstract {
 public:
  CallbackDDPVerbose(DDPVerboseLevel level = _1);
  ~CallbackDDPVerbose();

  void operator()(SolverAbstract *const solver);

 private:
  DDPVerboseLevel level;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_UTILS_CALLBACKS_HPP_