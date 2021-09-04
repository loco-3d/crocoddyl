///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/callbacks.hpp"

namespace crocoddyl {

CallbackVerbose::CallbackVerbose(VerboseLevel level) : CallbackAbstract(), level(level) {}

CallbackVerbose::~CallbackVerbose() {}

void CallbackVerbose::operator()(SolverAbstract& solver) {
  if (solver.get_iter() % 10 == 0) {
    std::cout << "iter     cost         stop         grad         xreg         ureg       step    ||ffeas||";
    switch (level) {
      case _2: {
        std::cout << "     dV-exp         dV";
        break;
      }
      default: {}
    }
    std::cout << std::endl;
  }

  std::cout << std::setw(4) << solver.get_iter() << "  ";
  std::cout << std::scientific << std::setprecision(5) << solver.get_cost() << "  ";
  std::cout << solver.get_stop() << "  " << -solver.get_d()[1] << "  ";
  std::cout << solver.get_xreg() << "  " << solver.get_ureg() << "  ";
  std::cout << std::fixed << std::setprecision(4) << solver.get_steplength() << "  ";
  std::cout << std::scientific << std::setprecision(5) << solver.get_ffeas();
  switch (level) {
    case _2: {
      std::cout << "  " << std::scientific << std::setprecision(5) << solver.get_dVexp() << "  ";
      std::cout << solver.get_dV();
      break;
    }
    default: {}
  }
  std::cout << std::endl;
  std::cout << std::flush;
}

}  // namespace crocoddyl
