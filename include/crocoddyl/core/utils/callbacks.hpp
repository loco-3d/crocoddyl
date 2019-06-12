///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////


#ifndef CROCODDYL_CORE_UTILS_CALLBACKS_HPP_
#define CROCODDYL_CORE_UTILS_CALLBACKS_HPP_

#include <crocoddyl/core/solver-base.hpp>
#include <iostream>
#include <iomanip>

namespace crocoddyl {

class CallbackAbstract {
 public:
  CallbackAbstract() { }
  ~CallbackAbstract() { }
  virtual void operator()(SolverAbstract *const solver) = 0;
};

enum DDPVerboseLevel { _1=0, _2 };
class CallbackDDPVerbose : public CallbackAbstract {
 public:
  CallbackDDPVerbose(DDPVerboseLevel level = DDPVerboseLevel::_1) : CallbackAbstract(), level(level) { }
  ~CallbackDDPVerbose() { }

  void operator()(SolverAbstract *const solver) override {
    if (solver->get_iter() % 10 == 0) {
      switch (level) {
      case _1: {
        std::cout << "iter \t cost \t      stop \t    grad \t  xreg";
        std::cout << " \t      ureg \t step \t feas" << std::endl;
      break;
      } case _2: {
        std::cout << "iter \t cost \t      stop \t    grad \t  xreg";
        std::cout << " \t      ureg \t step \t feas \tdV-exp \t      dV" << std::endl;
      break;
      } default: {
        std::cout << "iter \t cost \t      stop \t    grad \t  xreg";
        std::cout << " \t      ureg \t step \t feas" << std::endl;
      }}
    }

    switch (level) {
    case _1: {
      std::cout << std::setw(4) << solver->get_iter() << "  ";
      std::cout << std::scientific << std::setprecision(5) << solver->get_cost() << "  ";
      std::cout << solver->get_stop() << "  " << -solver->get_d()[1] << "  ";
      std::cout << solver->get_Xreg() << "  " << solver->get_Ureg() << "   ";
      std::cout << std::fixed << std::setprecision(4) << solver->get_stepLength() << "     ";
      std::cout << solver->get_isFeasible() << std::endl;
    break;
    } case _2: {
      std::cout << std::setw(4) << solver->get_iter() << "  ";
      std::cout << std::scientific << std::setprecision(5) << solver->get_cost() << "  ";
      std::cout << solver->get_stop() << "  " << -solver->get_d()[1] << "  ";
      std::cout << solver->get_Xreg() << "  " << solver->get_Ureg() << "   ";
      std::cout << std::fixed << std::setprecision(4) << solver->get_stepLength() << "     ";
      std::cout << solver->get_isFeasible() << "  ";
      std::cout << std::scientific << std::setprecision(5) << solver->get_dV() << "  ";
      std::cout << solver->get_dVexp() << std::endl;
    break;
    } default: {
      std::cout << std::setw(4) << solver->get_iter() << "  ";
      std::cout << std::scientific << std::setprecision(5) << solver->get_cost() << "  ";
      std::cout << solver->get_stop() << "  " << -solver->get_d()[1] << "  ";
      std::cout << solver->get_Xreg() << "  " << solver->get_Ureg() << "   ";
      std::cout << std::fixed << std::setprecision(4) << solver->get_stepLength() << "     ";
      std::cout << solver->get_isFeasible() << std::endl;
    }}
  }

 private:
  DDPVerboseLevel level;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_UTILS_CALLBACKS_HPP_