#include "crocoddyl/core/utils/callbacks.hpp"

namespace crocoddyl {

CallbackVerbose::CallbackVerbose(VerboseLevel level) : CallbackAbstract(), level(level) {}

CallbackVerbose::~CallbackVerbose() {}

void CallbackVerbose::operator()(SolverAbstract& solver) {
  if (solver.get_iter() % 10 == 0) {
    switch (level) {
      case _1: {
        std::cout << "iter \t cost \t      stop \t    grad \t  xreg";
        std::cout << " \t      ureg \t step \t feas" << std::endl;
        break;
      }
      case _2: {
        std::cout << "iter \t cost \t      stop \t    grad \t  xreg";
        std::cout << " \t      ureg \t step \t feas \tdV-exp \t      dV" << std::endl;
        break;
      }
      default: {
        std::cout << "iter \t cost \t      stop \t    grad \t  xreg";
        std::cout << " \t      ureg \t step \t feas" << std::endl;
      }
    }
  }

  switch (level) {
    case _1: {
      std::cout << std::setw(4) << solver.get_iter() << "  ";
      std::cout << std::scientific << std::setprecision(5) << solver.get_cost() << "  ";
      std::cout << solver.get_stop() << "  " << -solver.get_d()[1] << "  ";
      std::cout << solver.get_xreg() << "  " << solver.get_ureg() << "   ";
      std::cout << std::fixed << std::setprecision(4) << solver.get_stepLength() << "     ";
      std::cout << solver.get_isFeasible() << std::endl;
      break;
    }
    case _2: {
      std::cout << std::setw(4) << solver.get_iter() << "  ";
      std::cout << std::scientific << std::setprecision(5) << solver.get_cost() << "  ";
      std::cout << solver.get_stop() << "  " << -solver.get_d()[1] << "  ";
      std::cout << solver.get_xreg() << "  " << solver.get_ureg() << "   ";
      std::cout << std::fixed << std::setprecision(4) << solver.get_stepLength() << "     ";
      std::cout << solver.get_isFeasible() << "  ";
      std::cout << std::scientific << std::setprecision(5) << solver.get_dV() << "  ";
      std::cout << solver.get_dVexp() << std::endl;
      break;
    }
    default: {
      std::cout << std::setw(4) << solver.get_iter() << "  ";
      std::cout << std::scientific << std::setprecision(5) << solver.get_cost() << "  ";
      std::cout << solver.get_stop() << "  " << -solver.get_d()[1] << "  ";
      std::cout << solver.get_xreg() << "  " << solver.get_ureg() << "   ";
      std::cout << std::fixed << std::setprecision(4) << solver.get_stepLength() << "     ";
      std::cout << solver.get_isFeasible() << std::endl;
    }
  }
}

}  // namespace crocoddyl
