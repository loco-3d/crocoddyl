///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          University of Oxford, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/callbacks.hpp"

namespace crocoddyl {

CallbackVerbose::CallbackVerbose(VerboseLevel level, int precision)
    : CallbackAbstract(),
      level_(level),
      separator_("  "),
      separator_short_(" ") {
  set_precision(precision);
}

CallbackVerbose::~CallbackVerbose() {}

VerboseLevel CallbackVerbose::get_level() const { return level_; }

void CallbackVerbose::set_level(VerboseLevel level) {
  level_ = level;
  update_header();
}

int CallbackVerbose::get_precision() const { return precision_; }

void CallbackVerbose::set_precision(int precision) {
  if (precision < 0) throw_pretty("The precision needs to be at least 0.");
  precision_ = precision;
  update_header();
}

void CallbackVerbose::update_header() {
  auto center_string = [](const std::string& str, int width,
                          bool right_padding = true) {
    const int padding_size = width - static_cast<int>(str.length());
    const int padding_left = padding_size > 0 ? padding_size / 2 : 0;
    const int padding_right =
        padding_size % 2 != 0
            ? padding_left + 1
            : padding_left;  // If the padding is odd, add additional space
    if (right_padding) {
      return std::string(padding_left, ' ') + str +
             std::string(padding_right, ' ');
    } else {
      return std::string(padding_left, ' ') + str;
    }
  };

  header_.clear();
  // Scientific mode requires a column width of 6 + precision
  const int columnwidth = 6 + precision_;
  header_ += "iter" + separator_;
  switch (level_) {
    case _0: {
      header_ += center_string("cost", columnwidth) + separator_;
      header_ += center_string("stop", columnwidth) + separator_;
      header_ += center_string("|grad|", columnwidth) + separator_;
      header_ += center_string("preg", columnwidth) + separator_;
      header_ += center_string("step", 6) + separator_;
      header_ += center_string("dV-exp", columnwidth) + separator_;
      header_ += center_string("dV", columnwidth, false);
      break;
    }
    case _1: {
      header_ += center_string("cost", columnwidth) + separator_;
      header_ += center_string("merit", columnwidth) + separator_;
      header_ += center_string("stop", columnwidth) + separator_;
      header_ += center_string("|grad|", columnwidth) + separator_;
      header_ += center_string("preg", columnwidth) + separator_;
      header_ += center_string("step", 6) + separator_;
      header_ += center_string("||ffeas||", columnwidth) + separator_;
      header_ += center_string("dV-exp", columnwidth) + separator_;
      header_ += center_string("dV", columnwidth, false);
      break;
    }
    case _2: {
      header_ += center_string("cost", columnwidth) + separator_;
      header_ += center_string("merit", columnwidth) + separator_;
      header_ += center_string("stop", columnwidth) + separator_;
      header_ += center_string("|grad|", columnwidth) + separator_;
      header_ += center_string("preg", columnwidth) + separator_;
      header_ += center_string("dreg", columnwidth) + separator_;
      header_ += center_string("step", 6) + separator_;
      header_ += center_string("||ffeas||", columnwidth) + separator_;
      header_ += center_string("dV-exp", columnwidth) + separator_;
      header_ += center_string("dV", columnwidth, false);
      break;
    }
    case _3: {
      header_ += center_string("cost", columnwidth) + separator_;
      header_ += center_string("merit", columnwidth) + separator_;
      header_ += center_string("stop", columnwidth) + separator_;
      header_ += center_string("|grad|", columnwidth) + separator_;
      header_ += center_string("preg", columnwidth) + separator_;
      header_ += center_string("dreg", columnwidth) + separator_;
      header_ += center_string("step", 6) + separator_;
      header_ += center_string("||ffeas||", columnwidth) + separator_;
      header_ += center_string("||gfeas||", columnwidth) + separator_;
      header_ += center_string("||hfeas||", columnwidth) + separator_;
      header_ += center_string("dV-exp", columnwidth) + separator_;
      header_ += center_string("dV", columnwidth, false);
      break;
    }
    case _4: {
      header_ += center_string("cost", columnwidth) + separator_;
      header_ += center_string("merit", columnwidth) + separator_;
      header_ += center_string("stop", columnwidth) + separator_;
      header_ += center_string("|grad|", columnwidth) + separator_;
      header_ += center_string("preg", columnwidth) + separator_;
      header_ += center_string("dreg", columnwidth) + separator_;
      header_ += center_string("step", 6) + separator_;
      header_ += center_string("||ffeas||", columnwidth) + separator_;
      header_ += center_string("||gfeas||", columnwidth) + separator_;
      header_ += center_string("||hfeas||", columnwidth) + separator_;
      header_ += center_string("dV-exp", columnwidth) + separator_;
      header_ += center_string("dV", columnwidth) + separator_;
      header_ += center_string("dPhi-exp", columnwidth) + separator_;
      header_ += center_string("dPhi", columnwidth, false);
      break;
    }
    default: {
    }
  }
}

void CallbackVerbose::operator()(SolverAbstract& solver) {
  if (solver.get_iter() % 10 == 0) {
    std::cout << header_ << std::endl << std::flush;
  }
  auto space_sign = [this](const double value) {
    std::stringstream stream;
    if (value >= 0.) {
      stream << " ";
    } else {
      stream << "-";
    }
    stream << std::scientific << std::setprecision(precision_) << abs(value);
    return stream.str();
  };

  std::cout << std::setw(4) << solver.get_iter() << separator_;  // iter
  switch (level_) {
    case _0: {
      std::cout << std::scientific << std::setprecision(precision_)
                << solver.get_cost() << separator_;       // cost
      std::cout << solver.get_stop() << separator_;       // stop
      std::cout << abs(solver.get_d()[0]) << separator_;  // grad
      std::cout << solver.get_preg() << separator_;       // preg
      std::cout << std::fixed << std::setprecision(4) << solver.get_steplength()
                << separator_short_;                                    // step
      std::cout << space_sign(solver.get_dVexp()) << separator_short_;  // dVexp
      std::cout << space_sign(solver.get_dV());                         // dV
      break;
    }
    case _1: {
      std::cout << std::scientific << std::setprecision(precision_)
                << solver.get_cost() << separator_short_;         // cost
      std::cout << space_sign(solver.get_merit()) << separator_;  // merit
      std::cout << solver.get_stop() << separator_;               // stop
      std::cout << abs(solver.get_d()[0]) << separator_;          // grad
      std::cout << solver.get_preg() << separator_;               // preg
      std::cout << std::fixed << std::setprecision(4) << solver.get_steplength()
                << separator_;  // step
      std::cout << std::scientific << std::setprecision(precision_)
                << solver.get_ffeas() << separator_short_;              // ffeas
      std::cout << space_sign(solver.get_dVexp()) << separator_short_;  // dVexp
      std::cout << space_sign(solver.get_dV());                         // dV
      break;
    }
    case _2: {
      std::cout << std::scientific << std::setprecision(precision_)
                << solver.get_cost() << separator_short_;         // cost
      std::cout << space_sign(solver.get_merit()) << separator_;  // merit
      std::cout << solver.get_stop() << separator_;               // stop
      std::cout << abs(solver.get_d()[0]) << separator_;          // grad
      std::cout << solver.get_preg() << separator_;               // preg
      std::cout << solver.get_dreg() << separator_;               // dreg
      std::cout << std::fixed << std::setprecision(4) << solver.get_steplength()
                << separator_;  // step
      std::cout << std::scientific << std::setprecision(precision_)
                << solver.get_ffeas() << separator_short_;              // ffeas
      std::cout << space_sign(solver.get_dVexp()) << separator_short_;  // dVexp
      std::cout << space_sign(solver.get_dV());                         // dV
      break;
    }
    case _3: {
      std::cout << std::scientific << std::setprecision(precision_)
                << solver.get_cost() << separator_short_;         // cost
      std::cout << space_sign(solver.get_merit()) << separator_;  // merit
      std::cout << solver.get_stop() << separator_;               // stop
      std::cout << abs(solver.get_d()[0]) << separator_;          // grad
      std::cout << solver.get_preg() << separator_;               // preg
      std::cout << solver.get_dreg() << separator_;               // dreg
      std::cout << std::fixed << std::setprecision(4) << solver.get_steplength()
                << separator_;  // step
      std::cout << std::scientific << std::setprecision(precision_)
                << solver.get_ffeas() << separator_;                    // ffeas
      std::cout << solver.get_gfeas() << separator_;                    // gfeas
      std::cout << solver.get_hfeas() << separator_short_;              // hfeas
      std::cout << space_sign(solver.get_dVexp()) << separator_short_;  // dVexp
      std::cout << space_sign(solver.get_dV());                         // dV
      break;
    }
    case _4: {
      std::cout << std::scientific << std::setprecision(precision_)
                << solver.get_cost() << separator_short_;         // cost
      std::cout << space_sign(solver.get_merit()) << separator_;  // merit
      std::cout << solver.get_stop() << separator_;               // stop
      std::cout << abs(solver.get_d()[0]) << separator_;          // grad
      std::cout << solver.get_preg() << separator_;               // preg
      std::cout << solver.get_dreg() << separator_;               // dreg
      std::cout << std::fixed << std::setprecision(4) << solver.get_steplength()
                << separator_;  // step
      std::cout << std::scientific << std::setprecision(precision_)
                << solver.get_ffeas() << separator_;                    // ffeas
      std::cout << solver.get_gfeas() << separator_;                    // gfeas
      std::cout << solver.get_hfeas() << separator_short_;              // hfeas
      std::cout << space_sign(solver.get_dVexp()) << separator_short_;  // dVexp
      std::cout << space_sign(solver.get_dV()) << separator_short_;     // dV
      std::cout << space_sign(solver.get_dPhiexp())
                << separator_short_;               // dPhiexp
      std::cout << space_sign(solver.get_dPhi());  // dPhi
      break;
    }
    default: {
    }
  }
  std::cout << std::endl;
  std::cout << std::flush;
}

}  // namespace crocoddyl
