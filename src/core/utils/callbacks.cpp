///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2022, LAAS-CNRS, University of Edinburgh,
//                          University of Oxford, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "crocoddyl/core/utils/callbacks.hpp"

#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

CallbackVerbose::CallbackVerbose(VerboseLevel level, int precision)
    : CallbackAbstract(), level_(level) {
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
  if (precision < 1) throw_pretty("The precision needs to be at least 1.");
  precision_ = precision;
  update_header();
}

void CallbackVerbose::update_header() {
  header_.clear();
  const int columnwidth =
      6 +
      precision_;  // Scientific mode requires a column width of 6 + precision
  const std::string separator{"  "};  // We use two spaces between columns
  header_ += "iter" + separator;
  auto center_string = [](const std::string& str, int width) {
    const int padding_size = width - static_cast<int>(str.length());
    const int padding_left = padding_size > 0 ? padding_size / 2 : 0;
    const int padding_right =
        padding_size % 2 != 0
            ? padding_left + 1
            : padding_left;  // If the padding is odd, add additional space
    return std::string(padding_left, ' ') + str +
           std::string(padding_right, ' ');
  };
  header_ += center_string("cost", columnwidth) + separator;
  header_ += center_string("stop", columnwidth) + separator;
  header_ += center_string("grad", columnwidth) + separator;
  header_ += center_string("xreg", columnwidth) + separator;
  header_ += center_string("ureg", columnwidth) + separator;
  header_ += center_string("step", 2 + 4) + separator;
  header_ += center_string("||ffeas||", columnwidth) + separator;
  header_ += center_string("||gfeas||", columnwidth) + separator;
  header_ += center_string("||hfeas||", columnwidth);
  switch (level_) {
    case _2: {
      header_ += separator + center_string("dV-exp", columnwidth) + separator;
      header_ += center_string("dV", columnwidth);
      break;
    }
    default: {
    }
  }
}

void CallbackVerbose::operator()(SolverAbstract& solver) {
  if (solver.get_iter() % 10 == 0) {
    std::cout << header_ << std::endl;
  }

  std::cout << std::setw(4) << solver.get_iter() << "  ";
  std::cout << std::scientific << std::setprecision(precision_)
            << solver.get_cost() << "  ";
  std::cout << solver.get_stop() << "  " << -solver.get_d()[1] << "  ";
  std::cout << solver.get_xreg() << "  " << solver.get_ureg() << "  ";
  std::cout << std::fixed << std::setprecision(4) << solver.get_steplength()
            << "  ";
  std::cout << std::scientific << std::setprecision(precision_)
            << solver.get_ffeas() << "  ";
  std::cout << std::scientific << std::setprecision(precision_)
            << solver.get_gfeas() << "  ";
  std::cout << std::scientific << std::setprecision(precision_)
            << solver.get_hfeas();
  switch (level_) {
    case _2: {
      std::cout << "  " << std::scientific << std::setprecision(precision_)
                << solver.get_dVexp() << "  ";
      std::cout << solver.get_dV();
      break;
    }
    default: {
    }
  }
  std::cout << std::endl;
  std::cout << std::flush;
}

}  // namespace crocoddyl
