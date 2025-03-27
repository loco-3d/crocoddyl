///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, LAAS-CNRS, University of Oxford,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_UTILS_CALLBACKS_HPP_
#define CROCODDYL_CORE_UTILS_CALLBACKS_HPP_

#include <iomanip>

#include "crocoddyl/core/solver-base.hpp"

namespace crocoddyl {

enum VerboseLevel {
  _0 = 0,  //<! Zero print level that doesn't contain merit-function and
           // constraints information
  _1,      //<! First print level that includes level-0, merit-function and
           // dynamics-constraints information
  _2,      //<! Second print level that includes level-1 and dual-variable
           // regularization
  _3,  //<! Third print level that includes level-2, and equality and inequality
       // constraints information
  _4,  //<! Fourht print level that includes expected and current improvements
       // in the merit function
};
class CallbackVerbose : public CallbackAbstract {
 public:
  explicit CallbackVerbose(VerboseLevel level = _4, int precision = 3);
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
  std::string separator_;
  std::string separator_short_;

  void update_header();
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_UTILS_CALLBACKS_HPP_
