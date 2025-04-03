///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Oxford,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_UTILS_CALLBACKS_HPP_
#define CROCODDYL_CORE_UTILS_CALLBACKS_HPP_

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

template <typename _Scalar>
class CallbackVerboseTpl : public CallbackAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_FLOATINGPOINT_CAST(CallbackBase, CallbackVerboseTpl)

  typedef _Scalar Scalar;
  typedef CallbackAbstractTpl<Scalar> CallbackAbstract;
  typedef SolverAbstractTpl<Scalar> SolverAbstract;

  explicit CallbackVerboseTpl(VerboseLevel level = _4, int precision = 3);
  ~CallbackVerboseTpl() = default;

  void operator()(SolverAbstract& solver) override;

  VerboseLevel get_level() const;
  void set_level(VerboseLevel level);

  int get_precision() const;
  void set_precision(int precision);

  /**
   * @brief Cast the verbose callback
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return CallbackVerboseTpl<NewScalar> A verbose callback with the new
   * scalar type.
   */
  template <typename NewScalar>
  CallbackVerboseTpl<NewScalar> cast() const;

 private:
  VerboseLevel level_;
  int precision_;
  std::string header_;
  std::string separator_;
  std::string separator_short_;

  void update_header();
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/utils/callbacks.hxx"

#endif  // CROCODDYL_CORE_UTILS_CALLBACKS_HPP_
