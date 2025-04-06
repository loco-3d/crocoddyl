///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2025-2025, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_CODEGEN_PINOCCHIO_CAST_HPP_
#define CROCODDYL_MULTIBODY_CODEGEN_PINOCCHIO_CAST_HPP_

#include <pinocchio/autodiff/cppad.hpp>

#include "crocoddyl/multibody/fwd.hpp"

namespace pinocchio {

#if !PINOCCHIO_VERSION_AT_LEAST(3, 5, 0)
// Implement missing casting in Pinocchio
template <typename NewScalar, typename Scalar>
struct ScalarCast<NewScalar, CppAD::cg::CG<Scalar>> {
  static NewScalar cast(const CppAD::cg::CG<Scalar>& cg_value) {
    return static_cast<NewScalar>(cg_value.getValue());
  }
};
#endif

}  // namespace pinocchio

#endif  // CROCODDYL_MULTIBODY_CODEGEN_PINOCCHIO_CAST_HPP_
