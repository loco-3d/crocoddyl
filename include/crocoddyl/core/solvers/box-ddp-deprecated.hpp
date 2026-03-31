///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, CNRS-LAAS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_BOX_DDP_DEPRECATED_HPP_
#define CROCODDYL_CORE_SOLVERS_BOX_DDP_DEPRECATED_HPP_

#include "crocoddyl/core/solvers/box-fddp.hpp"
#include "crocoddyl/core/solvers/ddp-deprecated.hpp"

namespace crocoddyl {

class SolverBoxDDP : public SolverBoxFDDP {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  DEPRECATED(
      "Do not use SolverBoxDDP. Instead, you should use SolverBoxFDDP(problem, "
      "DynamicsSolverType::SingleShoot)",
      explicit SolverBoxDDP(
          std::shared_ptr<ShootingProblem> problem) : SolverBoxFDDP(problem){})
  virtual ~SolverBoxDDP() = default;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVERS_BOX_DDP_DEPRECATED_HPP_
