///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019, The University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_BOX_DDP_HPP_
#define CROCODDYL_CORE_SOLVERS_BOX_DDP_HPP_

#include <Eigen/Cholesky>
#include <vector>
#include "crocoddyl/core/solvers/ddp.hpp"
#include "crocoddyl/core/solvers/box_qp.h"

namespace crocoddyl {

class SolverBoxDDP : public SolverDDP {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit SolverBoxDDP(ShootingProblem& problem);
  ~SolverBoxDDP();

  void allocateData();

  bool solve(const std::vector<Eigen::VectorXd>& init_xs = DEFAULT_VECTOR,
             const std::vector<Eigen::VectorXd>& init_us = DEFAULT_VECTOR, unsigned int const& maxiter = 100,
             const bool& is_feasible = false, const double& regInit = 1e-9);

  void computeDirection(const bool& recalc = true);
  void computeGains(unsigned int const& t);
  void backwardPass();
protected:
  std::vector<Eigen::MatrixXd> Quu_inv_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVERS_BOX_DDP_HPP_
