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
#include "crocoddyl/core/solvers/fddp.hpp"
#include "crocoddyl/core/solvers/box_qp.h"

namespace crocoddyl {

class SolverBoxDDP : public SolverDDP {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit SolverBoxDDP(ShootingProblem& problem);
  ~SolverBoxDDP();

  void allocateData() override;
  void computeGains(unsigned int const& t) override;
  void forwardPass(const double& steplength) override;
protected:
  std::vector<Eigen::MatrixXd> Quu_inv_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVERS_BOX_DDP_HPP_
