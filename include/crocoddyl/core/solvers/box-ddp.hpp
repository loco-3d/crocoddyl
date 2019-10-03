///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019, CNRS-LAAS, The University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_BOX_DDP_HPP_
#define CROCODDYL_CORE_SOLVERS_BOX_DDP_HPP_

#include <Eigen/Cholesky>
#include <vector>
#include "crocoddyl/core/solvers/fddp.hpp"
#include "crocoddyl/core/solvers/box-qp.hpp"

namespace crocoddyl {

class SolverBoxDDP : public SolverDDP {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit SolverBoxDDP(ShootingProblem& problem);
  ~SolverBoxDDP();

  virtual void allocateData();
  virtual void computeGains(unsigned int const& t);
  virtual void forwardPass(const double& steplength);

 protected:
  std::vector<Eigen::MatrixXd> Quu_inv_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVERS_BOX_DDP_HPP_
