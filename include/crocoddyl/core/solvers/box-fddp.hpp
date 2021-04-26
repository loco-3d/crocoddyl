///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_BOX_FDDP_HPP_
#define CROCODDYL_CORE_SOLVERS_BOX_FDDP_HPP_

#include <Eigen/Cholesky>
#include <vector>

#include "crocoddyl/core/solvers/fddp.hpp"
#include "crocoddyl/core/solvers/box-qp.hpp"

namespace crocoddyl {

class SolverBoxFDDP : public SolverFDDP {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit SolverBoxFDDP(boost::shared_ptr<ShootingProblem> problem);
  virtual ~SolverBoxFDDP();

  virtual void allocateData();
  virtual void computeGains(const std::size_t t);
  virtual void forwardPass(const double steplength);

  const crocoddyl::aligned_vector<Eigen::MatrixXd>& get_Quu_inv() const;

 protected:
  BoxQP qp_;
  crocoddyl::aligned_vector<Eigen::MatrixXd> Quu_inv_;
  Eigen::VectorXd du_lb_;
  Eigen::VectorXd du_ub_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVERS_BOX_FDDP_HPP_
