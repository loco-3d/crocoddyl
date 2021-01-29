///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, CNRS-LAAS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_BOX_DDP_HPP_
#define CROCODDYL_CORE_SOLVERS_BOX_DDP_HPP_

#include <Eigen/Cholesky>
#include <vector>

#include "crocoddyl/core/solvers/box-qp.hpp"
#include "crocoddyl/core/solvers/ddp.hpp"

namespace crocoddyl {

class SolverBoxDDP : public SolverDDP {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit SolverBoxDDP(boost::shared_ptr<ShootingProblem> problem);
  virtual ~SolverBoxDDP();

  virtual void allocateData();
  virtual void computeGains(const std::size_t &t);
  virtual void forwardPass(const double &steplength);

  const std::vector<Eigen::MatrixXd> &get_Quu_inv() const;

protected:
  BoxQP qp_;
  std::vector<Eigen::MatrixXd> Quu_inv_;
  Eigen::VectorXd du_lb_;
  Eigen::VectorXd du_ub_;
};

} // namespace crocoddyl

#endif // CROCODDYL_CORE_SOLVERS_BOX_DDP_HPP_
