///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////


#ifndef CROCODDYL_CORE_SOLVERS_DDP_HPP_
#define CROCODDYL_CORE_SOLVERS_DDP_HPP_

#include <crocoddyl/core/solver-base.hpp>
#include <Eigen/Cholesky>

namespace crocoddyl {

class SolverDDP : public SolverAbstract {
 public:
  SolverDDP(ShootingProblem& problem);
  ~SolverDDP();

  bool solve(const std::vector<Eigen::VectorXd>& init_xs,
             const std::vector<Eigen::VectorXd>& init_us,
             const unsigned int& maxiter=100,
             const bool& _isFeasible=false,
             const double& regInit=NAN) override;
  void computeDirection(const bool& recalc=true) override;
  double tryStep(const double& stepLength) override;
  double stoppingCriteria() override;
  const Eigen::Vector2d& expectedImprovement() override;

private:
  double calc();
  void backwardPass();
  void forwardPass(const double& stepLength);
  void computeGains(const long unsigned int& t);
  void increaseRegularization();
  void decreaseRegularization();
  void allocateData();

 protected:
  double regFactor;
  double regMin;
  double regMax;
  double cost_try;

  std::vector<Eigen::VectorXd> xs_try;
  std::vector<Eigen::VectorXd> us_try;
  std::vector<Eigen::VectorXd> dx;

  //allocate data
  std::vector<Eigen::MatrixXd> Vxx;
  std::vector<Eigen::VectorXd> Vx;
  std::vector<Eigen::MatrixXd> Qxx;
  std::vector<Eigen::MatrixXd> Qxu;
  std::vector<Eigen::MatrixXd> Quu;
  std::vector<Eigen::VectorXd> Qx;
  std::vector<Eigen::VectorXd> Qu;
  std::vector<Eigen::MatrixXd> K;
  std::vector<Eigen::VectorXd> k;
  std::vector<Eigen::VectorXd> gaps;

 private:
  Eigen::VectorXd x_next;
  std::vector<double> alphas;
  double th_grad;
  double th_step;
  bool wasFeasible;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVERS_DDP_HPP_
