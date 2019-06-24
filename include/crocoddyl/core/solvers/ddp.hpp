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
             const bool& is_feasible=false,
             const double& regInit=NAN) override;
  void computeDirection(const bool& recalc=true) override;
  double tryStep(const double& steplength) override;
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
  double regfactor_;
  double regmin_;
  double regmax_;

  double cost_try_;
  std::vector<Eigen::VectorXd> xs_try_;
  std::vector<Eigen::VectorXd> us_try_;
  std::vector<Eigen::VectorXd> dx_;

  //allocate data
  std::vector<Eigen::MatrixXd> Vxx_;
  std::vector<Eigen::VectorXd> Vx_;
  std::vector<Eigen::MatrixXd> Qxx_;
  std::vector<Eigen::MatrixXd> Qxu_;
  std::vector<Eigen::MatrixXd> Quu_;
  std::vector<Eigen::VectorXd> Qx_;
  std::vector<Eigen::VectorXd> Qu_;
  std::vector<Eigen::MatrixXd> K_;
  std::vector<Eigen::VectorXd> k_;
  std::vector<Eigen::VectorXd> gaps_;

 private:
  Eigen::VectorXd xnext_;
  std::vector<double> alphas_;
  double th_grad_;
  double th_step_;
  bool was_feasible_;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVERS_DDP_HPP_
