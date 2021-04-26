///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, New York University, Max Planck Gesellschaft,
//                          University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_KKT_HPP_
#define CROCODDYL_CORE_SOLVERS_KKT_HPP_

#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include "crocoddyl/core/solver-base.hpp"

namespace crocoddyl {

class SolverKKT : public SolverAbstract {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit SolverKKT(boost::shared_ptr<ShootingProblem> problem);
  virtual ~SolverKKT();

  virtual bool solve(const std::vector<Eigen::VectorXd>& init_xs = DEFAULT_VECTOR,
                     const std::vector<Eigen::VectorXd>& init_us = DEFAULT_VECTOR, const std::size_t maxiter = 100,
                     const bool is_feasible = false, const double regInit = 1e-9);
  virtual void computeDirection(const bool recalc = true);
  virtual double tryStep(const double steplength = 1);
  virtual double stoppingCriteria();
  virtual const Eigen::Vector2d& expectedImprovement();

  const Eigen::MatrixXd& get_kkt() const;
  const Eigen::VectorXd& get_kktref() const;
  const Eigen::VectorXd& get_primaldual() const;
  const std::vector<Eigen::VectorXd>& get_dxs() const;
  const std::vector<Eigen::VectorXd>& get_dus() const;
  const std::vector<Eigen::VectorXd>& get_lambdas() const;
  std::size_t get_nx() const;
  std::size_t get_ndx() const;
  std::size_t get_nu() const;

 protected:
  double reg_incfactor_;
  double reg_decfactor_;
  double reg_min_;
  double reg_max_;
  double cost_try_;
  std::vector<Eigen::VectorXd> xs_try_;
  std::vector<Eigen::VectorXd> us_try_;

 private:
  double calcDiff();
  void computePrimalDual();
  void increaseRegularization();
  void decreaseRegularization();
  void allocateData();

  std::size_t nx_;
  std::size_t ndx_;
  std::size_t nu_;
  std::vector<Eigen::VectorXd> dxs_;
  std::vector<Eigen::VectorXd> dus_;
  std::vector<Eigen::VectorXd> lambdas_;

  // allocate data
  Eigen::MatrixXd kkt_;
  Eigen::VectorXd kktref_;
  Eigen::VectorXd primaldual_;
  Eigen::VectorXd primal_;
  Eigen::VectorXd dual_;
  std::vector<double> alphas_;
  double th_grad_;
  bool was_feasible_;
  Eigen::VectorXd kkt_primal_;
  Eigen::VectorXd dF;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVERS_KKT_HPP_
