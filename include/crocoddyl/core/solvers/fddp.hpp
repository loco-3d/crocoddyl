///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_FDDP_HPP_
#define CROCODDYL_CORE_SOLVERS_FDDP_HPP_

#include <Eigen/Cholesky>
#include <vector>

#include "crocoddyl/core/solvers/ddp.hpp"

namespace crocoddyl {

class SolverFDDP : public SolverDDP {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit SolverFDDP(boost::shared_ptr<ShootingProblem> problem);
  virtual ~SolverFDDP();

  virtual bool solve(const std::vector<Eigen::VectorXd>& init_xs = DEFAULT_VECTOR,
                     const std::vector<Eigen::VectorXd>& init_us = DEFAULT_VECTOR, const std::size_t& maxiter = 100,
                     const bool& is_feasible = false, const double& regInit = 1e-9);
  virtual const Eigen::Vector2d& expectedImprovement();
  void updateExpectedImprovement();
  virtual double calcDiff();
  virtual void forwardPass(const double& stepLength);

  double get_th_acceptnegstep() const;
  void set_th_acceptnegstep(const double& th_acceptnegstep);

 protected:
  double dg_;
  double dq_;
  double dv_;

 private:
  double th_acceptnegstep_;  //!< Threshold used for accepting step along ascent direction
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVERS_FDDP_HPP_
