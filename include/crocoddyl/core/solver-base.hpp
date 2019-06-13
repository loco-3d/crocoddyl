///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////


#ifndef CROCODDYL_CORE_SOLVER_BASE_HPP_
#define CROCODDYL_CORE_SOLVER_BASE_HPP_

#include <crocoddyl/core/optctrl/shooting.hpp>

namespace crocoddyl {

class CallbackAbstract; // forward declaration

class SolverAbstract {
 public:
  SolverAbstract(ShootingProblem& problem);
  ~SolverAbstract();

  virtual bool solve(const std::vector<Eigen::VectorXd>& init_xs,
                     const std::vector<Eigen::VectorXd>& init_us,
                     const unsigned int& maxiter,
                     const bool& isFeasible,
                     const double& regInit) = 0;
  //TODO: computeDirection (polimorfism) returning descent direction and lambdas
  virtual void computeDirection(const bool& recalc) = 0;
  virtual double tryStep(const double& stepLength) = 0;
  virtual double stoppingCriteria() = 0;
  virtual const Eigen::Vector2d& expectedImprovement() = 0;
  void setCandidate(const std::vector<Eigen::VectorXd>& xs_warm,
                    const std::vector<Eigen::VectorXd>& us_warm,
                    const bool& _isFeasible=false);

  void setCallbacks(std::vector<CallbackAbstract*>& _callbacks);

  const bool& get_isFeasible() const;
  const unsigned int& get_iter() const;
  const double& get_cost() const;
  const double& get_stop() const;
  const Eigen::Vector2d& get_d() const;
  const double& get_Xreg() const;
  const double& get_Ureg() const;
  const double& get_stepLength() const;
  const double& get_dV() const;
  const double& get_dVexp() const;

 protected:
  ShootingProblem problem;
  std::vector<Eigen::VectorXd> xs;
  std::vector<Eigen::VectorXd> us;
  std::vector<CallbackAbstract*> callbacks;
  bool isFeasible;
  double cost;
  double stop;
  Eigen::Vector2d d;
  double x_reg;
  double u_reg;
  double stepLength;
  double dV;
  double dV_exp;
  double th_acceptStep;
  double th_stop;
  unsigned int iter;
};

class CallbackAbstract {
 public:
  CallbackAbstract() {}
  ~CallbackAbstract() {}
  virtual void operator()(SolverAbstract *const solver) = 0;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVER_BASE_HPP_
