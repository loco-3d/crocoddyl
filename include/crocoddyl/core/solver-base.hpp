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
  SolverAbstract(ShootingProblem& problem) : problem(problem), isFeasible(false),
    cost(0.), stop(0.), x_reg(NAN), u_reg(NAN), stepLength(1.), dV(0.), dV_exp(0.),
    th_acceptStep(0.1), th_stop(1e-9), iter(0) { }
  ~SolverAbstract() { }

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
                    const bool& _isFeasible=false) {
    const long unsigned int& T = problem.get_T();

    if (xs_warm.size() == 0) {
      for (long unsigned int t = 0; t < T; ++t) {
        xs[t] = problem.runningModels[t]->get_state()->zero();
      }
      xs.back() = problem.terminalModel->get_state()->zero();
    } else {
      assert(xs_warm.size()==T+1);
      std::copy(xs_warm.begin(), xs_warm.end(), xs.begin());
    }

    if(us_warm.size() == 0) {
      for (long unsigned int t = 0; t < T; ++t) {
        const int& nu = problem.runningModels[t]->get_nu();
        us[t] = Eigen::VectorXd::Zero(nu);
      }
    } else {
      assert(us_warm.size()==T);
      std::copy(us_warm.begin(), us_warm.end(), us.begin());
    }
    isFeasible = _isFeasible;
  }

  void setCallbacks(std::vector<CallbackAbstract*>& _callbacks) {
    callbacks = _callbacks;
  }

  const bool& get_isFeasible() const { return isFeasible; }
  const int& get_iter() const { return iter; }
  const double& get_cost() const { return cost; }
  const double& get_stop() const { return stop; }
  const Eigen::Vector2d& get_d() const { return d; }
  const double& get_Xreg() const { return x_reg; }
  const double& get_Ureg() const { return u_reg; }
  const double& get_stepLength() const { return stepLength; }
  const double& get_dV() const { return dV; }
  const double& get_dVexp() const { return dV_exp; }

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
  int iter;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVER_BASE_HPP_
