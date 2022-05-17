#ifdef CROCODDYL_WITH_IPOPT
#ifndef CROCODDYL_CORE_SOLVERS_IPOPT_HPP_
#define CROCODDYL_CORE_SOLVERS_IPOPT_HPP_

#include <coin-or/IpIpoptApplication.hpp>

#include "crocoddyl/core/solver-base.hpp"
#include "crocoddyl/core/solvers/ipopt/ipopt-iface.hpp"

namespace crocoddyl {

class SolverIpopt : public SolverAbstract {
 public:
  SolverIpopt(boost::shared_ptr<crocoddyl::ShootingProblem> problem);
  ~SolverIpopt();

  bool solve(const std::vector<Eigen::VectorXd>& init_xs = DEFAULT_VECTOR,
             const std::vector<Eigen::VectorXd>& init_us = DEFAULT_VECTOR, const std::size_t maxiter = 100,
             const bool is_feasible = false, const double regInit = 1e-9);

 private:
  Ipopt::SmartPtr<IpoptInterface> ipopt_iface_;
  Ipopt::SmartPtr<Ipopt::IpoptApplication> ipopt_app_;
  Ipopt::ApplicationReturnStatus ipopt_status_;

  virtual void computeDirection(const bool recalc);
  virtual double tryStep(const double steplength = 1);
  virtual double stoppingCriteria();
  virtual const Eigen::Vector2d& expectedImprovement();
};
}  // namespace crocoddyl

#endif
#endif
