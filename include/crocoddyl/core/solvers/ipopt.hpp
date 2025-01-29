///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2022-2023, IRI: CSIC-UPC, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_IPOPT_HPP_
#define CROCODDYL_CORE_SOLVERS_IPOPT_HPP_

#define HAVE_CSTDDEF
#include <IpIpoptApplication.hpp>
#include <IpSolveStatistics.hpp>
#undef HAVE_CSTDDEF

#include "crocoddyl/core/solver-base.hpp"
#include "crocoddyl/core/solvers/ipopt/ipopt-iface.hpp"

namespace crocoddyl {

/**
 * @brief Ipopt solver
 *
 * This solver solves the optimal control problem by transcribing with the
 * multiple shooting approach.
 *
 * \sa `solve()`
 */
class SolverIpopt : public SolverAbstract {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Initialize the Ipopt solver
   *
   * @param[in]  problem solver to be diagnostic
   */
  SolverIpopt(std::shared_ptr<crocoddyl::ShootingProblem> problem);
  ~SolverIpopt();

  bool solve(const std::vector<Eigen::VectorXd>& init_xs = DEFAULT_VECTOR,
             const std::vector<Eigen::VectorXd>& init_us = DEFAULT_VECTOR,
             const std::size_t maxiter = 100, const bool is_feasible = false,
             const double reg_init = 1e-9);
  virtual void resizeData();

  /**
   * @brief Set a string ipopt option
   *
   * @param[in]  tag name of the parameter
   * @param[in]  value string value for the parameter
   */
  void setStringIpoptOption(const std::string& tag, const std::string& value);

  /**
   * @brief Set a string ipopt option
   *
   * @param[in]  tag name of the parameter
   * @param[in]  value numeric value for the parameter
   */
  void setNumericIpoptOption(const std::string& tag, Ipopt::Number value);

  void set_th_stop(const double th_stop);

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
