///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh, University of Pisa
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_INTRO_HPP_
#define CROCODDYL_CORE_SOLVERS_INTRO_HPP_

#include "crocoddyl/core/solvers/ddp.hpp"

namespace crocoddyl {

class SolverIntro : public SolverDDP {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /**
   * @brief Initialize the INTRO solver
   *
   * @param[in] problem  Shooting problem
   */
  explicit SolverIntro(boost::shared_ptr<ShootingProblem> problem);
  virtual ~SolverIntro();

  virtual bool solve(const std::vector<Eigen::VectorXd>& init_xs = DEFAULT_VECTOR,
                     const std::vector<Eigen::VectorXd>& init_us = DEFAULT_VECTOR, const std::size_t maxiter = 100,
                     const bool is_feasible = false, const double regInit = 1e-9);
  virtual double tryStep(const double step_length = 1);
  virtual double stoppingCriteria();
 
   /**
   * @brief Return the rho parameter used in the merit function
   */
  double get_rho() const;

  /**
   * @brief Modify the rho parameter used in the merit function
   */
  void set_rho(const double rho);

  protected:
  double dPhi_; //!< Reduction in the merit function obtained by `tryStep()`
  double dPhiexp_; //!< Expected reduction in the merit function
  double rho_;  //!< Parameter used in the merit function to predict the expected reduction
  double hfeas_try_;  //!< Feasibility of the equality constraint computed by the line search
  double upsilon_;  //!< Estimated penalty paramter that balances relative contribution of the cost function and equality constraints
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVERS_INTRO_HPP_