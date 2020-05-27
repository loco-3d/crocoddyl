///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_DDP_HPP_
#define CROCODDYL_CORE_SOLVERS_DDP_HPP_

#include <Eigen/Cholesky>
#include <vector>

#include "crocoddyl/core/solver-base.hpp"

namespace crocoddyl {

class SolverDDP : public SolverAbstract {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit SolverDDP(boost::shared_ptr<ShootingProblem> problem);
  virtual ~SolverDDP();

  virtual bool solve(const std::vector<Eigen::VectorXd>& init_xs = DEFAULT_VECTOR,
                     const std::vector<Eigen::VectorXd>& init_us = DEFAULT_VECTOR, const std::size_t& maxiter = 100,
                     const bool& is_feasible = false, const double& regInit = 1e-9);
  virtual void computeDirection(const bool& recalc = true);
  virtual double tryStep(const double& steplength = 1);
  virtual double stoppingCriteria();
  virtual const Eigen::Vector2d& expectedImprovement();
  virtual double calcDiff();
  virtual void backwardPass();
  virtual void forwardPass(const double& stepLength);

  virtual void computeGains(const std::size_t& t);
  void increaseRegularization();
  void decreaseRegularization();
  virtual void allocateData();

  const double& get_regfactor() const;
  const double& get_regmin() const;
  const double& get_regmax() const;
  const std::vector<double>& get_alphas() const;
  const double& get_th_stepdec() const;
  const double& get_th_stepinc() const;
  const double& get_th_grad() const;
  const std::vector<Eigen::MatrixXd>& get_Vxx() const;
  const std::vector<Eigen::VectorXd>& get_Vx() const;
  const std::vector<Eigen::MatrixXd>& get_Qxx() const;
  const std::vector<Eigen::MatrixXd>& get_Qxu() const;
  const std::vector<Eigen::MatrixXd>& get_Quu() const;
  const std::vector<Eigen::VectorXd>& get_Qx() const;
  const std::vector<Eigen::VectorXd>& get_Qu() const;
  const std::vector<Eigen::MatrixXd>& get_K() const;
  const std::vector<Eigen::VectorXd>& get_k() const;
  const std::vector<Eigen::VectorXd>& get_fs() const;

  void set_regfactor(const double& reg_factor);
  void set_regmin(const double& regmin);
  void set_regmax(const double& regmax);
  void set_alphas(const std::vector<double>& alphas);
  void set_th_stepdec(const double& th_step);
  void set_th_stepinc(const double& th_step);
  void set_th_grad(const double& th_grad);

 protected:
  double regfactor_;  //!< Regularization factor used to decrease / increase it
  double regmin_;     //!< Minimum allowed regularization value
  double regmax_;     //!< Maximum allowed regularization value

  double cost_try_;                      //!< Total cost computed by line-search procedure
  std::vector<Eigen::VectorXd> xs_try_;  //!< State trajectory computed by line-search procedure
  std::vector<Eigen::VectorXd> us_try_;  //!< Control trajectory computed by line-search procedure
  std::vector<Eigen::VectorXd> dx_;

  // allocate data
  std::vector<Eigen::MatrixXd> Vxx_;  //!< Hessian of the Value function
  std::vector<Eigen::VectorXd> Vx_;   //!< Gradient of the Value function
  std::vector<Eigen::MatrixXd> Qxx_;  //!< Hessian of the Hamiltonian
  std::vector<Eigen::MatrixXd> Qxu_;  //!< Hessian of the Hamiltonian
  std::vector<Eigen::MatrixXd> Quu_;  //!< Hessian of the Hamiltonian
  std::vector<Eigen::VectorXd> Qx_;   //!< Gradient of the Hamiltonian
  std::vector<Eigen::VectorXd> Qu_;   //!< Gradient of the Hamiltonian
  std::vector<Eigen::MatrixXd> K_;    //!< Feedback gains
  std::vector<Eigen::VectorXd> k_;    //!< Feed-forward terms
  std::vector<Eigen::VectorXd> fs_;   //!< Gaps/defects between shooting nodes

  Eigen::VectorXd xnext_;
  Eigen::MatrixXd FxTVxx_p_;
  std::vector<Eigen::MatrixXd> FuTVxx_p_;
  Eigen::VectorXd fTVxx_p_;
  std::vector<Eigen::LLT<Eigen::MatrixXd> > Quu_llt_;
  std::vector<Eigen::VectorXd> Quuk_;
  std::vector<double> alphas_;  //!< Set of step lengths using by the line-search procedure
  double th_grad_;              //!< Tolerance of the expected gradient used for testing the step
  double th_stepdec_;           //!< Step-length threshold used to decrease regularization
  double th_stepinc_;           //!< Step-length threshold used to increase regularization
  bool was_feasible_;           //!< Label that indicates in the previous iterate was feasible
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVERS_DDP_HPP_
