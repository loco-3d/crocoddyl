///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVER_BASE_HPP_
#define CROCODDYL_CORE_SOLVER_BASE_HPP_

#include <vector>
#include "crocoddyl/core/optctrl/shooting.hpp"

namespace crocoddyl {

class CallbackAbstract;  // forward declaration
static std::vector<Eigen::VectorXd> DEFAULT_VECTOR;

class SolverAbstract {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit SolverAbstract(boost::shared_ptr<ShootingProblem> problem);
  virtual ~SolverAbstract();

  virtual bool solve(const std::vector<Eigen::VectorXd>& init_xs = DEFAULT_VECTOR,
                     const std::vector<Eigen::VectorXd>& init_us = DEFAULT_VECTOR, const std::size_t& maxiter = 100,
                     const bool& is_feasible = false, const double& reg_init = 1e-9) = 0;
  // TODO(cmastalli): computeDirection (polymorphism) returning descent direction and lambdas
  virtual void computeDirection(const bool& recalc) = 0;
  virtual double tryStep(const double& step_length = 1) = 0;
  virtual double stoppingCriteria() = 0;
  virtual const Eigen::Vector2d& expectedImprovement() = 0;
  void setCandidate(const std::vector<Eigen::VectorXd>& xs_warm = DEFAULT_VECTOR,
                    const std::vector<Eigen::VectorXd>& us_warm = DEFAULT_VECTOR, const bool& is_feasible = false);

  void setCallbacks(const std::vector<boost::shared_ptr<CallbackAbstract> >& callbacks);
  const std::vector<boost::shared_ptr<CallbackAbstract> >& getCallbacks() const;

  const boost::shared_ptr<ShootingProblem>& get_problem() const;
  const std::vector<boost::shared_ptr<ActionModelAbstract> >& get_models() const;
  const std::vector<boost::shared_ptr<ActionDataAbstract> >& get_datas() const;
  const std::vector<Eigen::VectorXd>& get_xs() const;
  const std::vector<Eigen::VectorXd>& get_us() const;
  const bool& get_isFeasible() const;
  const double& get_cost() const;
  const double& get_stop() const;
  const Eigen::Vector2d& get_d() const;
  const double& get_xreg() const;
  const double& get_ureg() const;
  const double& get_stepLength() const;
  const double& get_dV() const;
  const double& get_dVexp() const;
  const double& get_th_acceptstep() const;
  const double& get_th_stop() const;
  const std::size_t& get_iter() const;

  void set_xs(const std::vector<Eigen::VectorXd>& xs);
  void set_us(const std::vector<Eigen::VectorXd>& us);
  void set_xreg(const double& xreg);
  void set_ureg(const double& ureg);
  void set_th_acceptstep(const double& th_acceptstep);
  void set_th_stop(const double& th_stop);

 protected:
  boost::shared_ptr<ShootingProblem> problem_;
  std::vector<boost::shared_ptr<ActionModelAbstract> > models_;
  std::vector<boost::shared_ptr<ActionDataAbstract> > datas_;
  std::vector<Eigen::VectorXd> xs_;
  std::vector<Eigen::VectorXd> us_;
  std::vector<boost::shared_ptr<CallbackAbstract> > callbacks_;
  bool is_feasible_;
  double cost_;
  double stop_;
  Eigen::Vector2d d_;
  double xreg_;
  double ureg_;
  double steplength_;
  double dV_;
  double dVexp_;
  double th_acceptstep_;
  double th_stop_;
  std::size_t iter_;
};

class CallbackAbstract {
 public:
  CallbackAbstract() {}
  virtual ~CallbackAbstract() {}

  virtual void operator()(SolverAbstract& solver) = 0;
};

bool raiseIfNaN(const double& value);

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVER_BASE_HPP_
