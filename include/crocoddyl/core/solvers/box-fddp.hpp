///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2024, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_BOX_FDDP_HPP_
#define CROCODDYL_CORE_SOLVERS_BOX_FDDP_HPP_

#include "crocoddyl/core/solvers/box-qp.hpp"
#include "crocoddyl/core/solvers/fddp.hpp"

namespace crocoddyl {

class SolverBoxFDDP : public SolverFDDP {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  explicit SolverBoxFDDP(std::shared_ptr<ShootingProblem> problem,
                         const DynamicsSolverType dyn_solver = FeasShoot,
                         const EqualitySolverType term_solver = LuNull);
  virtual ~SolverBoxFDDP();

  virtual void computePolicy(const std::size_t t);
  virtual void forwardPass(const double steplength);

  const std::vector<Eigen::MatrixXd>& get_Quu_inv() const;

 protected:
  void allocateData();
  virtual void resizeRunningData();

  BoxQP qp_;
  std::vector<Eigen::MatrixXd> Quu_inv_;
  std::vector<Eigen::VectorXd> du_lb_;
  std::vector<Eigen::VectorXd> du_ub_;
  Eigen::VectorXd xnext_;  //!< Next state \f$\mathbf{x}^{'}\f$
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_SOLVERS_BOX_FDDP_HPP_
