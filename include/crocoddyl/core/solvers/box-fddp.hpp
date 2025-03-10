///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, University of Edinburgh, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_BOX_FDDP_HPP_
#define CROCODDYL_CORE_SOLVERS_BOX_FDDP_HPP_

#include "crocoddyl/core/solvers/box-qp.hpp"
#include "crocoddyl/core/solvers/fddp.hpp"

namespace crocoddyl {

template <typename _Scalar>
class SolverBoxFDDPTpl : public SolverFDDPTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef SolverFDDPTpl<Scalar> SolverFDDP;
  typedef BoxQPTpl<Scalar> BoxQP;
  typedef ShootingProblemTpl<Scalar> ShootingProblem;
  typedef typename ShootingProblem::ActionModelAbstract ActionModelAbstract;
  typedef typename ShootingProblem::ActionDataAbstract ActionDataAbstract;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  explicit SolverBoxFDDPTpl(std::shared_ptr<ShootingProblem> problem,
                            const DynamicsSolverType dyn_solver = FeasShoot,
                            const EqualitySolverType term_solver = LuNull);
  virtual ~SolverBoxFDDPTpl() = default;

  virtual void computePolicy(const std::size_t t) override;
  virtual void forwardPass(const Scalar steplength);

  const std::vector<MatrixXs>& get_Quu_inv() const;

 protected:
  void allocateData();
  virtual void resizeRunningData() override;
  using SolverFDDP::alphas_;
  using SolverFDDP::cost_try_;
  using SolverFDDP::dx_;
  using SolverFDDP::fs_;
  using SolverFDDP::is_feasible_;
  using SolverFDDP::K_;
  using SolverFDDP::k_;
  using SolverFDDP::problem_;
  using SolverFDDP::Qu_;
  using SolverFDDP::Quu_;
  using SolverFDDP::Qxu_;
  using SolverFDDP::th_stop_;
  using SolverFDDP::us_;
  using SolverFDDP::us_try_;
  using SolverFDDP::Vx_;
  using SolverFDDP::Vxx_;
  using SolverFDDP::xs_;
  using SolverFDDP::xs_try_;

  BoxQP qp_;
  std::vector<MatrixXs> Quu_inv_;
  std::vector<VectorXs> du_lb_;
  std::vector<VectorXs> du_ub_;
  VectorXs xnext_;  //!< Next state \f$\mathbf{x}^{'}\f$
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/solvers/box-fddp.hxx"

#endif  // CROCODDYL_CORE_SOLVERS_BOX_FDDP_HPP_
