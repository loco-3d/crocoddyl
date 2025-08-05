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
  CROCODDYL_DERIVED_FLOATINGPOINT_CAST(SolverBase, SolverBoxFDDPTpl)

  typedef _Scalar Scalar;
  typedef SolverFDDPTpl<Scalar> SolverFDDP;
  typedef BoxQPTpl<Scalar> BoxQP;
  typedef BoxQPSolutionTpl<Scalar> BoxQPSolution;
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

  /**
   * @brief Cast the Box-FDDP solver to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return SolverBoxFDDPTpl<NewScalar> An Box-FDDP solver with the new scalar
   * type.
   */
  template <typename NewScalar>
  SolverBoxFDDPTpl<NewScalar> cast() const;

  const std::vector<MatrixXs>& get_Quu_inv() const;

 protected:
  void allocateData();
  virtual void resizeRunningData() override;
  using SolverFDDP::alphas_;
  using SolverFDDP::callbacks_;
  using SolverFDDP::cost_try_;
  using SolverFDDP::dx_;
  using SolverFDDP::dyn_solver_;
  using SolverFDDP::fs_;
  using SolverFDDP::is_feasible_;
  using SolverFDDP::K_;
  using SolverFDDP::k_;
  using SolverFDDP::problem_;
  using SolverFDDP::Qu_;
  using SolverFDDP::Quu_;
  using SolverFDDP::Qxu_;
  using SolverFDDP::reg_decfactor_;
  using SolverFDDP::reg_incfactor_;
  using SolverFDDP::reg_max_;
  using SolverFDDP::reg_min_;
  using SolverFDDP::rho_;
  using SolverFDDP::term_solver_;
  using SolverFDDP::th_acceptminstep_;
  using SolverFDDP::th_acceptnegstep_;
  using SolverFDDP::th_acceptstep_;
  using SolverFDDP::th_grad_;
  using SolverFDDP::th_minfeas_;
  using SolverFDDP::th_minimprove_;
  using SolverFDDP::th_noimprovement_;
  using SolverFDDP::th_stepdec_;
  using SolverFDDP::th_stepinc_;
  using SolverFDDP::th_stop_;
  using SolverFDDP::upsilon_decfactor_;
  using SolverFDDP::us_;
  using SolverFDDP::us_try_;
  using SolverFDDP::Vx_;
  using SolverFDDP::Vxx_;
  using SolverFDDP::xs_;
  using SolverFDDP::xs_try_;
  using SolverFDDP::zero_upsilon_;

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

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::SolverBoxFDDPTpl)

#endif  // CROCODDYL_CORE_SOLVERS_BOX_FDDP_HPP_
