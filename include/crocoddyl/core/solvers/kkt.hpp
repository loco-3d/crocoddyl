///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, New York University,
//                          Max Planck Gesellschaft, University of Edinburgh
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_SOLVERS_KKT_HPP_
#define CROCODDYL_CORE_SOLVERS_KKT_HPP_

#include "crocoddyl/core/solver-base.hpp"

namespace crocoddyl {

template <typename _Scalar>
class SolverKKTTpl : public SolverAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef SolverAbstractTpl<Scalar> SolverAbstract;
  typedef ShootingProblemTpl<Scalar> ShootingProblem;
  typedef typename ShootingProblem::ActionModelAbstract ActionModelAbstract;
  typedef typename ShootingProblem::ActionDataAbstract ActionDataAbstract;
  typedef CallbackAbstractTpl<Scalar> CallbackAbstract;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Vector2s Vector2s;
  using SolverAbstract::setCandidate;

  explicit SolverKKTTpl(std::shared_ptr<ShootingProblem> problem);
  virtual ~SolverKKTTpl() = default;

  virtual bool solve(
      const std::vector<VectorXs>& init_xs = DefaultVector<Scalar>::value,
      const std::vector<VectorXs>& init_us = DefaultVector<Scalar>::value,
      const std::size_t maxiter = 100, const bool is_feasible = false,
      const Scalar regInit = 1e-9) override;
  virtual void computeDirection(const bool recalc = true) override;
  virtual Scalar tryStep(const Scalar steplength = Scalar(1.)) override;
  virtual Scalar stoppingCriteria() override;
  virtual const Vector2s& expectedImprovement() override;

  const MatrixXs& get_kkt() const;
  const VectorXs& get_kktref() const;
  const VectorXs& get_primaldual() const;
  const std::vector<VectorXs>& get_dxs() const;
  const std::vector<VectorXs>& get_dus() const;
  const std::vector<VectorXs>& get_lambdas() const;
  std::size_t get_nx() const;
  std::size_t get_ndx() const;
  std::size_t get_nu() const;

 protected:
  void allocateData();
  using SolverAbstract::callbacks_;
  using SolverAbstract::cost_;
  using SolverAbstract::cost_try_;
  using SolverAbstract::d_;
  using SolverAbstract::dreg_;
  using SolverAbstract::DV_;
  using SolverAbstract::dV_;
  using SolverAbstract::dVexp_;
  using SolverAbstract::is_feasible_;
  using SolverAbstract::iter_;
  using SolverAbstract::preg_;
  using SolverAbstract::problem_;
  using SolverAbstract::steplength_;
  using SolverAbstract::stop_;
  using SolverAbstract::th_acceptstep_;
  using SolverAbstract::th_stop_;
  using SolverAbstract::us_;
  using SolverAbstract::us_try_;
  using SolverAbstract::xs_;
  using SolverAbstract::xs_try_;

  Scalar reg_incfactor_;
  Scalar reg_decfactor_;
  Scalar reg_min_;
  Scalar reg_max_;

 private:
  Scalar calcDiff();
  void computePrimalDual();
  void increaseRegularization();
  void decreaseRegularization();

  std::size_t nx_;
  std::size_t ndx_;
  std::size_t nu_;
  std::vector<VectorXs> dxs_;
  std::vector<VectorXs> dus_;
  std::vector<VectorXs> lambdas_;

  // allocate data
  MatrixXs kkt_;
  VectorXs kktref_;
  VectorXs primaldual_;
  VectorXs primal_;
  VectorXs dual_;
  std::vector<Scalar> alphas_;
  Scalar th_grad_;
  bool was_feasible_;
  VectorXs kkt_primal_;
  VectorXs dF;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/solvers/kkt.hxx"

#endif  // CROCODDYL_CORE_SOLVERS_KKT_HPP_
