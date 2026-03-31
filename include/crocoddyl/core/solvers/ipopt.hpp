///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2022-2025, IRI: CSIC-UPC, Heriot-Watt University
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
class SolverIpopt : public SolverAbstractTpl<Ipopt::Number> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_BASE_DERIVED_FLOATINGPOINT_CAST(SolverBase, SolverIpopt)

  typedef Ipopt::Number Scalar;
  typedef SolverAbstractTpl<Scalar> SolverAbstract;
  typedef ShootingProblemTpl<Scalar> ShootingProblem;
  typedef IpoptInterfaceTpl<Scalar> IpoptInterface;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Vector3s Vector3s;
  using SolverAbstract::resizeData;
  using SolverAbstract::setCandidate;

  /**
   * @brief Initialize the Ipopt solver
   *
   * @param[in]  problem solver to be diagnostic
   */
  SolverIpopt(std::shared_ptr<ShootingProblem> problem);
  ~SolverIpopt() = default;

  bool solve(
      const std::vector<VectorXs>& init_xs = DefaultVector<Scalar>::value,
      const std::vector<VectorXs>& init_us = DefaultVector<Scalar>::value,
      const std::size_t maxiter = 100, const bool is_feasible = false,
      const Scalar reg_init = Scalar(1e-9)) override;
  virtual void resizeData() override;

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

  void set_th_stop(const Scalar th_stop);

 private:
  Ipopt::SmartPtr<IpoptInterface> ipopt_iface_;
  Ipopt::SmartPtr<Ipopt::IpoptApplication> ipopt_app_;
  Ipopt::ApplicationReturnStatus ipopt_status_;

  virtual void computeDirection(const bool recalc) override;
  virtual Scalar tryStep(const Scalar steplength = Scalar(1.)) override;
  virtual Scalar stoppingCriteria() override;
  virtual Vector3s expectedImprovement() override;

  using SolverAbstract::cost_;
  using SolverAbstract::DV_;
  using SolverAbstract::iter_;
  using SolverAbstract::problem_;
  using SolverAbstract::th_stop_;
  using SolverAbstract::us_;
  using SolverAbstract::xs_;
};
}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/solvers/ipopt.hxx"

#endif
