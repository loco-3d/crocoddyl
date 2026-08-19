///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_OPTCTRL_PROBLEM_ABSTRACT_HPP_
#define CROCODDYL_CORE_OPTCTRL_PROBLEM_ABSTRACT_HPP_

#include <memory>
#include <vector>

#include "crocoddyl/core/action-base.hpp"
#include "crocoddyl/core/fwd.hpp"

namespace crocoddyl {

/**
 * @brief Abstract base class for optimal control problems
 *
 * `ProblemAbstractTpl` defines the common interface used by solvers to
 * evaluate and roll out a trajectory, inspect the problem structure and
 * model/data objects, and track structural updates. Problems without
 * parameterized phases use the default empty phase interface; parameterized
 * problems override those hooks and `update_p()`.
 */
template <typename _Scalar>
class ProblemAbstractTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ActionModelAbstractTpl<Scalar> ActionModelAbstract;
  typedef ActionDataAbstractTpl<Scalar> ActionDataAbstract;
  typedef ParameterPhaseModelTpl<Scalar> ParameterPhaseModel;
  typedef ParameterPhaseDataTpl<Scalar> ParameterPhaseData;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;

  virtual ~ProblemAbstractTpl() = default;

  /**
   * @brief Compute the cost and next states along a trajectory
   *
   * @param[in] xs  State trajectory (size T+1)
   * @param[in] us  Control trajectory (size T)
   * @return Total cost
   */
  virtual Scalar calc(const std::vector<VectorXs>& xs,
                      const std::vector<VectorXs>& us) = 0;

  /**
   * @brief Compute the derivatives of cost and dynamics along a trajectory
   *
   * @param[in] xs  State trajectory (size T+1)
   * @param[in] us  Control trajectory (size T)
   * @return Total cost
   */
  virtual Scalar calcDiff(const std::vector<VectorXs>& xs,
                          const std::vector<VectorXs>& us) = 0;

  /**
   * @brief Integrate the system dynamics given a control sequence
   *
   * @param[in] us   Control trajectory (size T)
   * @param[out] xs  State trajectory populated by the rollout (size T+1)
   */
  virtual void rollout(const std::vector<VectorXs>& us,
                       std::vector<VectorXs>& xs) = 0;

  /**
   * @brief Integrate the dynamics and return the state trajectory
   *
   * The default implementation allocates a trajectory of size T+1 and calls
   * `rollout()`.
   */
  virtual std::vector<VectorXs> rollout_us(const std::vector<VectorXs>& us) {
    std::vector<VectorXs> xs(get_T() + 1);
    rollout(us, xs);
    return xs;
  }

  /** @brief Return the number of running nodes */
  virtual std::size_t get_T() const = 0;

  /** @brief Return the initial state */
  virtual const VectorXs& get_x0() const = 0;

  /** @brief Return the dimension of the state tuple */
  virtual std::size_t get_nx() const = 0;

  /** @brief Return the dimension of the state tangent space */
  virtual std::size_t get_ndx() const = 0;

  /** @brief Return the number of threads used for parallel evaluation */
  virtual std::size_t get_nthreads() const = 0;

  /** @brief Return the running action models (size T) */
  virtual const std::vector<std::shared_ptr<ActionModelAbstract> >&
  get_runningModels() const = 0;

  /** @brief Return the terminal action model */
  virtual const std::shared_ptr<ActionModelAbstract>& get_terminalModel()
      const = 0;

  /** @brief Return the running action data (size T) */
  virtual const std::vector<std::shared_ptr<ActionDataAbstract> >&
  get_runningDatas() const = 0;

  /** @brief Return the terminal action data */
  virtual const std::shared_ptr<ActionDataAbstract>& get_terminalData()
      const = 0;

  /**
   * @brief Return true once when the problem has been structurally modified
   */
  virtual bool is_updated() = 0;

  /** @brief Set the structural-update flag */
  virtual void set_is_updated(const bool is_updated) = 0;

  /**
   * @brief Return the number of phases
   *
   * Standard shooting problems have no explicit phases.
   */
  virtual std::size_t get_n_phases() const { return 0; }

  /**
   * @brief Update the parameter vector of a phase
   *
   * Parameterized problems override this method. The default implementation
   * reports that parameter updates are unsupported.
   */
  virtual void update_p(const Eigen::Ref<const VectorXs>&,
                        const std::size_t = 0) {
    throw_pretty("Invalid call: update_p is not supported for this problem");
  }

  /** @brief Return the inclusive start index of each phase */
  virtual const std::vector<std::size_t>& get_phase_idxs() const {
    static const std::vector<std::size_t> empty;
    return empty;
  }

  /** @brief Return the exclusive end index of each phase */
  virtual const std::vector<std::size_t>& get_phase_edxs() const {
    static const std::vector<std::size_t> empty;
    return empty;
  }

  /** @brief Return the phase-level parameter models */
  virtual const std::vector<std::shared_ptr<ParameterPhaseModel> >&
  get_paramsModel() const {
    static const std::vector<std::shared_ptr<ParameterPhaseModel> > empty;
    return empty;
  }

  /** @brief Return the phase-level parameter data */
  virtual const std::vector<std::shared_ptr<ParameterPhaseData> >&
  get_paramsData() const {
    static const std::vector<std::shared_ptr<ParameterPhaseData> > empty;
    return empty;
  }

  /** @brief Return true when parameter constraints are active */
  virtual bool has_parameter_constraints() const { return false; }
};

}  // namespace crocoddyl

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ProblemAbstractTpl)

#endif  // CROCODDYL_CORE_OPTCTRL_PROBLEM_ABSTRACT_HPP_
