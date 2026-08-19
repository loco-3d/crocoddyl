///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026-2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_PARAMS_PARAMETER_PHASE_HPP_
#define CROCODDYL_CORE_PARAMS_PARAMETER_PHASE_HPP_

#include "crocoddyl/core/constraints/constraint-manager.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/params/parameter-manager.hpp"

namespace crocoddyl {

/**
 * @brief Parameterization and optional constraints associated with one phase
 *
 * This class composes a parameter manager with the optional constraints that
 * act on the same phase parameter vector. It owns neither node data nor
 * trajectory variables. `createData()` allocates both data objects together,
 * ensuring that the constraint data shares the phase parameter payload.
 */
template <typename _Scalar>
class ParameterPhaseModelTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ParameterManagerTpl<Scalar> ParameterManager;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef ConstraintModelManagerTpl<Scalar> ConstraintModelManager;
  typedef ConstraintDataManagerTpl<Scalar> ConstraintDataManager;
  typedef ParameterPhaseDataTpl<Scalar> ParameterPhaseData;
  typedef StateAbstractTpl<Scalar> StateAbstract;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Construct a phase parameterization
   *
   * @param[in] params Parameter manager
   * @param[in] constraints Optional constraints on the phase parameters
   */
  explicit ParameterPhaseModelTpl(
      std::shared_ptr<ParameterManager> params,
      std::shared_ptr<ConstraintModelManager> constraints = nullptr);
  ParameterPhaseModelTpl(const ParameterPhaseModelTpl&) = default;
  ParameterPhaseModelTpl& operator=(const ParameterPhaseModelTpl&) = delete;
  ~ParameterPhaseModelTpl() = default;

  /** @brief Create the parameter and optional constraint data together */
  std::shared_ptr<ParameterPhaseData> createData() const;

  /** @brief Update the active parameter vector */
  void update(const std::shared_ptr<ParameterPhaseData>& data,
              const Eigen::Ref<const VectorXs>& p) const;

  /** @brief Evaluate the optional phase constraints */
  void calc(const std::shared_ptr<ParameterPhaseData>& data,
            const Eigen::Ref<const VectorXs>& x,
            const Eigen::Ref<const VectorXs>& u) const;

  /** @brief Evaluate the derivatives of the optional phase constraints */
  void calcDiff(const std::shared_ptr<ParameterPhaseData>& data,
                const Eigen::Ref<const VectorXs>& x,
                const Eigen::Ref<const VectorXs>& u) const;

  /** @brief Cast the complete phase parameterization to another scalar */
  template <typename NewScalar>
  ParameterPhaseModelTpl<NewScalar> cast() const;

  /** @brief Return the parameter manager */
  const std::shared_ptr<ParameterManager>& get_params() const;

  /** @brief Return the optional phase constraint manager */
  const std::shared_ptr<ConstraintModelManager>& get_constraints() const;

  /** @brief Return the shared state description */
  const std::shared_ptr<StateAbstract>& get_state() const;

  /** @brief Return the active parameter dimension */
  std::size_t get_np() const;

  /** @brief Return true when the phase has active constraints */
  bool has_constraints() const;

 private:
  std::shared_ptr<ParameterManager> params_;
  std::shared_ptr<ConstraintModelManager> constraints_;
};

/**
 * @brief Data associated with one parameterized phase
 *
 * The parameter data is created first and used as the shared collector for the
 * optional constraint data. Both payloads therefore have the same lifetime and
 * cannot become misaligned across phase indices.
 */
template <typename _Scalar>
struct ParameterPhaseDataTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ParameterDataManagerTpl<Scalar> ParameterDataManager;
  typedef ConstraintDataManagerTpl<Scalar> ConstraintDataManager;

  ParameterPhaseDataTpl(
      std::shared_ptr<ParameterDataManager> params,
      std::shared_ptr<ConstraintDataManager> constraints = nullptr)
      : params(params), constraints(constraints) {}
  ParameterPhaseDataTpl(const ParameterPhaseDataTpl&) = default;
  ParameterPhaseDataTpl& operator=(const ParameterPhaseDataTpl&) = delete;
  ~ParameterPhaseDataTpl() = default;

  std::shared_ptr<ParameterDataManager> params;        //!< Parameter data
  std::shared_ptr<ConstraintDataManager> constraints;  //!< Constraint data
};

}  // namespace crocoddyl

#include "crocoddyl/core/params/parameter-phase.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ParameterPhaseModelTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ParameterPhaseDataTpl)

#endif  // CROCODDYL_CORE_PARAMS_PARAMETER_PHASE_HPP_
