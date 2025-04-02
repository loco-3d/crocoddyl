///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, Heriot-Watt University, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTIONS_FREE_INVDYN_HPP_
#define CROCODDYL_MULTIBODY_ACTIONS_FREE_INVDYN_HPP_

#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/constraints/constraint-manager.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

/**
 * @brief Differential action model for free inverse dynamics in multibody
 * systems.
 *
 * This class implements forward kinematic with an inverse-dynamics computed
 * using the Recursive Newton Euler Algorithm (RNEA). The stack of cost and
 * constraint functions are implemented in `CostModelSumTpl` and
 * `ConstraintModelManagerTpl`, respectively. The accelerations are the decision
 * variables defined as the control inputs, and the under-actuation constraint
 * is under the name `tau`, thus the user is not allowed to use it.
 *
 *
 * \sa `DifferentialActionModelAbstractTpl`, `calc()`, `calcDiff()`,
 * `createData()`
 */
template <typename _Scalar>
class DifferentialActionModelFreeInvDynamicsTpl
    : public DifferentialActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(DifferentialActionModelBase,
                         DifferentialActionModelFreeInvDynamicsTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DifferentialActionModelAbstractTpl<Scalar> Base;
  typedef DifferentialActionDataFreeInvDynamicsTpl<Scalar> Data;
  typedef DifferentialActionDataAbstractTpl<Scalar>
      DifferentialActionDataAbstract;
  typedef CostModelSumTpl<Scalar> CostModelSum;
  typedef ConstraintModelManagerTpl<Scalar> ConstraintModelManager;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ActuationModelAbstractTpl<Scalar> ActuationModelAbstract;
  typedef ConstraintModelResidualTpl<Scalar> ConstraintModelResidual;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize the free inverse-dynamics action model
   *
   * It describes the kinematic evolution of the multibody system and computes
   * the needed torques using inverse dynamics.
   *
   * @param[in] state      State of the multibody system
   * @param[in] actuation  Actuation model
   * @param[in] costs      Cost model
   */
  DifferentialActionModelFreeInvDynamicsTpl(
      std::shared_ptr<StateMultibody> state,
      std::shared_ptr<ActuationModelAbstract> actuation,
      std::shared_ptr<CostModelSum> costs);

  /**
   * @brief Initialize the free inverse-dynamics action model
   *
   * @param[in] state        State of the multibody system
   * @param[in] actuation    Actuation model
   * @param[in] costs        Cost model
   * @param[in] constraints  Constraints model
   */
  DifferentialActionModelFreeInvDynamicsTpl(
      std::shared_ptr<StateMultibody> state,
      std::shared_ptr<ActuationModelAbstract> actuation,
      std::shared_ptr<CostModelSum> costs,
      std::shared_ptr<ConstraintModelManager> constraints);
  virtual ~DifferentialActionModelFreeInvDynamicsTpl() = default;

  /**
   * @brief Compute the system acceleration, cost value and constraint residuals
   *
   * It extracts the acceleration value from control vector and also computes
   * the cost and constraints.
   *
   * @param[in] data  Free inverse-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<DifferentialActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief @copydoc Base::calc(const
   * std::shared_ptr<DifferentialActionDataAbstract>& data, const
   * Eigen::Ref<const VectorXs>& x)
   */
  virtual void calc(const std::shared_ptr<DifferentialActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Compute the derivatives of the dynamics, cost and constraint
   * functions
   *
   * It computes the partial derivatives of the dynamical system and the cost
   * and contraint functions. It assumes that `calc()` has been run first. This
   * function builds a quadratic approximation of the time-continuous action
   * model (i.e., dynamical system, cost and constraint functions).
   *
   * @param[in] data  Free inverse-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(
      const std::shared_ptr<DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x,
      const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief @copydoc Base::calcDiff(const
   * std::shared_ptr<DifferentialActionDataAbstract>& data, const
   * Eigen::Ref<const VectorXs>& x)
   */
  virtual void calcDiff(
      const std::shared_ptr<DifferentialActionDataAbstract>& data,
      const Eigen::Ref<const VectorXs>& x) override;

  /**
   * @brief Create the free inverse-dynamics data
   *
   * @return free inverse-dynamics data
   */
  virtual std::shared_ptr<DifferentialActionDataAbstract> createData() override;

  /**
   * @brief Cast the free-invdyn model to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return DifferentialActionModelFreeInvDynamicsTpl<NewScalar> A
   * differential-action model with the new scalar type.
   */
  template <typename NewScalar>
  DifferentialActionModelFreeInvDynamicsTpl<NewScalar> cast() const;

  /**
   * @brief Checks that a specific data belongs to the free inverse-dynamics
   * model
   */
  virtual bool checkData(
      const std::shared_ptr<DifferentialActionDataAbstract>& data) override;

  /**
   * @brief Computes the quasic static commands
   *
   * The quasic static commands are the ones produced for a reference posture as
   * an equilibrium point with zero acceleration, i.e., for
   * \f$\mathbf{f^q_x}\delta\mathbf{q}+\mathbf{f_u}\delta\mathbf{u}=\mathbf{0}\f$
   *
   * @param[in] data     Free inverse-dynamics data
   * @param[out] u       Quasic-static commands
   * @param[in] x        State point (velocity has to be zero)
   * @param[in] maxiter  Maximum allowed number of iterations (default 100)
   * @param[in] tol      Tolerance (default 1e-9)
   */
  virtual void quasiStatic(
      const std::shared_ptr<DifferentialActionDataAbstract>& data,
      Eigen::Ref<VectorXs> u, const Eigen::Ref<const VectorXs>& x,
      const std::size_t maxiter = 100,
      const Scalar tol = Scalar(1e-9)) override;

  /**
   * @brief Return the number of inequality constraints
   */
  virtual std::size_t get_ng() const override;

  /**
   * @brief Return the number of equality constraints
   */
  virtual std::size_t get_nh() const override;

  /**
   * @brief Return the number of equality terminal constraints
   */
  virtual std::size_t get_ng_T() const override;

  /**
   * @brief Return the number of equality terminal constraints
   */
  virtual std::size_t get_nh_T() const override;

  /**
   * @brief Return the lower bound of the inequality constraints
   */
  virtual const VectorXs& get_g_lb() const override;

  /**
   * @brief Return the upper bound of the inequality constraints
   */
  virtual const VectorXs& get_g_ub() const override;

  /**
   * @brief Return the actuation model
   */
  const std::shared_ptr<ActuationModelAbstract>& get_actuation() const;

  /**
   * @brief Return the cost model
   */
  const std::shared_ptr<CostModelSum>& get_costs() const;

  /**
   * @brief Return the constraint model manager
   */
  const std::shared_ptr<ConstraintModelManager>& get_constraints() const;

  /**
   * @brief Return the Pinocchio model
   */
  pinocchio::ModelTpl<Scalar>& get_pinocchio() const;

  /**
   * @brief Print relevant information of the free inverse-dynamics model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override;

 protected:
  using Base::g_lb_;   //!< Lower bound of the inequality constraints
  using Base::g_ub_;   //!< Upper bound of the inequality constraints
  using Base::ng_;     //!< Number of inequality constraints
  using Base::nh_;     //!< Number of equality constraints
  using Base::nu_;     //!< Control dimension
  using Base::state_;  //!< Model of the state

 private:
  void init(const std::shared_ptr<StateMultibody>& state);
  std::shared_ptr<ActuationModelAbstract> actuation_;    //!< Actuation model
  std::shared_ptr<CostModelSum> costs_;                  //!< Cost model
  std::shared_ptr<ConstraintModelManager> constraints_;  //!< Constraint model
  pinocchio::ModelTpl<Scalar>* pinocchio_;               //!< Pinocchio model

 public:
  /**
   * @brief Actuation residual
   *
   * This residual function enforces the torques of under-actuated joints (e.g.,
   * floating-base joints) to be zero. We compute these torques and their
   * derivatives using RNEA inside `DifferentialActionModelFreeInvDynamicsTpl`.
   *
   * As described in `ResidualModelAbstractTpl`, the residual value and its
   * Jacobians are calculated by `calc` and `calcDiff`, respectively.
   *
   * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
   */
  class ResidualModelActuation : public ResidualModelAbstractTpl<_Scalar> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CROCODDYL_INNER_DERIVED_CAST(ResidualModelBase,
                                 DifferentialActionModelFreeInvDynamicsTpl,
                                 ResidualModelActuation)

    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef ResidualModelAbstractTpl<Scalar> Base;
    typedef StateMultibodyTpl<Scalar> StateMultibody;
    typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
    typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
    typedef ActuationModelAbstractTpl<Scalar> ActuationModelAbstract;
    typedef typename MathBase::VectorXs VectorXs;
    typedef typename MathBase::MatrixXs MatrixXs;

    /**
     * @brief Initialize the actuation residual model
     *
     * @param[in] state  State of the multibody system
     * @param[in] nu     Dimension of the joint torques
     */
    ResidualModelActuation(std::shared_ptr<StateMultibody> state,
                           const std::size_t nu)
        : Base(state, state->get_nv() - nu, state->get_nv(), true, true, true),
          na_(nu) {}
    virtual ~ResidualModelActuation() = default;

    /**
     * @brief Compute the actuation residual
     *
     * @param[in] data  Actuation residual data
     * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
     * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nv+nu}\f$
     */
    virtual void calc(const std::shared_ptr<ResidualDataAbstract>& data,
                      const Eigen::Ref<const VectorXs>&,
                      const Eigen::Ref<const VectorXs>&) override {
      typename Data::ResidualDataActuation* d =
          static_cast<typename Data::ResidualDataActuation*>(data.get());
      // Update the under-actuation set and residual
      std::size_t nrow = 0;
      for (std::size_t k = 0;
           k < static_cast<std::size_t>(d->actuation->tau_set.size()); ++k) {
        if (!d->actuation->tau_set[k]) {
          data->r(nrow) = d->pinocchio->tau(k);
          nrow += 1;
        }
      }
    }

    /**
     * @brief @copydoc Base::calc(const std::shared_ptr<ResidualDataAbstract>&
     * data, const Eigen::Ref<const VectorXs>& x)
     */
    virtual void calc(const std::shared_ptr<ResidualDataAbstract>& data,
                      const Eigen::Ref<const VectorXs>&) override {
      data->r.setZero();
    }

    /**
     * @brief Compute the derivatives of the actuation residual
     *
     * @param[in] data  Actuation residual data
     * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
     * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
     */
    virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                          const Eigen::Ref<const VectorXs>&,
                          const Eigen::Ref<const VectorXs>&) override {
      typename Data::ResidualDataActuation* d =
          static_cast<typename Data::ResidualDataActuation*>(data.get());
      std::size_t nrow = 0;
      const std::size_t nv = state_->get_nv();
      d->dtau_dx.leftCols(nv) = d->pinocchio->dtau_dq;
      d->dtau_dx.rightCols(nv) = d->pinocchio->dtau_dv;
      d->dtau_dx -= d->actuation->dtau_dx;
      for (std::size_t k = 0;
           k < static_cast<std::size_t>(d->actuation->tau_set.size()); ++k) {
        if (!d->actuation->tau_set[k]) {
          d->Rx.row(nrow) = d->dtau_dx.row(k);
          d->Ru.row(nrow) = d->pinocchio->M.row(k);
          nrow += 1;
        }
      }
    }

    /**
     * @brief @copydoc Base::calcDiff(const
     * std::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const
     * VectorXs>& x)
     */
    virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                          const Eigen::Ref<const VectorXs>&) override {
      data->Rx.setZero();
      data->Ru.setZero();
    }

    /**
     * @brief Create the actuation residual data
     *
     * @return Actuation residual data
     */
    virtual std::shared_ptr<ResidualDataAbstract> createData(
        DataCollectorAbstract* const data) override {
      return std::allocate_shared<typename Data::ResidualDataActuation>(
          Eigen::aligned_allocator<typename Data::ResidualDataActuation>(),
          this, data);
    }

    /**
     * @brief Cast the actuation-residual model to a different scalar type.
     *
     * It is useful for operations requiring different precision or scalar
     * types.
     *
     * @tparam NewScalar The new scalar type to cast to.
     * @return typename
     * DifferentialActionModelFreeInvDynamicsTpl<NewScalar>::ResidualModelActuation
     * A residual model with the new scalar type.
     */
    template <typename NewScalar>
    typename DifferentialActionModelFreeInvDynamicsTpl<
        NewScalar>::ResidualModelActuation
    cast() const {
      typedef typename DifferentialActionModelFreeInvDynamicsTpl<
          NewScalar>::ResidualModelActuation ReturnType;
      typedef StateMultibodyTpl<NewScalar> StateType;
      ReturnType ret(std::static_pointer_cast<StateType>(
                         state_->template cast<NewScalar>()),
                     na_);
      return ret;
    }

    /**
     * @brief Print relevant information of the actuation residual model
     *
     * @param[out] os  Output stream object
     */
    virtual void print(std::ostream& os) const override {
      os << "ResidualModelActuation {nx=" << state_->get_nx()
         << ", ndx=" << state_->get_ndx() << ", nu=" << nu_ << ", na=" << na_
         << "}";
    }

   protected:
    std::size_t na_;  //!< Dimension of the joint torques
    using Base::nu_;
    using Base::state_;
  };
};

template <typename _Scalar>
struct DifferentialActionDataFreeInvDynamicsTpl
    : public DifferentialActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DifferentialActionDataAbstractTpl<Scalar> Base;
  typedef JointDataAbstractTpl<Scalar> JointDataAbstract;
  typedef DataCollectorJointActMultibodyTpl<Scalar>
      DataCollectorJointActMultibody;
  typedef CostDataSumTpl<Scalar> CostDataSum;
  typedef ConstraintDataManagerTpl<Scalar> ConstraintDataManager;
  typedef typename MathBase::VectorXs VectorXs;

  template <template <typename Scalar> class Model>
  explicit DifferentialActionDataFreeInvDynamicsTpl(Model<Scalar>* const model)
      : Base(model),
        pinocchio(pinocchio::DataTpl<Scalar>(model->get_pinocchio())),
        multibody(
            &pinocchio, model->get_actuation()->createData(),
            std::make_shared<JointDataAbstract>(
                model->get_state(), model->get_actuation(), model->get_nu())),
        costs(model->get_costs()->createData(&multibody)),
        constraints(model->get_constraints()->createData(&multibody)),
        tmp_xstatic(model->get_state()->get_nx()) {
    const std::size_t nv = model->get_state()->get_nv();
    Fu.leftCols(nv).diagonal().setOnes();
    multibody.joint->da_du.leftCols(nv).diagonal().setOnes();
    costs->shareMemory(this);
    constraints->shareMemory(this);
    tmp_xstatic.setZero();
  }
  virtual ~DifferentialActionDataFreeInvDynamicsTpl() = default;

  pinocchio::DataTpl<Scalar> pinocchio;                //!< Pinocchio data
  DataCollectorJointActMultibody multibody;            //!< Multibody data
  std::shared_ptr<CostDataSum> costs;                  //!< Costs data
  std::shared_ptr<ConstraintDataManager> constraints;  //!< Constraints data
  VectorXs
      tmp_xstatic;  //!< State point used for computing the quasi-static input
  using Base::cost;
  using Base::Fu;
  using Base::Fx;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::r;
  using Base::xout;

  struct ResidualDataActuation : public ResidualDataAbstractTpl<_Scalar> {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef ResidualDataAbstractTpl<Scalar> Base;
    typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
    typedef DataCollectorActMultibodyTpl<Scalar> DataCollectorActMultibody;
    typedef ActuationDataAbstractTpl<Scalar> ActuationDataAbstract;
    typedef typename MathBase::MatrixXs MatrixXs;

    template <template <typename Scalar> class Model>
    ResidualDataActuation(Model<Scalar>* const model,
                          DataCollectorAbstract* const data)
        : Base(model, data),
          dtau_dx(model->get_state()->get_nv(), model->get_state()->get_ndx()) {
      dtau_dx.setZero();
      // Check that proper shared data has been passed
      DataCollectorActMultibody* d =
          dynamic_cast<DataCollectorActMultibody*>(shared);
      if (d == NULL) {
        throw_pretty(
            "Invalid argument: the shared data should be derived from "
            "DataCollectorActMultibody");
      }

      // Avoids data casting at runtime
      pinocchio = d->pinocchio;
      actuation = d->actuation;
    }

    pinocchio::DataTpl<Scalar>* pinocchio;             //!< Pinocchio data
    std::shared_ptr<ActuationDataAbstract> actuation;  //!< Actuation data
    MatrixXs dtau_dx;
    using Base::r;
    using Base::Ru;
    using Base::Rx;
    using Base::shared;
  };
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include <crocoddyl/multibody/actions/free-invdyn.hxx>

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::DifferentialActionModelFreeInvDynamicsTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DifferentialActionDataFreeInvDynamicsTpl)

#endif  // CROCODDYL_MULTIBODY_ACTIONS_FREE_INVDYN_HPP_
