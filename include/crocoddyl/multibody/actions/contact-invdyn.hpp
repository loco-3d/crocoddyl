///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, Heriot-Watt University, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTIONS_CONTACT_INVDYN_HPP_
#define CROCODDYL_MULTIBODY_ACTIONS_CONTACT_INVDYN_HPP_

#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/constraints/constraint-manager.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/multibody/data/contacts.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

/**
 * @brief Differential action model for contact inverse dynamics in multibody
 * systems.
 *
 * This class implements forward kinematic with holonomic contact constraints
 * (defined at the acceleration level) and inverse-dynamics computation using
 * the Recursive Newton Euler Algorithm (RNEA). The stack of cost and constraint
 * functions are implemented in `CostModelSumTpl` and
 * `ConstraintModelManagerTpl`, respectively. The acceleration and contact
 * forces are decision variables defined as the control inputs, and the
 * under-actuation and contact constraint are under the name `tau` and its frame
 * name, thus the user is not allowed to use it.
 *
 * Additionally, it is important to note that `calcDiff()` computes the
 * derivatives using the latest stored values by `calc()`. Thus, we need to
 * first run `calc()`.
 *
 * \sa `DifferentialActionModelAbstractTpl`, `calc()`, `calcDiff()`,
 * `createData()`
 */
template <typename _Scalar>
class DifferentialActionModelContactInvDynamicsTpl
    : public DifferentialActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(DifferentialActionModelBase,
                         DifferentialActionModelContactInvDynamicsTpl)

  typedef _Scalar Scalar;
  typedef DifferentialActionModelAbstractTpl<Scalar> Base;
  typedef DifferentialActionDataContactInvDynamicsTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ActuationModelAbstractTpl<Scalar> ActuationModelAbstract;
  typedef CostModelSumTpl<Scalar> CostModelSum;
  typedef ConstraintModelManagerTpl<Scalar> ConstraintModelManager;
  typedef ContactModelMultipleTpl<Scalar> ContactModelMultiple;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef DifferentialActionDataAbstractTpl<Scalar>
      DifferentialActionDataAbstract;
  typedef ContactItemTpl<Scalar> ContactItem;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the contact inverse-dynamics action model
   *
   * It describes the kinematic evolution of the multibody system with contacts,
   * and computes the needed torques using inverse-dynamics.
   *
   * @param[in] state      State of the multibody system
   * @param[in] actuation  Actuation model
   * @param[in] contacts   Multiple contacts
   * @param[in] costs      Cost model
   */
  DifferentialActionModelContactInvDynamicsTpl(
      std::shared_ptr<StateMultibody> state,
      std::shared_ptr<ActuationModelAbstract> actuation,
      std::shared_ptr<ContactModelMultiple> contacts,
      std::shared_ptr<CostModelSum> costs);

  /**
   * @brief Initialize the contact inverse-dynamics action model
   *
   * @param[in] state        State of the multibody system
   * @param[in] actuation    Actuation model
   * @param[in] contacts     Multiple contacts
   * @param[in] costs        Cost model
   * @param[in] constraints  Constraints model
   */
  DifferentialActionModelContactInvDynamicsTpl(
      std::shared_ptr<StateMultibody> state,
      std::shared_ptr<ActuationModelAbstract> actuation,
      std::shared_ptr<ContactModelMultiple> contacts,
      std::shared_ptr<CostModelSum> costs,
      std::shared_ptr<ConstraintModelManager> constraints);
  virtual ~DifferentialActionModelContactInvDynamicsTpl() = default;

  /**
   * @brief Compute the system acceleration, cost value and constraint residuals
   *
   * It extracts the acceleration value from control vector and also computes
   * the cost and constraints.
   *
   * @param[in] data  Contact inverse-dynamics data
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
   * @param[in] data  Contact inverse-dynamics data
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
   * @brief Create the contact inverse-dynamics data
   *
   * @return contact inverse-dynamics data
   */
  virtual std::shared_ptr<DifferentialActionDataAbstract> createData() override;

  /**
   * @brief Cast the contact-invdyn model to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return DifferentialActionModelContactInvDynamicsTpl<NewScalar> A
   * differential-action model with the new scalar type.
   */
  template <typename NewScalar>
  DifferentialActionModelContactInvDynamicsTpl<NewScalar> cast() const;

  /**
   * @brief Checks that a specific data belongs to the contact inverse-dynamics
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
   * @param[in] data     Contact inverse-dynamics data
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
   * @brief Return the contact model
   */
  const std::shared_ptr<ContactModelMultiple>& get_contacts() const;

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
   * @brief Print relevant information of the contact inverse-dynamics model
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
  std::shared_ptr<ContactModelMultiple> contacts_;       //!< Contact model
  std::shared_ptr<CostModelSum> costs_;                  //!< Cost model
  std::shared_ptr<ConstraintModelManager> constraints_;  //!< Constraint model
  pinocchio::ModelTpl<Scalar>* pinocchio_;               //!< Pinocchio model

 public:
  /**
   * @brief Actuation residual
   *
   * This residual function enforces the torques of under-actuated joints (e.g.,
   * floating-base joints) to be zero. We compute these torques and their
   * derivatives using RNEA inside
   * `DifferentialActionModelContactInvDynamicsTpl`.
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
                                 DifferentialActionModelContactInvDynamicsTpl,
                                 ResidualModelActuation)

    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef ResidualModelAbstractTpl<Scalar> Base;
    typedef StateMultibodyTpl<Scalar> StateMultibody;
    typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
    typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
    typedef typename MathBase::VectorXs VectorXs;

    /**
     * @brief Initialize the actuation residual model
     *
     * @param[in] state  State of the multibody system
     * @param[in] nu     Dimension of the joint torques
     * @param[in] nc     Dimension of all the contacts
     */
    ResidualModelActuation(std::shared_ptr<StateMultibody> state,
                           const std::size_t nu, const std::size_t nc)
        : Base(state, state->get_nv() - nu, state->get_nv() + nc, true, true,
               true),
          na_(nu),
          nc_(nc) {}
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
      d->dtau_du.leftCols(nv) = d->pinocchio->M;
      d->dtau_du.rightCols(nc_) = -d->contact->Jc.transpose();
      for (std::size_t k = 0;
           k < static_cast<std::size_t>(d->actuation->tau_set.size()); ++k) {
        if (!d->actuation->tau_set[k]) {
          d->Rx.row(nrow) = d->dtau_dx.row(k);
          d->Ru.row(nrow) = d->dtau_du.row(k);
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
     * @return
     * DifferentialActionModelContactInvDynamicsTpl<NewScalar>::ResidualModelActuation
     * A residual model with the new scalar type.
     */
    template <typename NewScalar>
    typename DifferentialActionModelContactInvDynamicsTpl<
        NewScalar>::ResidualModelActuation
    cast() const {
      typedef typename DifferentialActionModelContactInvDynamicsTpl<
          NewScalar>::ResidualModelActuation ReturnType;
      typedef StateMultibodyTpl<NewScalar> StateType;
      ReturnType ret(std::static_pointer_cast<StateType>(
                         state_->template cast<NewScalar>()),
                     na_, nc_);
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
    using Base::nu_;
    using Base::state_;

   private:
    std::size_t na_;  //!< Number of actuated joints
    std::size_t nc_;  //!< Number of contacts
  };

 public:
  /**
   * @brief Contact-acceleration residual
   *
   * This residual function is defined as \f$\mathbf{r} = \mathbf{a_0}\f$, where
   * \f$\mathbf{a_0}\f$ defines the desired contact acceleration, which might
   * also include the Baumgarte stabilization gains. Furthermore, the Jacobians
   * of the residual function are computed analytically. This is used by
   * `ConstraintModelManagerTpl` inside parent
   * `DifferentialActionModelContactInvDynamicsTpl` class.
   *
   * As described in `ResidualModelAbstractTpl`, the residual value and its
   * Jacobians are calculated by `calc` and `calcDiff`, respectively.
   *
   * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
   */
  class ResidualModelContact : public ResidualModelAbstractTpl<_Scalar> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CROCODDYL_INNER_DERIVED_CAST(ResidualModelBase,
                                 DifferentialActionModelContactInvDynamicsTpl,
                                 ResidualModelContact)

    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef ResidualModelAbstractTpl<Scalar> Base;
    typedef StateMultibodyTpl<Scalar> StateMultibody;
    typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
    typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
    typedef typename MathBase::VectorXs VectorXs;
    typedef typename MathBase::MatrixXs MatrixXs;

    /**
     * @brief Initialize the contact-acceleration residual model
     *
     * @param[in] state  State of the multibody system
     * @param[in] id     Contact frame id
     * @param[in] nr     Dimension of the contact-acceleration residual
     * @param[in] nc     Dimension of all contacts
     */
    ResidualModelContact(std::shared_ptr<StateMultibody> state,
                         const pinocchio::FrameIndex id, const std::size_t nr,
                         const std::size_t nc)
        : Base(state, nr, state->get_nv() + nc, true, true, true),
          id_(id),
          frame_name_(state->get_pinocchio()->frames[id].name),
          nc_(nc) {}
    virtual ~ResidualModelContact() = default;

    /**
     * @brief Compute the contact-acceleration residual
     *
     * @param[in] data  Contact-acceleration residual data
     * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
     * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nv+nu}\f$
     */
    void calc(const std::shared_ptr<ResidualDataAbstract>& data,
              const Eigen::Ref<const VectorXs>&,
              const Eigen::Ref<const VectorXs>&) override {
      typename Data::ResidualDataContact* d =
          static_cast<typename Data::ResidualDataContact*>(data.get());
      d->r = d->contact->a0;
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
     * @brief Compute the derivatives of the contact-acceleration residual
     *
     * @param[in] data  Contact-acceleration residual data
     * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
     * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
     */
    void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                  const Eigen::Ref<const VectorXs>&,
                  const Eigen::Ref<const VectorXs>&) override {
      typename Data::ResidualDataContact* d =
          static_cast<typename Data::ResidualDataContact*>(data.get());
      d->Rx = d->contact->da0_dx;
      d->Ru.leftCols(state_->get_nv()) = d->contact->Jc;
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
     * @brief Create the contact-acceleration residual data
     *
     * @return contact-acceleration residual data
     */
    virtual std::shared_ptr<ResidualDataAbstract> createData(
        DataCollectorAbstract* const data) override {
      return std::allocate_shared<typename Data::ResidualDataContact>(
          Eigen::aligned_allocator<typename Data::ResidualDataContact>(), this,
          data, id_);
    }

    /**
     * @brief Cast the contact-residual model to a different scalar type.
     *
     * It is useful for operations requiring different precision or scalar
     * types.
     *
     * @tparam NewScalar The new scalar type to cast to.
     * @return typename
     * DifferentialActionModelContactInvDynamicsTpl<NewScalar>::ResidualModelContact
     * A residual model with the new scalar type.
     */
    template <typename NewScalar>
    typename DifferentialActionModelContactInvDynamicsTpl<
        NewScalar>::ResidualModelContact
    cast() const {
      typedef typename DifferentialActionModelContactInvDynamicsTpl<
          NewScalar>::ResidualModelContact ReturnType;
      typedef StateMultibodyTpl<NewScalar> StateType;
      ReturnType ret(std::static_pointer_cast<StateType>(
                         state_->template cast<NewScalar>()),
                     id_, nr_, nc_);
      return ret;
    }

    /**
     * @brief Print relevant information of the contact-acceleration residual
     * model
     *
     * @param[out] os  Output stream object
     */
    virtual void print(std::ostream& os) const override {
      os << "ResidualModelContact {frame=" << frame_name_ << ", nr=" << nr_
         << "}";
    }

   protected:
    using Base::nr_;
    using Base::nu_;
    using Base::state_;

   private:
    pinocchio::FrameIndex id_;  //!< Reference frame id
    std::string frame_name_;    //!< Reference frame name
    std::size_t nc_;            //!< Dimension of all contacts
  };
};
template <typename _Scalar>
struct DifferentialActionDataContactInvDynamicsTpl
    : public DifferentialActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DifferentialActionDataAbstractTpl<Scalar> Base;
  typedef JointDataAbstractTpl<Scalar> JointDataAbstract;
  typedef DataCollectorJointActMultibodyInContactTpl<Scalar>
      DataCollectorJointActMultibodyInContact;
  typedef CostDataSumTpl<Scalar> CostDataSum;
  typedef ConstraintDataManagerTpl<Scalar> ConstraintDataManager;
  typedef ContactModelMultipleTpl<Scalar> ContactModelMultiple;
  typedef ContactItemTpl<Scalar> ContactItem;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit DifferentialActionDataContactInvDynamicsTpl(
      Model<Scalar>* const model)
      : Base(model),
        pinocchio(pinocchio::DataTpl<Scalar>(model->get_pinocchio())),
        multibody(
            &pinocchio, model->get_actuation()->createData(),
            std::make_shared<JointDataAbstract>(
                model->get_state(), model->get_actuation(), model->get_nu()),
            model->get_contacts()->createData(&pinocchio)),
        tmp_xstatic(model->get_state()->get_nx()),
        tmp_rstatic(model->get_actuation()->get_nu() +
                    model->get_contacts()->get_nc()),
        tmp_Jstatic(model->get_state()->get_nv(),
                    model->get_actuation()->get_nu() +
                        model->get_contacts()->get_nc()) {
    // Set constant values for Fu, df_dx, and df_du
    const std::size_t nv = model->get_state()->get_nv();
    const std::size_t nc = model->get_contacts()->get_nc_total();
    Fu.leftCols(nv).diagonal().setOnes();
    multibody.joint->da_du.leftCols(nv).diagonal().setOnes();
    MatrixXs df_dx = MatrixXs::Zero(nc, model->get_state()->get_ndx());
    MatrixXs df_du = MatrixXs::Zero(nc, model->get_nu());
    std::size_t fid = 0;
    for (typename ContactModelMultiple::ContactModelContainer::const_iterator
             it = model->get_contacts()->get_contacts().begin();
         it != model->get_contacts()->get_contacts().end(); ++it) {
      const std::size_t nc = it->second->contact->get_nc();
      df_du.block(fid, nv + fid, nc, nc).diagonal().setOnes();
      fid += nc;
    }
    std::vector<bool> contact_status;
    for (typename ContactModelMultiple::ContactModelContainer::const_iterator
             it = model->get_contacts()->get_contacts().begin();
         it != model->get_contacts()->get_contacts().end(); ++it) {
      const std::shared_ptr<ContactItem>& m_i = it->second;
      contact_status.push_back(m_i->active);
      m_i->active = true;
    }
    model->get_contacts()->updateForceDiff(multibody.contacts, df_dx, df_du);
    std::size_t cid = 0;
    for (typename ContactModelMultiple::ContactModelContainer::const_iterator
             it = model->get_contacts()->get_contacts().begin();
         it != model->get_contacts()->get_contacts().end(); ++it) {
      const std::shared_ptr<ContactItem>& m_i = it->second;
      m_i->active = contact_status[cid];
      cid++;
    }
    costs = model->get_costs()->createData(&multibody);
    constraints = model->get_constraints()->createData(&multibody);
    costs->shareMemory(this);
    constraints->shareMemory(this);

    tmp_xstatic.setZero();
    tmp_rstatic.setZero();
    tmp_Jstatic.setZero();
  }
  virtual ~DifferentialActionDataContactInvDynamicsTpl() = default;

  pinocchio::DataTpl<Scalar> pinocchio;                //!< Pinocchio data
  DataCollectorJointActMultibodyInContact multibody;   //!< Multibody data
  std::shared_ptr<CostDataSum> costs;                  //!< Costs data
  std::shared_ptr<ConstraintDataManager> constraints;  //!< Constraints data
  VectorXs
      tmp_xstatic;  //!< State point used for computing the quasi-static input
  VectorXs
      tmp_rstatic;  //!< Factorization used for computing the quasi-static input
  MatrixXs tmp_Jstatic;  //!< Jacobian used for computing the quasi-static input

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
    typedef DataCollectorActMultibodyInContactTpl<Scalar>
        DataCollectorActMultibodyInContact;
    typedef ActuationDataAbstractTpl<Scalar> ActuationDataAbstract;
    typedef ContactDataMultipleTpl<Scalar> ContactDataMultiple;
    typedef typename MathBase::MatrixXs MatrixXs;

    template <template <typename Scalar> class Model>
    ResidualDataActuation(Model<Scalar>* const model,
                          DataCollectorAbstract* const data)
        : Base(model, data),
          dtau_dx(model->get_state()->get_nv(), model->get_state()->get_ndx()),
          dtau_du(model->get_state()->get_nv(), model->get_nu()) {
      // Check that proper shared data has been passed
      DataCollectorActMultibodyInContact* d =
          dynamic_cast<DataCollectorActMultibodyInContact*>(shared);
      if (d == NULL) {
        throw_pretty(
            "Invalid argument: the shared data should be derived from "
            "DataCollectorActMultibodyInContact");
      }
      // Avoids data casting at runtime
      pinocchio = d->pinocchio;
      actuation = d->actuation;
      contact = d->contacts;
      dtau_dx.setZero();
      dtau_du.setZero();
    }
    virtual ~ResidualDataActuation() = default;

    pinocchio::DataTpl<Scalar>* pinocchio;             //!< Pinocchio data
    std::shared_ptr<ActuationDataAbstract> actuation;  //!< Actuation data
    std::shared_ptr<ContactDataMultiple> contact;      //!< Contact data
    MatrixXs dtau_dx;
    MatrixXs dtau_du;
    using Base::r;
    using Base::Ru;
    using Base::Rx;
    using Base::shared;
  };

  struct ResidualDataContact : public ResidualDataAbstractTpl<_Scalar> {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef ResidualDataAbstractTpl<Scalar> Base;
    typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
    typedef DataCollectorMultibodyInContactTpl<Scalar>
        DataCollectorMultibodyInContact;
    typedef ContactModelMultipleTpl<Scalar> ContactModelMultiple;

    template <template <typename Scalar> class Model>
    ResidualDataContact(Model<Scalar>* const model,
                        DataCollectorAbstract* const data, const std::size_t id)
        : Base(model, data) {
      DataCollectorMultibodyInContact* d =
          dynamic_cast<DataCollectorMultibodyInContact*>(shared);
      if (d == NULL) {
        throw_pretty(
            "Invalid argument: the shared data should be derived from "
            "DataCollectorMultibodyInContact");
      }
      typename ContactModelMultiple::ContactDataContainer::iterator it, end;
      for (it = d->contacts->contacts.begin(),
          end = d->contacts->contacts.end();
           it != end; ++it) {
        if (id == it->second->frame) {  // TODO(cmastalli): use model->get_id()
                                        // and avoid to pass id in constructor
          contact = it->second.get();
          break;
        }
      }
    }
    virtual ~ResidualDataContact() = default;

    ContactDataAbstractTpl<Scalar>* contact;  //!< Contact force data
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
#include <crocoddyl/multibody/actions/contact-invdyn.hxx>

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::DifferentialActionModelContactInvDynamicsTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::DifferentialActionDataContactInvDynamicsTpl)

#endif  // CROCODDYL_MULTIBODY_ACTIONS_CONTACT_INVDYN_HPP_
