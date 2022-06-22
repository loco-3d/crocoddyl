///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2022, Heriot-Watt University, University of Edinburgh,
//                          University of Pisa
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTIONS_CONTACT_INVDYN_REDUNDANT_HPP_
#define CROCODDYL_MULTIBODY_ACTIONS_CONTACT_INVDYN_REDUNDANT_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/constraints/constraint-manager.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/multibody/data/contacts.hpp"

namespace crocoddyl {

/**
 * @brief Differential action model for contact inverse dynamics in multibody systems.
 *
 * This class implements forward kinematic with an inverse-dynamics equality constraint computed using the Recursive
 * Newton Euler Algorithm (RNEA) and contact holonomic constraints (defined at the acceleration level). The stack of
 * cost and constraint functions are implemented in `CostModelSumTpl` and `ConstraintModelManagerTpl`, respectively.
 * The acceleration, torques and contact forces are decision variables defined as the control inputs, and the RNEA
 * and contact constraint are under the name `rnea` and its frame name, thus the user is not allow to use it.
 *
 * Additionally, it is important remark that `calcDiff()` computes the derivatives using the latest stored values by
 * `calc()`. Thus, we need to run first `calc()`.
 *
 * \sa `DifferentialActionModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class DifferentialActionModelContactInvDynamicsRedundantTpl : public DifferentialActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef DifferentialActionModelAbstractTpl<Scalar> Base;
  typedef DifferentialActionDataContactInvDynamicsRedundantTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ActuationModelAbstractTpl<Scalar> ActuationModelAbstract;
  typedef CostModelSumTpl<Scalar> CostModelSum;
  typedef ConstraintModelManagerTpl<Scalar> ConstraintModelManager;
  typedef ContactModelMultipleTpl<Scalar> ContactModelMultiple;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef DifferentialActionDataAbstractTpl<Scalar> DifferentialActionDataAbstract;
  typedef ContactItemTpl<Scalar> ContactItem;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBaseTpl<double>::MatrixXsRowMajor MatrixXdRowMajor;

  /**
   * @brief Initialize the contact inverse-dynamics action model
   *
   * It describes the kinematic evolution of the multibody system with contacts,
   * and imposes inverse-dynamics and contact-accelerations (equality) constraints.
   * Additionally, it computes the cost and extra constraint values associated to
   * this state and control pair. Note that the name `rnea` in the `ConstraintModelManagerTpl`
   * is reserved to store the inverse-dynamics constraint. We also reserve the name of the
   * contact frame to define the contact-acceleration constraints.
   *
   * @param[in] state      State of the multibody system
   * @param[in] actuation  Actuation model
   * @param[in] contacts   Multiple contacts
   * @param[in] costs      Cost model
   */
  DifferentialActionModelContactInvDynamicsRedundantTpl(boost::shared_ptr<StateMultibody> state,
                                                        boost::shared_ptr<ActuationModelAbstract> actuation,
                                                        boost::shared_ptr<ContactModelMultiple> contacts,
                                                        boost::shared_ptr<CostModelSum> costs);

  /**
   * @brief Initialize the contact inverse-dynamics action model
   *
   * @param[in] state        State of the multibody system
   * @param[in] actuation    Actuation model
   * @param[in] contacts   Multiple contacts
   * @param[in] costs        Cost model
   * @param[in] constraints  Constraints model
   */
  DifferentialActionModelContactInvDynamicsRedundantTpl(boost::shared_ptr<StateMultibody> state,
                                                        boost::shared_ptr<ActuationModelAbstract> actuation,
                                                        boost::shared_ptr<ContactModelMultiple> contacts,
                                                        boost::shared_ptr<CostModelSum> costs,
                                                        boost::shared_ptr<ConstraintModelManager> constraints);
  virtual ~DifferentialActionModelContactInvDynamicsRedundantTpl();

  /**
   * @brief Compute the system acceleration, cost value and constraint residuals
   *
   * It extracts the acceleration value from control vector and also computes the cost and constraints.
   *
   * @param[in] data  Free inverse-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief @copydoc Base::calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const
   * VectorXs>& x)
   */
  virtual void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Compute the derivatives of the dynamics, cost and constraint functions
   *
   * It computes the partial derivatives of the dynamical system and the cost and contraint functions.
   * It assumes that `calc()` has been run first. This function builds a quadratic approximation of the
   * time-continuous action model (i.e., dynamical system, cost and constraint functions).
   *
   * @param[in] data  Free inverse-dynamics data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief @copydoc Base::calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const
   * Eigen::Ref<const VectorXs>& x)
   */
  virtual void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x);

  /**
   * @brief Create the contact inverse-dynamics data
   *
   * @return contact inverse-dynamics data
   */
  virtual boost::shared_ptr<DifferentialActionDataAbstract> createData();

  /**
   * @brief Checks that a specific data belongs to the contact inverse-dynamics model
   */
  virtual bool checkData(const boost::shared_ptr<DifferentialActionDataAbstract>& data);

  /**
   * @brief Computes the quasic static commands
   *
   * The quasic static commands are the ones produced for a the reference posture as an equilibrium point with zero
   * acceleration, i.e., for \f$\mathbf{f^q_x}\delta\mathbf{q}+\mathbf{f_u}\delta\mathbf{u}=\mathbf{0}\f$
   *
   * @param[in] data     Action data
   * @param[out] u       Quasic static commands
   * @param[in] x        State point (velocity has to be zero)
   * @param[in] maxiter  Maximum allowed number of iterations (default 100)
   * @param[in] tol      Tolerance (default 1e-9)
   */
  virtual void quasiStatic(const boost::shared_ptr<DifferentialActionDataAbstract>& data, Eigen::Ref<VectorXs> u,
                           const Eigen::Ref<const VectorXs>& x, const std::size_t maxiter = 100,
                           const Scalar tol = Scalar(1e-9));

  /**
   * @brief Compute the Sparse product between the given matrix A and the Jacobian of the dynamics with respect to the
   * control
   *
   * It assumes that `calcDiff()` has been run first
   *
   * @param[in] Fu    Jacobian matrix of the dynamics with respect to the control
   * @param[in] A     A matrix to multiply times the Jacobian
   * @param[out] out  Product between A and the Jacobian of the dynamics with respect to the control
   * @param[in] op    Assignment operator which sets, adds, or removes the given results
   */
  virtual void multiplyByFu(const Eigen::Ref<const MatrixXs>& Fu, const Eigen::Ref<const MatrixXs>& A,
                            Eigen::Ref<MatrixXs> out, const AssignmentOp = setto) const;

  /**
   * @brief Compute the Sparse product between the Jacobian of the dynamics with respect to the
   * control and the given matrix A
   *
   * It assumes that `calcDiff()` has been run first
   *
   * @param[in] Fu           Jacobian matrix of the dynamics with respect to the control
   * @param[in] A            A matrix to multiply times the Jacobian
   * @param[out] out         Product between A and the Jacobian of the dynamics with respect to the control
   * @param[in] op           Assignment operator which sets, adds, or removes the given results
   */
  virtual void multiplyFuTransposeBy(const Eigen::Ref<const MatrixXs>& Fu, const Eigen::Ref<const MatrixXs>& A,
                                     Eigen::Ref<MatrixXdRowMajor> out, const AssignmentOp = setto) const;

  /**
   * @brief Return the actuation model
   */
  const boost::shared_ptr<ActuationModelAbstract>& get_actuation() const;

  /**
   * @brief Return the contact model
   */
  const boost::shared_ptr<ContactModelMultiple>& get_contacts() const;

  /**
   * @brief Return the cost model
   */
  const boost::shared_ptr<CostModelSum>& get_costs() const;

  /**
   * @brief Return the constraint model
   */
  const boost::shared_ptr<ConstraintModelManager>& get_constraints() const;

  /**
   * @brief Return the Pinocchio model
   */
  pinocchio::ModelTpl<Scalar>& get_pinocchio() const;

  /**
   * @brief Print relevant information of the contact inverse-dynamics model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 protected:
  using Base::ng_;     //!< Number of inequality constraints
  using Base::nh_;     //!< Number of equality constraints
  using Base::nu_;     //!< Control dimension
  using Base::state_;  //!< Model of the state

 private:
  void init(const boost::shared_ptr<StateMultibody>& state);
  boost::shared_ptr<ActuationModelAbstract> actuation_;    //!< Actuation model
  boost::shared_ptr<ContactModelMultiple> contacts_;       //!< Contact model
  boost::shared_ptr<CostModelSum> costs_;                  //!< Cost model
  boost::shared_ptr<ConstraintModelManager> constraints_;  //!< Constraint model
  pinocchio::ModelTpl<Scalar>& pinocchio_;                 //!< Pinocchio model

 public:
  /**
   * @brief RNEA residual
   *
   * This residual function is defined as \f$\mathbf{r} = \boldsymbol{\tau} -
   * \mathrm{RNEA}(\mathbf{q},\mathbf{v},\dot{\mathbf{a}},\boldsymbol{\lambda})\f$, where \f$\boldsymbol{\tau}\f$
   * is extracted from the control vector and \f$\mathrm{RNEA}\f$ evaluates the joint torque using \f$\mathbf{q},
   * \mathbf{v}, \dot{\mathbf{a}}, \boldsymbol{\lambda}\f$ values. Furthermore, the Jacobians of the residual
   * function are computed analytically. This is used by `ConstraintModelManagerTpl` inside parent
   * `DifferentialActionModelContactInvDynamicsRedundantTpl` class.
   *
   * As described in `ResidualModelAbstractTpl`, the residual value and its Jacobians are calculated by `calc` and
   * `calcDiff`, respectively.
   *
   * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
   */
  class ResidualModelRnea : public ResidualModelAbstractTpl<_Scalar> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef ResidualModelAbstractTpl<Scalar> Base;
    typedef StateMultibodyTpl<Scalar> StateMultibody;
    typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
    typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
    typedef typename MathBase::VectorXs VectorXs;

    /**
     * @brief Initialize the RNEA residual model
     *
     * @param[in] state  State of the multibody system
     * @param[in] nc     Dimension of all the contacts
     * @param[in] nu     Dimension of the control vector
     */
    ResidualModelRnea(boost::shared_ptr<StateMultibody> state, const std::size_t nc, const std::size_t nu)
        : Base(state, state->get_nv(), state->get_nv() + nu + nc, true, true, true), nc_(nc), na_(nu) {}
    virtual ~ResidualModelRnea() {}

    /**
     * @brief Compute the RNEA residual
     *
     * @param[in] data  RNEA residual data
     * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
     * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nv+nu}\f$
     */
    virtual void calc(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>&,
                      const Eigen::Ref<const VectorXs>&) {
      typename Data::ResidualDataRnea* d = static_cast<typename Data::ResidualDataRnea*>(data.get());
      data->r = d->pinocchio->tau - d->actuation->tau;
    }

    /**
     * @brief @copydoc Base::calc(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const
     * VectorXs>& x)
     */
    virtual void calc(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>&) {
      data->r.setZero();
    }

    /**
     * @brief Compute the derivatives of the RNEA residual
     *
     * @param[in] data  RNEA residual data
     * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
     * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
     */
    virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>&,
                          const Eigen::Ref<const VectorXs>&) {
      typename Data::ResidualDataRnea* d = static_cast<typename Data::ResidualDataRnea*>(data.get());
      const std::size_t nv = state_->get_nv();
      data->Rx.leftCols(nv) = d->pinocchio->dtau_dq;
      data->Rx.rightCols(nv) = d->pinocchio->dtau_dv;
      data->Rx -= d->actuation->dtau_dx;
      data->Ru.leftCols(nv) = d->pinocchio->M;
      data->Ru.middleCols(nv, na_) = -d->actuation->dtau_du;
      data->Ru.rightCols(nc_) = -d->contact->Jc.transpose();
    }

    /**
     * @brief @copydoc Base::calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const
     * VectorXs>& x)
     */
    virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>&) {
      data->Rx.setZero();
      data->Ru.setZero();
    }

    /**
     * @brief Create the RNEA residual data
     *
     * @return RNEA residual data
     */
    virtual boost::shared_ptr<ResidualDataAbstract> createData(DataCollectorAbstract* const data) {
      return boost::allocate_shared<typename Data::ResidualDataRnea>(
          Eigen::aligned_allocator<typename Data::ResidualDataRnea>(), this, data);
    }

    /**
     * @brief Print relevant information of the RNEA residual model
     *
     * @param[out] os  Output stream object
     */
    virtual void print(std::ostream& os) const {
      os << "ResidualModelRnea {nx=" << state_->get_nx() << ", ndx=" << state_->get_ndx() << ", nu=" << nu_
         << "nc=" << nc_;
    }

   protected:
    using Base::nu_;
    using Base::state_;

   private:
    std::size_t nc_;  //!< Number of all contacts
    std::size_t na_;  //!< Number of actuated joints
  };

 public:
  /**
   * @brief Contact-acceleration residual
   *
   * This residual function is defined as \f$\mathbf{r} = \mathbf{a_0}\f$, where \f$\mathbf{a_0}\f$ defines the
   * desired contact acceleration, which might also include the Baumgarte stabilization gains. Furthermore, the
   * Jacobians of the residual function are computed analytically. This is used by `ConstraintModelManagerTpl`
   * inside parent `DifferentialActionModelContactInvDynamicsRedundantTpl` class.
   *
   * As described in `ResidualModelAbstractTpl`, the residual value and its Jacobians are calculated by `calc` and
   * `calcDiff`, respectively.
   *
   * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
   */
  class ResidualModelContact : public ResidualModelAbstractTpl<_Scalar> {
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

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
     * @param[in] nu     Dimension of the control vector
     */
    ResidualModelContact(boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id, const std::size_t nr,
                         const std::size_t nc, const std::size_t nu)
        : Base(state, nr, state->get_nv() + nu + nc, true, true, true),
          id_(id),
          frame_name_(state->get_pinocchio()->frames[id].name) {}
    virtual ~ResidualModelContact() {}

    /**
     * @brief Compute the contact-acceleration residual
     *
     * @param[in] data  Contact-acceleration residual data
     * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
     * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nv+nu}\f$
     */
    void calc(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>&,
              const Eigen::Ref<const VectorXs>&) {
      typename Data::ResidualDataContact* d = static_cast<typename Data::ResidualDataContact*>(data.get());
      d->r = d->contact->a0;
    }

    /**
     * @brief @copydoc Base::calc(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const
     * VectorXs>& x)
     */
    virtual void calc(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>&) {
      data->r.setZero();
    }

    /**
     * @brief Compute the derivatives of the contact-acceleration residual
     *
     * @param[in] data  Contact-acceleration residual data
     * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
     * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
     */
    void calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>&,
                  const Eigen::Ref<const VectorXs>&) {
      typename Data::ResidualDataContact* d = static_cast<typename Data::ResidualDataContact*>(data.get());
      d->Rx = d->contact->da0_dx;
      d->Ru.leftCols(state_->get_nv()) = d->contact->Jc;
    }

    /**
     * @brief @copydoc Base::calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const
     * VectorXs>& x)
     */
    virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>&) {
      data->Rx.setZero();
      data->Ru.setZero();
    }

    /**
     * @brief Create the contact-acceleration residual data
     *
     * @return contact-acceleration residual data
     */
    virtual boost::shared_ptr<ResidualDataAbstract> createData(DataCollectorAbstract* const data) {
      return boost::allocate_shared<typename Data::ResidualDataContact>(
          Eigen::aligned_allocator<typename Data::ResidualDataContact>(), this, data, id_);
    }

    /**
     * @brief Print relevant information of the contact-acceleration residual model
     *
     * @param[out] os  Output stream object
     */
    virtual void print(std::ostream& os) const {
      os << "ResidualModelContact {frame=" << frame_name_ << ", nr=" << nr_ << "}";
    }

   protected:
    using Base::nr_;
    using Base::nu_;
    using Base::state_;
    using Base::unone_;

   private:
    pinocchio::FrameIndex id_;  //!< Reference frame id
    std::string frame_name_;    //!< Reference frame name
  };
};
template <typename _Scalar>
struct DifferentialActionDataContactInvDynamicsRedundantTpl : public DifferentialActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DifferentialActionDataAbstractTpl<Scalar> Base;
  typedef ContactModelMultipleTpl<Scalar> ContactModelMultiple;
  typedef DataCollectorActMultibodyInContactTpl<Scalar> DataCollectorActMultibodyInContact;
  typedef CostDataSumTpl<Scalar> CostDataSum;
  typedef ConstraintDataManagerTpl<Scalar> ConstraintDataManager;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  explicit DifferentialActionDataContactInvDynamicsRedundantTpl(Model<Scalar>* const model)
      : Base(model),
        pinocchio(pinocchio::DataTpl<Scalar>(model->get_pinocchio())),
        multibody(&pinocchio, model->get_actuation()->createData(), model->get_contacts()->createData(&pinocchio)),
        tmp_xstatic(model->get_state()->get_nx()),
        tmp_rstatic(model->get_actuation()->get_nu() + model->get_contacts()->get_nc()),
        tmp_Jstatic(model->get_state()->get_nv(), model->get_actuation()->get_nu() + model->get_contacts()->get_nc()) {
    // Set constant values for Fu, df_dx, and df_du
    const std::size_t na = model->get_actuation()->get_nu();
    const std::size_t nv = model->get_state()->get_nv();
    const std::size_t nc = model->get_contacts()->get_nc_total();
    Fu.leftCols(nv).diagonal().setOnes();
    MatrixXs df_dx = MatrixXs::Zero(nc, model->get_state()->get_ndx());
    MatrixXs df_du = MatrixXs::Zero(nc, model->get_nu());
    std::size_t fid = 0;
    for (typename ContactModelMultiple::ContactModelContainer::const_iterator it =
             model->get_contacts()->get_contacts().begin();
         it != model->get_contacts()->get_contacts().end(); ++it) {
      const std::size_t nc = it->second->contact->get_nc();
      df_du.block(fid, nv + na + fid, nc, nc).diagonal().setOnes();
      fid += nc;
    }
    model->get_contacts()->updateForceDiff(multibody.contacts, df_dx, df_du);

    costs = model->get_costs()->createData(&multibody);
    constraints = model->get_constraints()->createData(&multibody);
    costs->shareMemory(this);
    constraints->shareMemory(this);

    tmp_xstatic.setZero();
    tmp_rstatic.setZero();
    tmp_Jstatic.setZero();
  }

  pinocchio::DataTpl<Scalar> pinocchio;                  //!< Pinocchio data
  DataCollectorActMultibodyInContact multibody;          //!< Multibody data
  boost::shared_ptr<CostDataSum> costs;                  //!< Costs data
  boost::shared_ptr<ConstraintDataManager> constraints;  //!< Constraints data
  VectorXs tmp_xstatic;                                  //!< State point used for computing the quasi-static input
  VectorXs tmp_rstatic;                                  //!< Factorization used for computing the quasi-static input
  MatrixXs tmp_Jstatic;                                  //!< Jacobian used for computing the quasi-static input

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

  struct ResidualDataRnea : public ResidualDataAbstractTpl<_Scalar> {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    typedef _Scalar Scalar;
    typedef MathBaseTpl<Scalar> MathBase;
    typedef ResidualDataAbstractTpl<Scalar> Base;
    typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
    typedef DataCollectorActMultibodyInContactTpl<Scalar> DataCollectorActMultibodyInContact;

    template <template <typename Scalar> class Model>
    ResidualDataRnea(Model<Scalar>* const model, DataCollectorAbstract* const data) : Base(model, data) {
      // Check that proper shared data has been passed
      DataCollectorActMultibodyInContact* d = dynamic_cast<DataCollectorActMultibodyInContact*>(shared);
      if (d == NULL) {
        throw_pretty("Invalid argument: the shared data should be derived from DataCollectorActMultibodyInContact");
      }
      // Avoids data casting at runtime
      pinocchio = d->pinocchio;
      actuation = d->actuation.get();
      contact = d->contacts.get();
    }

    pinocchio::DataTpl<Scalar>* pinocchio;        //!< Pinocchio data
    ActuationDataAbstractTpl<Scalar>* actuation;  //!< Actuation data
    ContactDataMultipleTpl<Scalar>* contact;      //!< Contact data
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
    typedef DataCollectorMultibodyInContactTpl<Scalar> DataCollectorMultibodyInContact;
    typedef ContactModelMultipleTpl<Scalar> ContactModelMultiple;

    template <template <typename Scalar> class Model>
    ResidualDataContact(Model<Scalar>* const model, DataCollectorAbstract* const data, const std::size_t id)
        : Base(model, data) {
      DataCollectorMultibodyInContact* d = dynamic_cast<DataCollectorMultibodyInContact*>(shared);
      if (d == NULL) {
        throw_pretty("Invalid argument: the shared data should be derived from DataCollectorMultibodyInContact");
      }
      typename ContactModelMultiple::ContactDataContainer::iterator it, end;
      for (it = d->contacts->contacts.begin(), end = d->contacts->contacts.end(); it != end; ++it) {
        if (id == it->second->frame) {  // TODO(cmastalli): use model->get_id() and avoid to pass id in constructor
          contact = it->second.get();
          break;
        }
      }
    }

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
#include <crocoddyl/multibody/actions/contact-invdyn-redundant.hxx>

#endif  // CROCODDYL_MULTIBODY_ACTIONS_CONTACT_INVDYN_REDUNDANT_HPP_
