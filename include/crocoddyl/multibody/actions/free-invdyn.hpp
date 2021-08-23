///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh, University of Pisa
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_ACTIONS_FREE_INVDYN_HPP_
#define CROCODDYL_MULTIBODY_ACTIONS_FREE_INVDYN_HPP_

#include <stdexcept>

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/diff-action-base.hpp"
#include "crocoddyl/core/costs/cost-sum.hpp"
#include "crocoddyl/core/constraints/constraint-manager.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

template <typename _Scalar>
/**
 * @brief Differential action model for free inverse dynamics in multibody systems.
 * This class implements  the dynamics using Recursive Newton Euler Algorithm (RNEA) as an equality constraint.\n"
 * The stack of cost and constraint functions are implemented in\n"
 * ConstraintModelManager() and CostModelSum(), respectively.
 * In Crocoddyl, a differential action model combines dynamics, cost and constraints models. We can use it in each node
 * of our optimal control problem thanks to dedicated integration rules (e.g. `IntegratedActionModelEulerTpl` or
 * `IntegratedActionModelRK4Tpl`). These integrated action models produces action models (`ActionModelAbstractTpl`).
 * Thus, every time that we want describe a problem, we need to provide ways of computing the dynamics, cost,
 * constraints functions and their derivatives. All these is described inside the differential action model.
 *
 *  The differential action model is the time-continuous version of an action model, i.e.
 * \f[
 * \begin{aligned}
 * &\dot{\mathbf{v}} = \mathbf{f}(\mathbf{q}, \mathbf{v}, \mathbf{u}), &\textrm{(dynamics)}\\
 * &l(\mathbf{q}, \mathbf{v},\mathbf{u}) = \int_0^{\delta t} a(\mathbf{r}(\mathbf{q}, \mathbf{v},\mathbf{u}))\,dt,
 * &\textrm{(cost)}\\
 * &\mathbf{g}(\mathbf{q}, \mathbf{v},\mathbf{u})<\mathbf{0}, &\textrm{(inequality constraint)}\\
 * &\mathbf{h}(\mathbf{q}, \mathbf{v},\mathbf{u})=\mathbf{0}, &\textrm{(equality constraint)}
 * \end{aligned}
 * \f]
 * where
 *  - the configuration \f$\mathbf{q}\in\mathcal{Q}\f$ lies in the configuration manifold described with a `nq`-tuple,
 *  - the velocity \f$\mathbf{v}\in T_{\mathbf{q}}\mathcal{Q}\f$ its a tangent vector to this manifold with `nv`
 * dimension,
 *  - the control input \f$\mathbf{u} }= (\mathbf{a},\mathbf{\tau}) \in\mathbb{R}^{nu+nv}\f$ is an Euclidean vector,
 *  - \f$\mathbf{r}(\cdot)\f$ and \f$a(\cdot)\f$ are the residual and activation functions (see
 * `ActivationModelAbstractTpl`),
 *  - \f$\mathbf{g}(\cdot)\in\mathbb{R}^{ng}\f$ and \f$\mathbf{h}(\cdot)\in\mathbb{R}^{nh}\f$ are the inequality and
 * equality vector functions, respectively.
 *
 * Both configuration and velocity describe the system space \f$\mathbf{x}=(\mathbf{q}, \mathbf{v})\in\mathbf{X}\f$
 * which lies in the state manifold.  * Note that the acceleration \f$\dot{\mathbf{v}}\in T_{\mathbf{q}}\mathcal{Q}\f$
 * lies also in the tangent space of the configuration manifold. In this model we use kinematic equations to rollout 
 * the system states.
 * The acceleration and the torques are decision variables in this differential Action Model.
 * To ensure dynamical feasibility RNEA residual constraint is imposed as an equality constraint.

 * The computation of these equations are carrying out inside `calc()` function. In short, this function computes
 * the cost and constraints values (also called constraints violations). It also sets the acceleration equal to 
 * that of the decision variable values rather than using ABA like the traditional DDP algorithm.
 * This procedure is  equivalent to running a forward pass of the action model. 
 *
 * However, during numerical optimization, we also need to run backward passes of the differential action model. These
 * calculations are performed by `calcDiff()`. In short, this function builds a linear-quadratic approximation of the
 * differential action model, i.e.:
 * \f[
 * \begin{aligned}
 * &\delta\dot{\mathbf{v}} =
 * \mathbf{f_{q}}\delta\mathbf{q}+\mathbf{f_{v}}\delta\mathbf{v}+\mathbf{f_{u}}\delta\mathbf{u}, &\textrm{(dynamics)}\\
 * &l(\delta\mathbf{q},\delta\mathbf{v},\delta\mathbf{u}) = \begin{bmatrix}1 \\ \delta\mathbf{q} \\ \delta\mathbf{v} \\
 * \delta\mathbf{u}\end{bmatrix}^T
 * \begin{bmatrix}0 & \mathbf{l_q}^T & \mathbf{l_v}^T & \mathbf{l_u}^T \\ \mathbf{l_q} & \mathbf{l_{qq}} &
 * \mathbf{l_{qv}} & \mathbf{l_{uq}}^T \\
 * \mathbf{l_v} & \mathbf{l_{vq}} & \mathbf{l_{vv}} & \mathbf{l_{uv}}^T \\
 * \mathbf{l_u} & \mathbf{l_{uq}} & \mathbf{l_{uv}} & \mathbf{l_{uu}}\end{bmatrix} \begin{bmatrix}1 \\ \delta\mathbf{q}
 * \\ \delta\mathbf{v} \\
 * \delta\mathbf{u}\end{bmatrix}, &\textrm{(cost)}\\
 * &\mathbf{g_q}\delta\mathbf{q}+\mathbf{g_v}\delta\mathbf{v}+\mathbf{g_u}\delta\mathbf{u}\leq\mathbf{0},
 * &\textrm{(inequality constraints)}\\
 * &\mathbf{h_q}\delta\mathbf{q}+\mathbf{h_v}\delta\mathbf{v}+\mathbf{h_u}\delta\mathbf{u}=\mathbf{0},
 * &\textrm{(equality constraints)} \end{aligned} \f] where
 *  - \f$\mathbf{f_x}=(\mathbf{f_q};\,\, \mathbf{f_v})\in\mathbb{R}^{nv\times ndx}\f$ and
 * \f$\mathbf{f_u}\in\mathbb{R}^{nv\times nv+nu}\f$ are the Jacobians of the dynamics and in this case these are constant matrices,
 *  - \f$\mathbf{l_x}=(\mathbf{l_q};\,\, \mathbf{l_v})\in\mathbb{R}^{ndx}\f$ and \f$\mathbf{l_u}\in\mathbb{R}^{nv+nu}\f$
 * are the Jacobians of the cost function,
 *  - \f$\mathbf{l_{xx}}=(\mathbf{l_{qq}}\,\, \mathbf{l_{qv}};\,\, \mathbf{l_{vq}}\,
 * \mathbf{l_{vv}})\in\mathbb{R}^{ndx\times ndx}\f$, \f$\mathbf{l_{xu}}=(\mathbf{l_q};\,\,
 * \mathbf{l_v})\in\mathbb{R}^{ndx\times nv+nu}\f$ and \f$\mathbf{l_{uu}}\in\mathbb{R}^{{nv+nu} \times {nv+nu}}\f$ are the Hessians
 * of the cost function,
 *  - \f$\mathbf{g_x}=(\mathbf{g_q};\,\, \mathbf{g_v})\in\mathbb{R}^{ng\times ndx}\f$ and
 * \f$\mathbf{g_u}\in\mathbb{R}^{ng\times {nv+nu}}\f$ are the Jacobians of the inequality constraints, and
 *  - \f$\mathbf{h_x}=(\mathbf{h_q};\,\, \mathbf{h_v})\in\mathbb{R}^{nh\times ndx}\f$ and
 * \f$\mathbf{h_u}\in\mathbb{R}^{nh\times {mv+nu}}\f$ are the Jacobians of the equality constraints.
 * Additionally, it is important remark that `calcDiff()` computes the derivates using the latest stored values by
 * `calc()`. Thus, we need to run first `calc()`.
 *
 * \sa `calc()`, `calcDiff()`, `createData()`
 */

class DifferentialActionModelFreeInvDynamicsTpl : public DifferentialActionModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DifferentialActionModelAbstractTpl<Scalar> Base;
  typedef DifferentialActionDataFreeInvDynamicsTpl<Scalar> Data;
  typedef DifferentialActionDataAbstractTpl<Scalar> DifferentialActionDataAbstract;
  typedef CostModelSumTpl<Scalar> CostModelSum;
  typedef ConstraintModelManagerTpl<Scalar> ConstraintModelManager;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ActuationModelAbstractTpl<Scalar> ActuationModelAbstract;
  typedef ConstraintModelResidualTpl<Scalar> ConstraintModelResidual;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize the differential action model
   *
   * It describes the kinematic evolution of the multibody system without any contact,
   * and imposes an inverse-dynamics (equality) constraint. Additionally, it computes
   * the cost and extra constraint values associated to this state and control pair.
   * Note that the name `rnea` in the ConstraintModelManager is reserved to store
   * the inverse-dynamics constraint\n.
  
   * @param[in] state       State description
   * @param[in] actuation   Actuation model
   * @param[in] costs       Cost model
   */
  DifferentialActionModelFreeInvDynamicsTpl(boost::shared_ptr<StateMultibody> state,
                                            boost::shared_ptr<ActuationModelAbstract> actuation,
                                            boost::shared_ptr<CostModelSum> costs);
  
  /**
   * @brief Initialize the differential action model
   *
   * @param[in] state       State description
   * @param[in] actuation   Actuation model
   * @param[in] costs       Cost model
   * @param[in] constraints Constraints model
   */
  DifferentialActionModelFreeInvDynamicsTpl(boost::shared_ptr<StateMultibody> state,
                                            boost::shared_ptr<ActuationModelAbstract> actuation,
                                            boost::shared_ptr<CostModelSum> costs,
                                            boost::shared_ptr<ConstraintModelManager> constraints);
  virtual ~DifferentialActionModelFreeInvDynamicsTpl();

  /**
   * @brief Compute the system acceleration and cost value
   *
   * It extracts the acceleration value from control vector and also computes the cost and constraints. 
   * 
   * @param[in] data  Differential action data
   * @param[in] x     State point
   * @param[in] u     Control input
   */  
  virtual void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);
  
  /**
   * @brief Compute the derivatives of the dynamics and cost functions
   *
   * It computes the partial derivatives of the dynamical system and the cost function. It assumes that `calc()` has
   * been run first. This function builds a quadratic approximation of the time-continuous action model (i.e. dynamical
   * system and cost function).
   *
   * @param[in] data  Differential action data
   * @param[in] x     State point
   * @param[in] u     Control input
   */
  virtual void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x, const Eigen::Ref<const VectorXs>& u);
  
  /**
   * @brief Create the differential action data
   *
   * @return the differential action data
   */
  virtual boost::shared_ptr<DifferentialActionDataAbstract> createData();
  
  /**
   * @brief Checks that a specific data belongs to this model
   */
  virtual bool checkData(const boost::shared_ptr<DifferentialActionDataAbstract>& data);

  virtual void quasiStatic(const boost::shared_ptr<DifferentialActionDataAbstract>& data, Eigen::Ref<VectorXs> u,
                           const Eigen::Ref<const VectorXs>& x, const std::size_t maxiter = 100,
                           const Scalar tol = Scalar(1e-9));

  const boost::shared_ptr<ActuationModelAbstract>& get_actuation() const;
  const boost::shared_ptr<CostModelSum>& get_costs() const;
  const boost::shared_ptr<ConstraintModelManager>& get_constraints() const;
  pinocchio::ModelTpl<Scalar>& get_pinocchio() const;

  /**
   * @brief Print relevant information of the free forward-dynamics model
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
  boost::shared_ptr<ActuationModelAbstract> actuation_;
  boost::shared_ptr<CostModelSum> costs_;
  boost::shared_ptr<ConstraintModelManager> constraints_;
  pinocchio::ModelTpl<Scalar>& pinocchio_;

/**
 * @brief RNEA residual
 *
 * This residual function is defined as r = tau - RNEA(q,q_dot,a), where
 * tau is extracted from the control vector and RNEA evaluates the joint torque using
 * q, q_dot, acc values. Furthermore, the Jacobians of the residual function are
 * computed analytically. This is used by constrainModelmanger inside parent 
 * DifferentialActionModelFreeInvDynamics class.
 *
 * As described in `ResidualModelAbstractTpl()`, the residual value and its Jacobians are calculated by `calc` and
 * `calcDiff`, respectively.
 *
 * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
 public:
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
   * @param[in] nu     Dimension of the control vector
   */
    ResidualModelRnea(boost::shared_ptr<StateMultibody> state, const std::size_t nu)
        : Base(state, state->get_nv(), state->get_nv() + nu, true, true, true), na_(nu) {}
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
      const boost::shared_ptr<typename Data::ResidualDataRnea>& d =
          boost::static_pointer_cast<typename Data::ResidualDataRnea>(data);
      data->r = d->pinocchio->tau - d->actuation->tau;
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
      const boost::shared_ptr<typename Data::ResidualDataRnea>& d =
          boost::static_pointer_cast<typename Data::ResidualDataRnea>(data);
      const std::size_t nv = state_->get_nv();
      data->Rx.leftCols(nv) = d->pinocchio->dtau_dq;
      data->Rx.rightCols(nv) = d->pinocchio->dtau_dv;
      data->Rx -= d->actuation->dtau_dx;
      data->Ru.leftCols(nv) = d->pinocchio->M;
      data->Ru.rightCols(na_) = -d->actuation->dtau_du;
    }

  /**
   * @brief Create the residual data
   *
   * @return the residual data
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
         << ", na=" << na_ << "}";
    }

   protected:
    std::size_t na_;
    using Base::nu_;
    using Base::state_;
  };
};

template <typename _Scalar>
struct DifferentialActionDataFreeInvDynamicsTpl : public DifferentialActionDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef DifferentialActionDataAbstractTpl<Scalar> Base;
  typedef typename MathBase::VectorXs VectorXs;

  template <template <typename Scalar> class Model>
  explicit DifferentialActionDataFreeInvDynamicsTpl(Model<Scalar>* const model)
      : Base(model),
        pinocchio(pinocchio::DataTpl<Scalar>(model->get_pinocchio())),
        multibody(&pinocchio, model->get_actuation()->createData()),
        costs(model->get_costs()->createData(&multibody)),
        constraints(model->get_constraints()->createData(&multibody)),
        tmp_xstatic(model->get_state()->get_nx()),
        tmp_ustatic(model->get_nu()) {
    costs->shareMemory(this);
    constraints->shareMemory(this);
    Fu.leftCols(model->get_state()->get_nv()).diagonal().array() = 1;
    tmp_xstatic.setZero();
    tmp_ustatic.setZero();
  }

  pinocchio::DataTpl<Scalar> pinocchio;  //!< Pinocchio data
  DataCollectorActMultibodyTpl<Scalar> multibody; //!< multibody data
  boost::shared_ptr<CostDataSumTpl<Scalar> > costs;  //!< Costs data
  boost::shared_ptr<ConstraintDataManagerTpl<Scalar> > constraints;  //!< Constraints data
  VectorXs tmp_xstatic;   //!< quasistatic state point (velocity has to be zero)
  VectorXs tmp_ustatic;   //!< quasistatic control vector
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

    template <template <typename Scalar> class Model>
    ResidualDataRnea(Model<Scalar>* const model, DataCollectorAbstract* const data) : Base(model, data) {
      // Check that proper shared data has been passed
      DataCollectorActMultibodyTpl<Scalar>* d = dynamic_cast<DataCollectorActMultibodyTpl<Scalar>*>(shared);
      if (d == NULL) {
        throw_pretty("Invalid argument: the shared data should be derived from DataCollectorActMultibody");
      }
      const std::size_t na = Ru.cols() - model->get_state()->get_nv();
      Ru.rightCols(na) = -d->actuation->dtau_du;

      // Avoids data casting at runtime
      pinocchio = d->pinocchio;
      actuation = d->actuation.get();
    }

    pinocchio::DataTpl<Scalar>* pinocchio;        //!< Pinocchio data
    ActuationDataAbstractTpl<Scalar>* actuation;  //!< Actuation data
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

#endif  // CROCODDYL_MULTIBODY_ACTIONS_FREE_INVDYN_HPP_
