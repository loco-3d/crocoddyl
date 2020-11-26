///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_COSTS_CONTROL_GRAVITY_CONTACT_HPP_
#define CROCODDYL_CORE_COSTS_CONTROL_GRAVITY_CONTACT_HPP_

#include "pinocchio/fwd.hpp"
#include "pinocchio/multibody/model.hpp"
#include "pinocchio/algorithm/model.hpp"
#include "pinocchio/algorithm/rnea.hpp"
#include "pinocchio/algorithm/rnea-derivatives.hpp"

#include "crocoddyl/multibody/contact-base.hpp"
#include "crocoddyl/multibody/contacts/multiple-contacts.hpp"
#include "crocoddyl/multibody/data/contacts.hpp"

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/cost-base.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/frames.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/utils/deprecate.hpp"

namespace crocoddyl {

/**
 * @brief Control cost
 *
 * This cost function defines a residual vector as \f$\mathbf{r}=\mathbf{u}-\mathbf{u}^*\f$, where
 * \f$\mathbf{u},\mathbf{u}^*\in~\mathbb{R}^{nu}\f$ are the current and reference control inputs, respetively. Note
 * that the dimension of the residual vector is obtained from `nu`.
 *
 * Both cost and residual derivatives are computed analytically.
 * For the computation of the cost Hessian, we use the Gauss-Newton approximation, e.g.
 * \f$\mathbf{l_{xx}} = \mathbf{l_{x}}^T \mathbf{l_{x}} \f$.
 *
 * As described in CostModelAbstractTpl(), the cost value and its derivatives are calculated by `calc` and `calcDiff`,
 * respectively.
 *
 * \sa `CostModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class CostModelControlGravContactTpl : public CostModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelAbstractTpl<Scalar> Base;
  typedef CostDataControlGravContactTpl<Scalar> Data;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef ActivationModelQuadTpl<Scalar> ActivationModelQuad;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the control cost model
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   * @param[in] uref        Reference control input
   */
  CostModelControlGravContactTpl(boost::shared_ptr<StateMultibody> state,
                      boost::shared_ptr<ActivationModelAbstract> activation);

  /**
   * @brief Initialize the control cost model
   *
   * The default reference configuration is obtained from `StateAbstractTpl::get_nq()`.
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   * @param[in] nu          Dimension of the control vector
   */
  CostModelControlGravContactTpl(boost::shared_ptr<StateMultibody> state,
                      boost::shared_ptr<ActivationModelAbstract> activation, const std::size_t& nu);

  /**
   * @brief Initialize the control cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$). The
   * default reference control is obtained from `MathBaseTpl<>::VectorXs::Zero(nu)` with `nu` defined by
   * `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model control vector
   */
  explicit CostModelControlGravContactTpl(boost::shared_ptr<StateMultibody> state);

  /**
   * @brief Initialize the control cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$). The
   * default reference control is obtained from `MathBaseTpl<>::VectorXs::Zero(nu)`.
   *
   * @param[in] state       State of the multibody system
   * @param[in] nu          Dimension of the control vector
   */
  CostModelControlGravContactTpl(boost::shared_ptr<StateMultibody> state, const std::size_t& nu);

  virtual ~CostModelControlGravContactTpl();

  /**
   * @brief Compute the control cost
   *
   * @param[in] data  Control cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the derivatives of the control cost
   *
   * @param[in] data  Control cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);
  
  virtual boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

 protected:

  using Base::activation_;
  using Base::nu_;
  using Base::state_;
  using Base::unone_;

private:
  boost::shared_ptr<typename StateMultibody::PinocchioModel> pin_model_;
};

template <typename _Scalar>
struct CostDataControlGravContactTpl : public CostDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  //typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  CostDataControlGravContactTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : Base(model, data),rnea_partial_dx(model->get_state()->get_ndx(),model->get_nu()),
                          rnea_partial_da(model->get_state()->get_nv(),model->get_nu()) {
    rnea_partial_dx.setZero();
    rnea_partial_da.setZero();
    // Check that proper shared data has been passed
    DataCollectorMultibodyInContactTpl<Scalar>* d = dynamic_cast<DataCollectorMultibodyInContactTpl<Scalar>*>(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorContact");
    }
    // Avoids data casting at runtime
    pinocchio = d->pinocchio;
    fext = d->contacts->fext;
    }
  
  pinocchio::DataTpl<Scalar>* pinocchio;
  pinocchio::container::aligned_vector<pinocchio::ForceTpl<Scalar> > fext;
  MatrixXs rnea_partial_dx;
  MatrixXs rnea_partial_da;
  using Base::activation;
  using Base::cost;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::r;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/costs/control-gravity-contact.hxx"

#endif  // CROCODDYL_MULTIBODY_COSTS_CONTROL_GRAVITY_CONTACT_HPP_
