///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_CONTACT_IMPULSE_HPP_
#define CROCODDYL_MULTIBODY_COSTS_CONTACT_IMPULSE_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/cost-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/residuals/contact-force.hpp"
#include "crocoddyl/multibody/frames.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/utils/deprecate.hpp"

namespace crocoddyl {

/**
 * @brief Define a contact impulse cost function
 *
 * This cost function defines a residual vector \f$\mathbf{r}=\boldsymbol{\lambda}-\boldsymbol{\lambda}^*\f$,
 * where \f$\boldsymbol{\lambda}, \boldsymbol{\lambda}^*\f$ are the current and reference spatial impulses,
 * respectively. The current spatial impulses \f$\boldsymbol{\lambda}\in\mathbb{R}^{ni}\f$is computed by
 * `ActionModelImpulseFwdDynamicsTpl`, with `ni` as the dimension of the impulse.
 *
 * Both cost and residual derivatives are computed analytically, where th force vector \f$\boldsymbol{\lambda}\f$ and
 * its derivatives \f$\left(\frac{\partial\boldsymbol{\lambda}}{\partial\mathbf{x}},
 * \frac{\partial\boldsymbol{\lambda}}{\partial\mathbf{u}}\right)\f$ are computed by
 * `ActionModelImpulseFwdDynamicsTpl`. These values are stored in a shared data (i.e.
 * DataCollectorImpulseTpl). Note that this cost function cannot be used with other action models.
 * For the computation of the cost Hessian, we use the Gauss-Newton approximation, e.g.
 * \f$\mathbf{l_{xu}} = \mathbf{l_{x}}^T \mathbf{l_{u}} \f$.
 *
 * \sa `ActionModelImpulseFwdDynamicsTpl`, `DataCollectorImpulseTpl`, `ActivationModelAbstractTpl`
 */
template <typename _Scalar>
class CostModelContactImpulseTpl : public CostModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelAbstractTpl<Scalar> Base;
  typedef CostDataContactImpulseTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef ResidualModelContactForceTpl<Scalar> ResidualModelContactForce;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef FrameForceTpl<Scalar> FrameForce;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the contact impulse cost model
   *
   * Note that the `nr`, defined in the activation model, has to be lower / equals than 6.
   *
   * @param[in] state       Multibody state
   * @param[in] activation  Activation model
   * @param[in] fref        Reference spatial contact impulse \f$\boldsymbol{\lambda}^*\f$
   */
  CostModelContactImpulseTpl(boost::shared_ptr<StateMultibody> state,
                             boost::shared_ptr<ActivationModelAbstract> activation, const FrameForce& fref);

  /**
   * @brief Initialize the contact impulse cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$).
   * Note that the `nr`, defined in the activation model, has to be lower / equals than 6.
   *
   * @param[in] state  Multibody state
   * @param[in] fref   Reference spatial contact impulse \f$\boldsymbol{\lambda}^*\f$
   * @param[in] nr     Dimension of residual vector
   */
  DEPRECATED("No needed to pass nr", CostModelContactImpulseTpl(boost::shared_ptr<StateMultibody> state,
                                                                const FrameForce& fref, const std::size_t nr);)

  /**
   * @brief Initialize the contact impulse cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$), and `nr`
   * is 6.
   *
   * @param[in] state  Multibody state
   * @param[in] fref   Reference spatial contact impulse \f$\boldsymbol{\lambda}^*\f$
   */
  CostModelContactImpulseTpl(boost::shared_ptr<StateMultibody> state, const FrameForce& fref);
  virtual ~CostModelContactImpulseTpl();

  /**
   * @brief Compute the contact impulse cost
   *
   * The impulse vector is computed by ActionModelImpulseFwdDynamicsTpl and stored in DataCollectorImpulseTpl.
   *
   * @param[in] data  Contact impulse data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the derivatives of the contact impulse cost
   *
   * The impulse derivatives are computed by ActionModelImpulseFwdDynamicsTpl and stored in DataCollectorImpulseTpl.
   *
   * @param[in] data  Contact impulse data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Create the contact impulse cost data
   *
   * @param[in] data  shared data (it should be of type DataCollectorImpulseTpl)
   * @return the cost data.
   */
  virtual boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

 protected:
  /**
   * @brief Return the reference spatial impulse \f$\boldsymbol{\lambda}^*\f$
   */
  virtual void set_referenceImpl(const std::type_info& ti, const void* pv);

  /**
   * @brief Modify the reference spatial impulse \f$\boldsymbol{\lambda}^*\f$
   */
  virtual void get_referenceImpl(const std::type_info& ti, void* pv);

  using Base::activation_;
  using Base::nu_;
  using Base::residual_;
  using Base::state_;
  using Base::unone_;

 protected:
  FrameForce fref_;  //!< Reference spatial impulse \f$\boldsymbol{\lambda}^*\f$
};

template <typename _Scalar>
struct CostDataContactImpulseTpl : public CostDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef ImpulseModelMultipleTpl<Scalar> ImpulseModelMultiple;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  CostDataContactImpulseTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : Base(model, data), Arr_Rx(model->get_residual()->get_nr(), model->get_state()->get_ndx()) {
    Arr_Rx.setZero();
  }

  MatrixXs Arr_Rx;
  using Base::activation;
  using Base::cost;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::residual;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/costs/contact-impulse.hxx"

#endif  // CROCODDYL_MULTIBODY_COSTS_CONTACT_IMPULSE_HPP_
