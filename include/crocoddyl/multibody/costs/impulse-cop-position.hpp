///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, University of Duisburg-Essen, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_IMPULSE_COP_POSITION_HPP_
#define CROCODDYL_MULTIBODY_COSTS_IMPULSE_COP_POSITION_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/cost-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/impulse-base.hpp"
#include "crocoddyl/multibody/impulses/impulse-3d.hpp"
#include "crocoddyl/multibody/impulses/impulse-6d.hpp"
#include "crocoddyl/multibody/data/impulses.hpp"
#include "crocoddyl/multibody/frames.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/core/activations/quadratic-barrier.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief Define a center of pressure cost function
 *
 * It builds a cost function that bounds the center of pressure (CoP) for one contact surface to
 * lie inside a certain geometric area defined around the reference contact frame. The cost residual
 * vector is described as \f$\mathbf{r}=\mathbf{A}\boldsymbol{\lambda}\geq\mathbf{0}\f$, where
 * \f[
 *   \mathbf{A}=
 *     \begin{bmatrix} 0 & 0 & X/2 & 0 & -1 & 0 \\ 0 & 0 & X/2 & 0 & 1 & 0 \\ 0 & 0 & Y/2 & 1 & 0 & 0 \\
 *      0 & 0 & Y/2 & -1 & 0 & 0
 *     \end{bmatrix}
 * \f]
 * is the inequality matrix and \f$\boldsymbol{\lambda}\f$ is the reference spatial contact force in the frame
 * coordinate. The CoP lies inside the convex hull of the foot, see eq.(18-19) of
 * https://hal.archives-ouvertes.fr/hal-02108449/document is we have:
 * \f[
 *  \begin{align}\begin{split}\tau^x &\leq
 * Yf^z \\-\tau^x &\leq Yf^z \\\tau^y &\leq Yf^z \\-\tau^y &\leq Yf^z
 *  \end{split}\end{align}
 * \f]
 * with \f$\boldsymbol{\lambda}= \begin{bmatrix}f^x & f^y & f^z & \tau^x & \tau^y & \tau^z \end{bmatrix}^T\f$.
 *
 * The cost is computed, from the residual vector \f$\mathbf{r}\f$, through an user defined activation model.
 * Additionally, the contact frame id, the desired support region for the CoP and the inequality matrix
 * are handled within `FrameCoPSupportTpl`. The force vector \f$\boldsymbol{\lambda}\f$ and its derivatives are
 * computed by `ActionModelImpulseFwdDynamicsTpl`. These values are stored in a shared data (i.e.
 * `DataCollectorImpulseTpl`). Note that this cost function cannot be used with other action models.
 *
 * \sa `ActionModelImpulseFwdDynamicsTpl`, `DataCollectorImpulseTpl`, `ActivationModelAbstractTpl`, `calc()`,
 * `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class CostModelImpulseCoPPositionTpl : public CostModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelAbstractTpl<Scalar> Base;
  typedef CostDataImpulseCoPPositionTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef ActivationModelQuadraticBarrierTpl<Scalar> ActivationModelQuadraticBarrier;
  typedef ActivationBoundsTpl<Scalar> ActivationBounds;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef FrameCoPSupportTpl<Scalar> FrameCoPSupport;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef Eigen::Matrix<Scalar, 4, 6> Matrix46;
  typedef typename MathBase::Vector2s Vector2s;

  /**
   * @brief Initialize the impulse CoP cost model
   *
   * @param[in] state        State of the multibody system
   * @param[in] activation   Activation model
   * @param[in] cop_support  Id of contact frame and support region of the CoP
   */
  CostModelImpulseCoPPositionTpl(boost::shared_ptr<StateMultibody> state,
                                 boost::shared_ptr<ActivationModelAbstract> activation,
                                 const FrameCoPSupport& cop_support);

  /**
   * @brief Initialize the impulse CoP cost model
   *
   * We use as default activation model a quadratic barrier `ActivationModelQuadraticBarrierTpl`, with 0 and inf as
   * lower and upper bounds, respectively.
   *
   * @param[in] state        State of the multibody system
   * @param[in] cop_support  Id of contact frame and support region of the cop
   */
  CostModelImpulseCoPPositionTpl(boost::shared_ptr<StateMultibody> state, const FrameCoPSupport& cop_support);
  virtual ~CostModelImpulseCoPPositionTpl();

  /**
   * @brief Compute the impulse CoP cost
   *
   * The CoP residual is computed based on the \f$\mathbf{A}\f$ matrix, the force vector is computed by
   * `ActionModelImpulseFwdDynamicsTpl` which is stored in `DataCollectorImpulseTpl.
   *
   * @param[in] data  Impulse CoP data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the derivatives of the impulse CoP cost
   *
   * The CoP derivatives are based on the force derivatives computed by
   * `ActionModelImpulseFwdDynamicsTpl` which are stored in `DataCollectorImpulseTpl`.
   *
   * @param[in] data  Impulse CoP data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Create the impulse CoP cost data
   *
   * Each cost model has its own data that needs to be allocated.
   * This function returns the allocated data for a predefined cost.
   *
   * @param[in] data  Shared data (it should be of type `DataCollectorImpulseTpl`)
   * @return the cost data.
   */
  virtual boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

 protected:
  /**
   * @brief Return the frame CoP support
   */
  virtual void set_referenceImpl(const std::type_info& ti, const void* pv);

  /**
   * @brief Modify the frame CoP support
   */
  virtual void get_referenceImpl(const std::type_info& ti, void* pv) const;

  using Base::activation_;
  using Base::nu_;
  using Base::state_;

 private:
  FrameCoPSupport cop_support_;  //!< Frame name of the impulse foot and support region of the CoP
};

template <typename _Scalar>
struct CostDataImpulseCoPPositionTpl : public CostDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef ImpulseModelMultipleTpl<Scalar> ImpulseModelMultiple;
  typedef FrameCoPSupportTpl<Scalar> FrameCoPSupport;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Model>
  CostDataImpulseCoPPositionTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : Base(model, data), Arr_Ru(model->get_activation()->get_nr(), model->get_state()->get_nv()) {
    Arr_Ru.setZero();

    // Check that proper shared data has been passed
    DataCollectorImpulseTpl<Scalar>* d = dynamic_cast<DataCollectorImpulseTpl<Scalar>*>(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorImpulse");
    }

    // Get the active 6d impulse (avoids data casting at runtime)
    FrameCoPSupport cop_support = model->template get_reference<FrameCoPSupport>();
    const boost::shared_ptr<StateMultibody>& state = boost::static_pointer_cast<StateMultibody>(model->get_state());
    std::string frame_name = state->get_pinocchio()->frames[cop_support.get_id()].name;
    bool found_impulse = false;
    for (typename ImpulseModelMultiple::ImpulseDataContainer::iterator it = d->impulses->impulses.begin();
         it != d->impulses->impulses.end(); ++it) {
      if (it->second->frame == cop_support.get_id()) {
        ImpulseData3DTpl<Scalar>* d3d = dynamic_cast<ImpulseData3DTpl<Scalar>*>(it->second.get());
        if (d3d != NULL) {
          throw_pretty("Domain error: a 6d impulse model is required in " + frame_name +
                       "in order to compute the CoP");
          break;
        }
        ImpulseData6DTpl<Scalar>* d6d = dynamic_cast<ImpulseData6DTpl<Scalar>*>(it->second.get());
        if (d6d != NULL) {
          found_impulse = true;
          impulse = it->second;
          break;
        }
      }
    }
    if (!found_impulse) {
      throw_pretty("Domain error: there isn't defined impulse data for " + frame_name);
    }
  }

  pinocchio::DataTpl<Scalar>* pinocchio;
  MatrixXs Arr_Ru;
  boost::shared_ptr<ImpulseDataAbstractTpl<Scalar> > impulse;  //!< impulse force
  using Base::activation;
  using Base::cost;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/costs/impulse-cop-position.hxx"

#endif  // CROCODDYL_MULTIBODY_COSTS_IMPULSE_COP_POSITION_HPP_
