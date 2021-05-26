///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_RESIDUALS_IMPULSE_COM_HPP_
#define CROCODDYL_MULTIBODY_RESIDUALS_IMPULSE_COM_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/impulse-base.hpp"
#include "crocoddyl/multibody/data/impulses.hpp"
#include "crocoddyl/multibody/frames.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief Impulse CoM residual
 *
 * This residual function is defined as \f$\mathbf{r}=\mathbf{J}_{com}*(\mathbf{v}_{next}-\mathbf{v})\f$,
 * \f$\mathbf{J}_{com}\in\mathbb{R}^{3\times nv}\f$ is the CoM Jacobian, and \f$\mathbf{v}_{next},\mathbf{v}\in
 * T_{\mathbf{q}}\mathcal{Q}\f$ are the generalized velocities after and before the impulse, respectively. Note that
 * the dimension of the residual vector is 3. Furthermore, the Jacobians of the residual function are
 * computed analytically.
 *
 * As described in `ResidualModelAbstractTpl()`, the residual value and its Jacobians are calculated by `calc` and
 * `calcDiff`, respectively.
 *
 * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ResidualModelImpulseCoMTpl : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataImpulseCoMTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize the impulse CoM residual model
   *
   * @param[in] state       State of the multibody system
   */
  ResidualModelImpulseCoMTpl(boost::shared_ptr<StateMultibody> state);
  virtual ~ResidualModelImpulseCoMTpl();

  /**
   * @brief Compute the impulse CoM residual
   *
   * @param[in] data  Impulse CoM residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the Jacobians of the impulse CoM residual
   *
   * @param[in] data  Impulse CoM residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Create the impulse CoM residual data
   */
  virtual boost::shared_ptr<ResidualDataAbstract> createData(DataCollectorAbstract* const data);

  /**
   * @brief Print relevant information of the impulse-com residual
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 protected:
  using Base::nu_;
  using Base::state_;
  using Base::u_dependent_;
  using Base::unone_;

 private:
  boost::shared_ptr<typename StateMultibody::PinocchioModel> pin_model_;
};

template <typename _Scalar>
struct ResidualDataImpulseCoMTpl : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Matrix3xs Matrix3xs;

  template <template <typename Scalar> class Model>
  ResidualDataImpulseCoMTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : Base(model, data),
        dvc_dq(3, model->get_state()->get_nv()),
        ddv_dv(model->get_state()->get_nv(), model->get_state()->get_nv()) {
    dvc_dq.setZero();
    ddv_dv.setZero();
    const boost::shared_ptr<StateMultibody>& state = boost::static_pointer_cast<StateMultibody>(model->get_state());
    pinocchio_internal = pinocchio::DataTpl<Scalar>(*state->get_pinocchio().get());
    // Check that proper shared data has been passed
    DataCollectorMultibodyInImpulseTpl<Scalar>* d = dynamic_cast<DataCollectorMultibodyInImpulseTpl<Scalar>*>(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorMultibodyInImpulse");
    }
    pinocchio = d->pinocchio;
    impulses = d->impulses;
  }

  pinocchio::DataTpl<Scalar>* pinocchio;                                   //!< Pinocchio data
  boost::shared_ptr<crocoddyl::ImpulseDataMultipleTpl<Scalar> > impulses;  //!< Impulses data
  Matrix3xs dvc_dq;                                                        //!< Jacobian of the CoM velocity
  MatrixXs ddv_dv;                                                         //!< Jacobian of the CoM velocity
  pinocchio::DataTpl<Scalar> pinocchio_internal;                           //!< Pinocchio data for internal computation
  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/residuals/impulse-com.hxx"

#endif  // CROCODDYL_MULTIBODY_RESIDUALS_IMPULSE_COM_HPP_
