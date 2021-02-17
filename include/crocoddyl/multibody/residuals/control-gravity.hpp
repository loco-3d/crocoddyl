///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_RESIDUALS_CONTROL_GRAVITY_HPP_
#define CROCODDYL_CORE_RESIDUALS_CONTROL_GRAVITY_HPP_

#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief Control gravity residual
 *
 * This residual function is defined as \f$\mathbf{r}=\mathbf{u}-\mathbf{g}(\mathbf{q})\f$, where
 * \f$\mathbf{u}\in~\mathbb{R}^{nu}\f$ is the current control input, \f$\mathbf{g}(\mathbf{q})\f$ is the
 * gravity torque corresponding to the current configuration, \f$\mathbf{q}\in~\mathbb{R}^{nq}\f$ the current position
 * joints input. Note that the dimension of the residual vector is obtained from `StateAbstractTpl::get_nv()`.
 *
 * Both residual and residual Jacobians are computed analytically.
 *
 * As described in ResidualModelAbstractTpl(), the residual value and its derivatives
 * are calculated by `calc` and `calcDiff`, respectively.
 *
 * \sa `ResidualModelAbstractTpl`, calc(), calcDiff(), createData()
 */
template <typename _Scalar>
class ResidualModelControlGravTpl : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataControlGravTpl<Scalar> Data;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the control gravity residual model
   *
   * @param[in] state       State of the multibody system
   * @param[in] nu          Dimension of control vector
   */
  ResidualModelControlGravTpl(boost::shared_ptr<StateMultibody> state, const std::size_t nu);

  /**
   * @brief Initialize the control gravity residual model
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state       State of the multibody system
   */
  ResidualModelControlGravTpl(boost::shared_ptr<StateMultibody> state);
  virtual ~ResidualModelControlGravTpl();

  /**
   * @brief Compute the control gravity residual
   *
   * @param[in] data  Control residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<ResidualDataAbstract> &data, const Eigen::Ref<const VectorXs> &x,
                    const Eigen::Ref<const VectorXs> &u);

  /**
   * @brief Compute the Jacobians of the control gravity residual
   *
   * @param[in] data  Control residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract> &data, const Eigen::Ref<const VectorXs> &x,
                        const Eigen::Ref<const VectorXs> &u);

  virtual boost::shared_ptr<ResidualDataAbstract> createData(DataCollectorAbstract *const data);

 protected:
  using Base::nu_;
  using Base::state_;
  using Base::unone_;

 private:
  typename StateMultibody::PinocchioModel pin_model_;  //!< Pinocchio model used for internal computations
};

template <typename _Scalar>
struct ResidualDataControlGravTpl : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef pinocchio::DataTpl<Scalar> PinocchioData;

  template <template <typename Scalar> class Model>
  ResidualDataControlGravTpl(Model<Scalar> *const model, DataCollectorAbstract *const data) : Base(model, data) {
    // Check that proper shared data has been passed
    DataCollectorActMultibodyTpl<Scalar> *d = dynamic_cast<DataCollectorActMultibodyTpl<Scalar> *>(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorActMultibodyTpl");
    }
    if (static_cast<std::size_t>(d->actuation->dtau_du.cols()) != model->get_state()->get_nv()) {
      throw_pretty(
          "Invalid argument: the actuation model should consider all the control dimensions (i.e., nu == state.nv)");
    }
    // Avoids data casting at runtime
    StateMultibody *sm = static_cast<StateMultibody *>(model->get_state().get());
    pinocchio = PinocchioData(*(sm->get_pinocchio().get()));
    actuation = d->actuation;
  }

  PinocchioData pinocchio;
  boost::shared_ptr<ActuationDataAbstractTpl<Scalar> > actuation;
  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/residuals/control-gravity.hxx"

#endif  // CROCODDYL_MULTIBODY_RESIDUALS_CONTROL_GRAVITY_HPP_
