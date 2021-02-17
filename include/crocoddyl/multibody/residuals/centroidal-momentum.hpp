///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_RESIDUALS_MOMENTUM_HPP_
#define CROCODDYL_MULTIBODY_RESIDUALS_MOMENTUM_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"

namespace crocoddyl {

/**
 * @brief Centroidal momentum residual
 *
 * This residual function defines the centroidal momentum tracking as \f$\mathbf{r}=\mathbf{h}-\mathbf{h}^*\f$, where
 * \f$\mathbf{h},\mathbf{h}^*\in~\mathcal{X}\f$ are the current and reference centroidal momenta, respectively. Note
 * that the dimension of the residual vector is 6.
 * Furthermore, the Jacobians of the residual function are computed analytically.
 *
 * As described in `ResidualModelAbstractTpl()`, the residual value and its Jacobians are calculated by `calc` and
 * `calcDiff`, respectively.
 *
 * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ResidualModelCentroidalMomentumTpl : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataCentroidalMomentumTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::Vector6s Vector6s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Matrix6xs Matrix6xs;

  /**
   * @brief Initialize the centroidal momentum residual model
   *
   * @param[in] state  State of the multibody system
   * @param[in] href   Reference centroidal momentum
   * @param[in] nu     Dimension of the control vector
   */
  ResidualModelCentroidalMomentumTpl(boost::shared_ptr<StateMultibody> state, const Vector6s& href,
                                     const std::size_t nu);

  /**
   * @brief Initialize the centroidal momentum residual model
   *
   * The default `nu` is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state  State of the multibody system
   * @param[in] href   Reference centroidal momentum
   */
  ResidualModelCentroidalMomentumTpl(boost::shared_ptr<StateMultibody> state, const Vector6s& href);
  virtual ~ResidualModelCentroidalMomentumTpl();

  /**
   * @brief Compute the centroidal momentum residual
   *
   * @param[in] data  Centroidal momentum residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the derivatives of the centroidal momentum residual
   *
   * @param[in] data  Centroidal momentum residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Create the centroidal momentum residual data
   */
  virtual boost::shared_ptr<ResidualDataAbstract> createData(DataCollectorAbstract* const data);

  /**
   * @brief Return the reference centroidal momentum
   */
  const Vector6s& get_reference() const;

  /**
   * @brief Modify the reference centroidal momentum
   */
  void set_reference(const Vector6s& href);

 protected:
  using Base::nu_;
  using Base::state_;
  using Base::unone_;

 private:
  Vector6s href_;                                                         //!< Reference centroidal momentum
  boost::shared_ptr<typename StateMultibody::PinocchioModel> pin_model_;  //!< Pinocchio model
};

template <typename _Scalar>
struct ResidualDataCentroidalMomentumTpl : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::Matrix6xs Matrix6xs;

  template <template <typename Scalar> class Model>
  ResidualDataCentroidalMomentumTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : Base(model, data), dhd_dq(6, model->get_state()->get_nv()), dhd_dv(6, model->get_state()->get_nv()) {
    dhd_dq.setZero();
    dhd_dv.setZero();

    // Check that proper shared data has been passed
    DataCollectorMultibodyTpl<Scalar>* d = dynamic_cast<DataCollectorMultibodyTpl<Scalar>*>(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorMultibody");
    }

    // Avoids data casting at runtime
    pinocchio = d->pinocchio;
  }

  pinocchio::DataTpl<Scalar>* pinocchio;  //!< Pinocchio data
  Matrix6xs dhd_dq;                       //!< Jacobian of the centroidal momentum
  Matrix6xs dhd_dv;                       //!< Jacobian of the centroidal momentum
  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/residuals/centroidal-momentum.hxx"

#endif  // CROCODDYL_MULTIBODY_RESIDUALS_MOMENTUM_HPP_
