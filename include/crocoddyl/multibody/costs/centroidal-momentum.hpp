///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_MOMENTUM_HPP_
#define CROCODDYL_MULTIBODY_COSTS_MOMENTUM_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/cost-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/residuals/centroidal-momentum.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief Centroidal momentum cost
 *
 * This cost function defines a residual vector as \f$\mathbf{r}=\mathbf{h}-\mathbf{h}^*\f$, where
 * \f$\mathbf{h},\mathbf{h}^*\in~\mathcal{X}\f$ are the current and reference centroidal momenta, respectively. Note
 * that the dimension of the residual vector is 6.
 *
 * Both cost and residual derivatives are computed analytically. For the computation of the cost Hessian, we use the
 * Gauss-Newton approximation, e.g. \f$\mathbf{l_{xu}} = \mathbf{l_{x}}^T \mathbf{l_{u}} \f$.
 *
 * As described in `CostModelAbstractTpl()`, the cost value and its derivatives are calculated by `calc` and
 * `calcDiff`, respectively.
 *
 * \sa `CostModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class CostModelCentroidalMomentumTpl : public CostModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelAbstractTpl<Scalar> Base;
  typedef CostDataCentroidalMomentumTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef ResidualModelCentroidalMomentumTpl<Scalar> ResidualModelCentroidalMomentum;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::Vector6s Vector6s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Matrix6xs Matrix6xs;

  /**
   * @brief Initialize the centroidal momentum cost model
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   * @param[in] mref        Reference centroidal momentum
   * @param[in] nu          Dimension of the control vector
   */
  CostModelCentroidalMomentumTpl(boost::shared_ptr<StateMultibody> state,
                                 boost::shared_ptr<ActivationModelAbstract> activation, const Vector6s& mref,
                                 const std::size_t nu);

  /**
   * @brief Initialize the centroidal momentum cost model
   *
   * The default `nu` is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   * @param[in] mref        Reference centroidal momentum
   */
  CostModelCentroidalMomentumTpl(boost::shared_ptr<StateMultibody> state,
                                 boost::shared_ptr<ActivationModelAbstract> activation, const Vector6s& mref);

  /**
   * @brief Initialize the centroidal momentum cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$).
   *
   * @param[in] state  State of the multibody system
   * @param[in] mref   Reference centroidal momentum
   * @param[in] nu     Dimension of the control vector
   */
  CostModelCentroidalMomentumTpl(boost::shared_ptr<StateMultibody> state, const Vector6s& mref, const std::size_t nu);

  /**
   * @brief Initialize the centroidal momentum cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$).
   * Furthermore, the default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state  State of the multibody system
   * @param[in] mref   Reference centroidal momentum
   */
  CostModelCentroidalMomentumTpl(boost::shared_ptr<StateMultibody> state, const Vector6s& mref);
  virtual ~CostModelCentroidalMomentumTpl();

  /**
   * @brief Compute the centroidal momentum cost
   *
   * @param[in] data  Centroidal momentum cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the derivatives of the centroidal momentum cost
   *
   * @param[in] data  Centroidal momentum cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Create the centroidal momentum cost data
   */
  virtual boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

 protected:
  /**
   * @brief Modify the centroidal momentum reference
   */
  virtual void set_referenceImpl(const std::type_info& ti, const void* pv);

  /**
   * @brief Return the centroidal momentum reference
   */
  virtual void get_referenceImpl(const std::type_info& ti, void* pv);

  using Base::activation_;
  using Base::nu_;
  using Base::residual_;
  using Base::state_;
  using Base::unone_;

 private:
  Vector6s href_;  //!< Reference centroidal momentum
};

template <typename _Scalar>
struct CostDataCentroidalMomentumTpl : public CostDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::Matrix6xs Matrix6xs;

  template <template <typename Scalar> class Model>
  CostDataCentroidalMomentumTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : Base(model, data), Arr_Rx(6, model->get_state()->get_ndx()) {
    Arr_Rx.setZero();
  }

  Matrix6xs Arr_Rx;
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
#include "crocoddyl/multibody/costs/centroidal-momentum.hxx"

#endif  // CROCODDYL_MULTIBODY_COSTS_MOMENTUM_HPP_
