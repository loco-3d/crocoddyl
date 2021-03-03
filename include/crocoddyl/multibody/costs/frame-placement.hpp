///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_FRAME_PLACEMENT_HPP_
#define CROCODDYL_MULTIBODY_COSTS_FRAME_PLACEMENT_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/cost-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/residuals/frame-placement.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/frames.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief Frame placement cost
 *
 * This cost function defines a residual vector as \f$\mathbf{r}=\mathbf{p}\ominus\mathbf{p}^*\f$, where
 * \f$\mathbf{p},\mathbf{p}^*\in~\mathbb{SE(3)}\f$ are the current and reference frame placements, respectively. Note
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
class CostModelFramePlacementTpl : public CostModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelAbstractTpl<Scalar> Base;
  typedef CostDataFramePlacementTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef ResidualModelFramePlacementTpl<Scalar> ResidualModelFramePlacement;
  typedef FramePlacementTpl<Scalar> FramePlacement;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Matrix6xs Matrix6xs;

  /**
   * @brief Initialize the frame placement cost model
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   * @param[in] Fref        Reference frame placement
   * @param[in] nu          Dimension of the control vector
   */
  CostModelFramePlacementTpl(boost::shared_ptr<StateMultibody> state,
                             boost::shared_ptr<ActivationModelAbstract> activation, const FramePlacement& Fref,
                             const std::size_t nu);

  /**
   * @brief Initialize the frame placement cost model
   *
   * The default `nu` is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   * @param[in] Fref        Reference frame placement
   */
  CostModelFramePlacementTpl(boost::shared_ptr<StateMultibody> state,
                             boost::shared_ptr<ActivationModelAbstract> activation, const FramePlacement& Fref);

  /**
   * @brief Initialize the frame placement cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$).
   *
   * @param[in] state  State of the multibody system
   * @param[in] Fref   Reference frame placement
   * @param[in] nu     Dimension of the control vector
   */
  CostModelFramePlacementTpl(boost::shared_ptr<StateMultibody> state, const FramePlacement& Fref,
                             const std::size_t nu);

  /**
   * @brief Initialize the frame placement cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$).
   * Furthermore, the default `nu` is obtained from `StateAbstractTpl::get_nv()`
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   * @param[in] Fref        Reference frame placement
   */
  CostModelFramePlacementTpl(boost::shared_ptr<StateMultibody> state, const FramePlacement& Fref);
  virtual ~CostModelFramePlacementTpl();

  /**
   * @brief Compute the frame placement cost
   *
   * @param[in] data  Frame placement cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the derivatives of the frame placement cost
   *
   * @param[in] data  Frame-placement cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Create the frame placement cost data
   */
  virtual boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

 protected:
  /**
   * @brief Return the frame placement reference
   */
  virtual void get_referenceImpl(const std::type_info& ti, void* pv);

  /**
   * @brief Modify the frame placement reference
   */
  virtual void set_referenceImpl(const std::type_info& ti, const void* pv);

  using Base::activation_;
  using Base::nu_;
  using Base::residual_;
  using Base::state_;
  using Base::unone_;

 private:
  FramePlacement Mref_;  //!< Reference frame placement
};

template <typename _Scalar>
struct CostDataFramePlacementTpl : public CostDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  typedef typename MathBase::Matrix6s Matrix6s;
  typedef typename MathBase::Vector6s Vector6s;

  template <template <typename Scalar> class Model>
  CostDataFramePlacementTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : Base(model, data), Arr_J(6, model->get_state()->get_nv()) {
    Arr_J.setZero();
  }

  Matrix6xs Arr_J;

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
#include "crocoddyl/multibody/costs/frame-placement.hxx"

#endif  // CROCODDYL_MULTIBODY_COSTS_FRAME_PLACEMENT_HPP_
