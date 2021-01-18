///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_COM_POSITION_HPP_
#define CROCODDYL_MULTIBODY_COSTS_COM_POSITION_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/cost-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/utils/deprecate.hpp"

namespace crocoddyl {

/**
 * @brief CoM position cost
 *
 * This cost function defines a residual vector as \f$\mathbf{r}=\mathbf{c}-\mathbf{c}^*\f$, where
 * \f$\mathbf{c},\mathbf{c}^*\in~\mathbb{R}^3\f$ are the current and reference CoM position, respetively. Note that the
 * dimension of the residual vector is obtained from 3.
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
class CostModelCoMPositionTpl : public CostModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelAbstractTpl<Scalar> Base;
  typedef CostDataCoMPositionTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize the CoM position cost model
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   * @param[in] cref        Reference CoM position
   * @param[in] nu          Dimension of the control vector
   */
  CostModelCoMPositionTpl(boost::shared_ptr<StateMultibody> state,
                          boost::shared_ptr<ActivationModelAbstract> activation, const Vector3s& cref,
                          const std::size_t& nu);

  /**
   * @brief Initialize the CoM position cost model
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   * @param[in] cref        Reference CoM position
   */
  CostModelCoMPositionTpl(boost::shared_ptr<StateMultibody> state,
                          boost::shared_ptr<ActivationModelAbstract> activation, const Vector3s& cref);

  /**
   * @brief Initialize the CoM position cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$).
   *
   * @param[in] state  State of the multibody system
   * @param[in] cref   Reference CoM position
   * @param[in] nu     Dimension of the control vector
   */
  CostModelCoMPositionTpl(boost::shared_ptr<StateMultibody> state, const Vector3s& cref, const std::size_t& nu);

  /**
   * @brief Initialize the CoM position cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$).
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state  State of the multibody system
   * @param[in] cref   Reference CoM position
   */
  CostModelCoMPositionTpl(boost::shared_ptr<StateMultibody> state, const Vector3s& cref);
  virtual ~CostModelCoMPositionTpl();

  /**
   * @brief Compute the CoM position cost
   *
   * @param[in] data  CoM position cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the derivatives of the CoM position cost
   *
   * @param[in] data  CoM position cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);
  virtual boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

  DEPRECATED("Use set_reference<MathBaseTpl<Scalar>::Vector3s>()", void set_cref(const Vector3s& cref_in));
  DEPRECATED("Use get_reference<MathBaseTpl<Scalar>::Vector3s>()", const Vector3s& get_cref() const);

 protected:
  /**
   * @brief Modify the CoM position reference
   */
  virtual void set_referenceImpl(const std::type_info& ti, const void* pv);

  /**
   * @brief Return the CoM position reference
   */
  virtual void get_referenceImpl(const std::type_info& ti, void* pv) const;

  using Base::activation_;
  using Base::nu_;
  using Base::state_;
  using Base::unone_;

 private:
  Vector3s cref_;  //!< Reference CoM position
};

template <typename _Scalar>
struct CostDataCoMPositionTpl : public CostDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::Matrix3xs Matrix3xs;

  template <template <typename Scalar> class Model>
  CostDataCoMPositionTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : Base(model, data), Arr_Jcom(3, model->get_state()->get_nv()) {
    Arr_Jcom.setZero();
    // Check that proper shared data has been passed
    DataCollectorMultibodyTpl<Scalar>* d = dynamic_cast<DataCollectorMultibodyTpl<Scalar>*>(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorMultibody");
    }

    // Avoids data casting at runtime
    pinocchio = d->pinocchio;
  }

  pinocchio::DataTpl<Scalar>* pinocchio;
  Matrix3xs Arr_Jcom;

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
#include "crocoddyl/multibody/costs/com-position.hxx"

#endif  // CROCODDYL_MULTIBODY_COSTS_COM_POSITION_HPP_
