///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_FRAME_ROTATION_HPP_
#define CROCODDYL_MULTIBODY_COSTS_FRAME_ROTATION_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/frames.hpp"
#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/core/utils/deprecate.hpp"

namespace crocoddyl {

/**
 * @brief Frame rotation cost
 *
 * This cost function defines a residual vector as \f$\mathbf{r}=\mathbf{R}\ominus\mathbf{R}^*\f$, where
 * \f$\mathbf{R},\mathbf{R}^*\in~\mathbb{SO(3)}\f$ are the current and reference frame rotations, respectively. Note
 * that the dimension of the residual vector is 3.
 *
 * Both cost and residual derivatives are computed analytically.
 * For the computation of the cost Hessian, we use the Gauss-Newton approximation, e.g.
 * \f$\mathbf{l_{xu}} = \mathbf{l_{x}}^T \mathbf{l_{u}} \f$.
 *
 * As described in CostModelAbstractTpl(), the cost value and its derivatives are calculated by `calc` and `calcDiff`,
 * respectively.
 *
 * \sa `CostModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class CostModelFrameRotationTpl : public CostModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelAbstractTpl<Scalar> Base;
  typedef CostDataFrameRotationTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef CostDataAbstractTpl<Scalar> CostDataAbstract;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef ActivationModelQuadTpl<Scalar> ActivationModelQuad;
  typedef FrameRotationTpl<Scalar> FrameRotation;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Matrix3s Matrix3s;

  /**
   * @brief Initialize the frame rotation cost model
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   * @param[in] Fref        Reference frame rotation
   * @param[in] nu          Dimension of the control vector
   */
  CostModelFrameRotationTpl(boost::shared_ptr<StateMultibody> state,
                            boost::shared_ptr<ActivationModelAbstract> activation, const FrameRotation& Fref,
                            const std::size_t& nu);

  /**
   * @brief Initialize the frame rotation cost model
   *
   * The default `nu` is equals to StateAbstractTpl::get_nv().
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   * @param[in] Fref        Reference frame rotation
   * @param[in] nu          Dimension of the control vector
   */
  CostModelFrameRotationTpl(boost::shared_ptr<StateMultibody> state,
                            boost::shared_ptr<ActivationModelAbstract> activation, const FrameRotation& Fref);

  /**
   * @brief Initialize the frame rotation cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$).
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   * @param[in] Fref        Reference frame rotation
   * @param[in] nu          Dimension of the control vector
   */
  CostModelFrameRotationTpl(boost::shared_ptr<StateMultibody> state, const FrameRotation& Fref, const std::size_t& nu);

  /**
   * @brief Initialize the frame rotation cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$).
   * Furthermore, the default `nu` is equals to StateAbstractTpl::get_nv()
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   * @param[in] Fref        Reference frame rotation
   * @param[in] nu          Dimension of the control vector
   */
  CostModelFrameRotationTpl(boost::shared_ptr<StateMultibody> state, const FrameRotation& Fref);
  virtual ~CostModelFrameRotationTpl();

  /**
   * @brief Compute the frame rotation cost
   *
   * @param[in] data  Frame rotation cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the derivatives of the frame rotation cost
   *
   * @param[in] data  Frame rotation cost data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Create the frame rotation cost data
   */
  virtual boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data);

  DEPRECATED("Use set_reference<FrameRotationTpl<Scalar> >()", void set_Rref(const FrameRotation& Rref_in));
  DEPRECATED("Use get_reference<FrameRotationTpl<Scalar> >()", const FrameRotation& get_Rref() const);

 protected:
  /**
   * @brief Modify the frame rotation reference
   */
  virtual void set_referenceImpl(const std::type_info& ti, const void* pv);

  /**
   * @brief Return the frame rotation reference
   */
  virtual void get_referenceImpl(const std::type_info& ti, void* pv) const;

  using Base::activation_;
  using Base::nu_;
  using Base::state_;
  using Base::unone_;

 private:
  FrameRotation Rref_;                                                    //!< Reference frame rotation
  Matrix3s oRf_inv_;                                                      //!< Inverser reference rotation
  boost::shared_ptr<typename StateMultibody::PinocchioModel> pin_model_;  //!< Pinocchio model
};

template <typename _Scalar>
struct CostDataFrameRotationTpl : public CostDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::Matrix3s Matrix3s;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Matrix3xs Matrix3xs;
  typedef typename MathBase::Matrix6xs Matrix6xs;

  template <template <typename Scalar> class Model>
  CostDataFrameRotationTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : Base(model, data),
        J(3, model->get_state()->get_nv()),
        rJf(3, 3),
        fJf(6, model->get_state()->get_nv()),
        Arr_J(3, model->get_state()->get_nv()) {
    r.setZero();
    rRf.setIdentity();
    J.setZero();
    rJf.setZero();
    fJf.setZero();
    Arr_J.setZero();
    // Check that proper shared data has been passed
    DataCollectorMultibodyTpl<Scalar>* d = dynamic_cast<DataCollectorMultibodyTpl<Scalar>*>(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorMultibody");
    }

    // Avoids data casting at runtime
    pinocchio = d->pinocchio;
  }

  pinocchio::DataTpl<Scalar>* pinocchio;
  Vector3s r;
  Matrix3s rRf;
  Matrix3xs J;
  Matrix3s rJf;
  Matrix6xs fJf;
  Matrix3xs Arr_J;

  using Base::activation;
  using Base::cost;
  using Base::Lu;
  using Base::Luu;
  using Base::Lx;
  using Base::Lxu;
  using Base::Lxx;
  using Base::shared;
  // using Base::r;
  using Base::Ru;
  using Base::Rx;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/costs/frame-rotation.hxx"

#endif  // CROCODDYL_MULTIBODY_COSTS_FRAME_ROTATION_HPP_
