///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, University of Edinburgh, IRI: CSIC-UPC
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_CONSTRAINTS_FRAME_VELOCITY_EQUALITY_HPP_
#define CROCODDYL_MULTIBODY_CONSTRAINTS_FRAME_VELOCITY_EQUALITY_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/constraint-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/frames.hpp"

namespace crocoddyl {

/**
 * @brief Frame velocity equality constraint
 *
 * This equality constraint function imposes a reference velocity for a given frame, i.e.
 * \f$\mathbf{v}-\mathbf{v}^*=\mathbf{0}\f$, where \f$\mathbf{v},\mathbf{v}^*\in~\mathbb{R}^3\f$ are the
 * current and reference velocities, respectively. Note that the dimension of the constraint residual vector
 * is 6.
 *
 * Both constraint residuals and its Jacobians are computed analytically.
 * As described in ConstraintModelAbstractTpl(), the constraint residual and its Jacobians are calculated by `calc` and
 * `calcDiff`, respectively.
 *
 * \sa `ConstraintModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ConstraintModelFrameVelocityEqualityTpl : public ConstraintModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ConstraintModelAbstractTpl<Scalar> Base;
  typedef ConstraintDataFrameVelocityEqualityTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ConstraintDataAbstractTpl<Scalar> ConstraintDataAbstract;
  typedef FrameMotionTpl<Scalar> FrameMotion;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize the frame velocity equality constraint model
   *
   * @param[in] state       State of the multibody system
   * @param[in] Fref        Reference frame motion
   * @param[in] nu          Dimension of the control vector
   */
  ConstraintModelFrameVelocityEqualityTpl(boost::shared_ptr<StateMultibody> state, const FrameMotion& Fref,
                                          const std::size_t& nu);

  /**
   * @brief Initialize the frame velocity equality constraint model
   *
   * The default `nu` is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state       State of the multibody system
   * @param[in] Fref        Reference frame motion
   */
  ConstraintModelFrameVelocityEqualityTpl(boost::shared_ptr<StateMultibody> state, const FrameMotion& Fref);
  virtual ~ConstraintModelFrameVelocityEqualityTpl();

  /**
   * @brief Compute the residual of the frame velocity constraint
   *
   * @param[in] data  Frame velocity constraint data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<ConstraintDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the Jacobians of the frame velocity constraint
   *
   * @param[in] data  Frame-velocity constraint data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ConstraintDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Create the frame velocity constraint data
   */
  virtual boost::shared_ptr<ConstraintDataAbstract> createData(DataCollectorAbstract* const data);

 protected:
  /**
   * @brief Modify the frame velocity reference
   */
  virtual void set_referenceImpl(const std::type_info& ti, const void* pv);

  /**
   * @brief Return the frame velocity reference
   */
  virtual void get_referenceImpl(const std::type_info& ti, void* pv) const;

  using Base::nu_;
  using Base::state_;
  using Base::unone_;

 private:
  FrameMotion vref_;                                                      //!< Reference frame velocity
  boost::shared_ptr<typename StateMultibody::PinocchioModel> pin_model_;  //!< Pinocchio model
};

template <typename _Scalar>
struct ConstraintDataFrameVelocityEqualityTpl : public ConstraintDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ConstraintDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::Vector6s Vector6s;

  template <template <typename Scalar> class Model>
  ConstraintDataFrameVelocityEqualityTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : Base(model, data) {
    h.setZero();
    // Check that proper shared data has been passed
    DataCollectorMultibodyTpl<Scalar>* d = dynamic_cast<DataCollectorMultibodyTpl<Scalar>*>(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorMultibody");
    }

    // Avoids data casting at runtime
    pinocchio = d->pinocchio;
  }

  pinocchio::DataTpl<Scalar>* pinocchio;
  Vector6s h;
  using Base::Hu;
  using Base::Hx;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/constraints/frame-velocity-equality.hxx"

#endif  // CROCODDYL_MULTIBODY_CONSTRAINTS_FRAME_VELOCITY_EQUALITY_HPP_
