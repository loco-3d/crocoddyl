///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_CONSTRAINTS_FRAME_PLACEMENT_EQUALITY_HPP_
#define CROCODDYL_MULTIBODY_CONSTRAINTS_FRAME_PLACEMENT_EQUALITY_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/constraint-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/frames.hpp"
// #include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief Frame placement equality constraint
 *
 * This equality constraint function imposes a reference placement of a given frame, i.e.
 * \f$\mathbf{p}\ominus\mathbf{p}^*=\mathbf{0}\f$, where \f$\mathbf{p},\mathbf{p}^*\in~\mathbb{SE(3)}\f$ are the
 * current and reference frame placements, respectively. Note that the dimension of the constraint residual vector
 * is 6.
 *
 * Both constraint residuals and its Jacobians are computed analytically.
 * As described in ConstraintModelAbstractTpl(), the constraint residual and its Jacobians are calculated by `calc` and
 * `calcDiff`, respectively.
 *
 * \sa `ConstraintModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ConstraintModelFramePlacementEqualityTpl : public ConstraintModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ConstraintModelAbstractTpl<Scalar> Base;
  typedef ConstraintDataFramePlacementEqualityTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ConstraintDataAbstractTpl<Scalar> ConstraintDataAbstract;
  typedef FramePlacementTpl<Scalar> FramePlacement;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize the frame placement equality constraint model
   *
   * @param[in] state       State of the multibody system
   * @param[in] Fref        Reference frame placement
   * @param[in] nu          Dimension of the control vector
   */
  ConstraintModelFramePlacementEqualityTpl(boost::shared_ptr<StateMultibody> state, const FramePlacement& Fref,
                                           const std::size_t& nu);

  /**
   * @brief Initialize the frame placement equality constraint model
   *
   * The default `nu` is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state       State of the multibody system
   * @param[in] Fref        Reference frame placement
   */
  ConstraintModelFramePlacementEqualityTpl(boost::shared_ptr<StateMultibody> state, const FramePlacement& Fref);
  virtual ~ConstraintModelFramePlacementEqualityTpl();

  /**
   * @brief Compute the residual of the frame placement constraint
   *
   * @param[in] data  Frame placement constraint data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<ConstraintDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the Jacobians of the frame placement constraint
   *
   * @param[in] data  Frame-placement constraint data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ConstraintDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Create the frame placement constraint data
   */
  virtual boost::shared_ptr<ConstraintDataAbstract> createData(DataCollectorAbstract* const data);

 protected:
  /**
   * @brief Modify the frame placement reference
   */
  virtual void set_referenceImpl(const std::type_info& ti, const void* pv);

  /**
   * @brief Return the frame placement reference
   */
  virtual void get_referenceImpl(const std::type_info& ti, void* pv) const;

  using Base::nu_;
  using Base::state_;
  using Base::unone_;

 private:
  FramePlacement Mref_;                                                   //!< Reference frame placement
  pinocchio::SE3Tpl<Scalar> oMf_inv_;                                     //!< Inverse reference placement
  boost::shared_ptr<typename StateMultibody::PinocchioModel> pin_model_;  //!< Pinocchio model
};

template <typename _Scalar>
struct ConstraintDataFramePlacementEqualityTpl : public ConstraintDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ConstraintDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  typedef typename MathBase::Matrix6s Matrix6s;
  typedef typename MathBase::Vector6s Vector6s;

  template <template <typename Scalar> class Model>
  ConstraintDataFramePlacementEqualityTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : Base(model, data), J(6, model->get_state()->get_nv()), rJf(6, 6), fJf(6, model->get_state()->get_nv()) {
    h.setZero();
    J.setZero();
    rJf.setZero();
    fJf.setZero();
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
  pinocchio::SE3Tpl<Scalar> rMf;
  Matrix6xs J;
  Matrix6s rJf;
  Matrix6xs fJf;

  using Base::Hu;
  using Base::Hx;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/constraints/frame-placement-equality.hxx"

#endif  // CROCODDYL_MULTIBODY_CONSTRAINTS_FRAME_PLACEMENT_EQUALITY_HPP_
