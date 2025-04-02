///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_RESIDUALS_FRAME_ROTATION_HPP_
#define CROCODDYL_MULTIBODY_RESIDUALS_FRAME_ROTATION_HPP_

#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

/**
 * @brief Frame rotation residual
 *
 * This residual function is defined as
 * \f$\mathbf{r}=\mathbf{R}\ominus\mathbf{R}^*\f$, where
 * \f$\mathbf{R},\mathbf{R}^*\in~\mathbb{SO(3)}\f$ are the current and reference
 * frame rotations, respectively. Note that the dimension of the residual vector
 * is 3. Furthermore, the Jacobians of the residual function are computed
 * analytically.
 *
 * As described in `ResidualModelAbstractTpl()`, the residual value and its
 * Jacobians are calculated by `calc` and `calcDiff`, respectively.
 *
 * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ResidualModelFrameRotationTpl : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ResidualModelBase, ResidualModelFrameRotationTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataFrameRotationTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Matrix3s Matrix3s;

  /**
   * @brief Initialize the frame rotation residual model
   *
   * @param[in] state  State of the multibody system
   * @param[in] id     Reference frame id
   * @param[in] Rref   Reference frame rotation
   * @param[in] nu     Dimension of the control vector
   */
  ResidualModelFrameRotationTpl(std::shared_ptr<StateMultibody> state,
                                const pinocchio::FrameIndex id,
                                const Matrix3s& Rref, const std::size_t nu);

  /**
   * @brief Initialize the frame rotation residual model
   *
   * The default `nu` is equals to StateAbstractTpl::get_nv().
   *
   * @param[in] state  State of the multibody system
   * @param[in] id     Reference frame id
   * @param[in] Rref   Reference frame rotation
   */
  ResidualModelFrameRotationTpl(std::shared_ptr<StateMultibody> state,
                                const pinocchio::FrameIndex id,
                                const Matrix3s& Rref);
  virtual ~ResidualModelFrameRotationTpl() = default;

  /**
   * @brief Compute the frame rotation residual
   *
   * @param[in] data  Frame rotation residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<ResidualDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Compute the derivatives of the frame rotation residual
   *
   * @param[in] data  Frame rotation residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Create the frame rotation residual data
   */
  virtual std::shared_ptr<ResidualDataAbstract> createData(
      DataCollectorAbstract* const data) override;

  /**
   * @brief Cast the frame-rotation residual model to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return ResidualModelFrameRotationTpl<NewScalar> A residual model with the
   * new scalar type.
   */
  template <typename NewScalar>
  ResidualModelFrameRotationTpl<NewScalar> cast() const;

  /**
   * @brief Return the reference frame id
   */
  pinocchio::FrameIndex get_id() const;

  /**
   * @brief Return the reference frame rotation
   */
  const Matrix3s& get_reference() const;

  /**
   * @brief Modify the reference frame id
   */
  void set_id(const pinocchio::FrameIndex id);

  /**
   * @brief Modify the reference frame rotation
   */
  void set_reference(const Matrix3s& reference);

  /**
   * @brief Print relevant information of the frame-rotation residual
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override;

 protected:
  using Base::nu_;
  using Base::state_;
  using Base::u_dependent_;
  using Base::v_dependent_;

 private:
  pinocchio::FrameIndex id_;  //!< Reference frame id
  Matrix3s Rref_;             //!< Reference frame rotation
  Matrix3s oRf_inv_;          //!< Inverse reference rotation
  std::shared_ptr<typename StateMultibody::PinocchioModel>
      pin_model_;  //!< Pinocchio model
};

template <typename _Scalar>
struct ResidualDataFrameRotationTpl : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::Vector3s Vector3s;
  typedef typename MathBase::Matrix3s Matrix3s;
  typedef typename MathBase::Matrix3xs Matrix3xs;
  typedef typename MathBase::Matrix6xs Matrix6xs;

  template <template <typename Scalar> class Model>
  ResidualDataFrameRotationTpl(Model<Scalar>* const model,
                               DataCollectorAbstract* const data)
      : Base(model, data), rJf(3, 3), fJf(6, model->get_state()->get_nv()) {
    r.setZero();
    rRf.setIdentity();
    rJf.setZero();
    fJf.setZero();
    // Check that proper shared data has been passed
    DataCollectorMultibodyTpl<Scalar>* d =
        dynamic_cast<DataCollectorMultibodyTpl<Scalar>*>(shared);
    if (d == NULL) {
      throw_pretty(
          "Invalid argument: the shared data should be derived from "
          "DataCollectorMultibody");
    }

    // Avoids data casting at runtime
    pinocchio = d->pinocchio;
  }
  virtual ~ResidualDataFrameRotationTpl() = default;

  pinocchio::DataTpl<Scalar>* pinocchio;  //!< Pinocchio data
  Matrix3s rRf;                           //!< Rotation error of the frame
  Matrix3s rJf;                           //!< Error Jacobian of the frame
  Matrix6xs fJf;                          //!< Local Jacobian of the frame

  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/residuals/frame-rotation.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::ResidualModelFrameRotationTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::ResidualDataFrameRotationTpl)

#endif  // CROCODDYL_MULTIBODY_RESIDUALS_FRAME_ROTATION_HPP_
