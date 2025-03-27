///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_RESIDUALS_FRAME_VELOCITY_HPP_
#define CROCODDYL_MULTIBODY_RESIDUALS_FRAME_VELOCITY_HPP_

#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"

namespace crocoddyl {

/**
 * @brief Frame velocity residual
 *
 * This residual function defines a tracking of frame velocity as
 * \f$\mathbf{r}=\mathbf{v}-\mathbf{v}^*\f$, where
 * \f$\mathbf{v},\mathbf{v}^*\in~T_{\mathbf{p}}~\mathbb{SE(3)}\f$ are the
 * current and reference frame velocities, respectively. Note that the tangent
 * vector is described by the frame placement \f$\mathbf{p}\f$, and the
 * dimension of the residual vector is 6. Furthermore, the Jacobians of the
 * residual function are computed analytically.
 *
 * As described in `ResidualModelAbstractTpl`, the residual vector and its
 * Jacobians are calculated by `calc` and `calcDiff`, respectively.
 *
 * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ResidualModelFrameVelocityTpl : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ResidualModelBase, ResidualModelFrameVelocityTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataFrameVelocityTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef pinocchio::MotionTpl<Scalar> Motion;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize the frame velocity residual model
   *
   * @param[in] state   State of the multibody system
   * @param[in] id      Reference frame id
   * @param[in] vref    Reference velocity
   * @param[in] type    Reference type of velocity (WORLD, LOCAL,
   * LOCAL_WORLD_ALIGNED)
   * @param[in] nu      Dimension of the control vector
   */
  ResidualModelFrameVelocityTpl(std::shared_ptr<StateMultibody> state,
                                const pinocchio::FrameIndex id,
                                const Motion& vref,
                                const pinocchio::ReferenceFrame type,
                                const std::size_t nu);

  /**
   * @brief Initialize the frame velocity residual model
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state   State of the multibody system
   * @param[in] id      Reference frame id
   * @param[in] vref    Reference velocity
   * @param[in] type    Reference type of velocity (WORLD, LOCAL,
   * LOCAL_WORLD_ALIGNED)
   */
  ResidualModelFrameVelocityTpl(std::shared_ptr<StateMultibody> state,
                                const pinocchio::FrameIndex id,
                                const Motion& vref,
                                const pinocchio::ReferenceFrame type);
  virtual ~ResidualModelFrameVelocityTpl() = default;

  /**
   * @brief Compute the frame velocity residual vector
   *
   * @param[in] data  Frame velocity residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const std::shared_ptr<ResidualDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Compute the Jacobians of the frame velocity residual
   *
   * @param[in] data  Frame velocity residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ResidualDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u) override;

  /**
   * @brief Create the frame velocity residual data
   */
  virtual std::shared_ptr<ResidualDataAbstract> createData(
      DataCollectorAbstract* const data) override;

  /**
   * @brief Cast the frame-velocity residual model to a different scalar type.
   *
   * It is useful for operations requiring different precision or scalar types.
   *
   * @tparam NewScalar The new scalar type to cast to.
   * @return ResidualModelFrameVelocityTpl<NewScalar> A residual model with the
   * new scalar type.
   */
  template <typename NewScalar>
  ResidualModelFrameVelocityTpl<NewScalar> cast() const;

  /**
   * @brief Return the reference frame id
   */
  pinocchio::FrameIndex get_id() const;

  /**
   * @brief Return the reference velocity
   */
  const Motion& get_reference() const;

  /**
   * @brief Return the reference type of velocity
   */
  pinocchio::ReferenceFrame get_type() const;

  /**
   * @brief Modify reference frame id
   */
  void set_id(const pinocchio::FrameIndex id);

  /**
   * @brief Modify reference velocity
   */
  void set_reference(const Motion& velocity);

  /**
   * @brief Modify reference type of velocity
   */
  void set_type(const pinocchio::ReferenceFrame type);

  /**
   * @brief Print relevant information of the frame-velocity residual
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override;

 protected:
  using Base::nr_;
  using Base::nu_;
  using Base::state_;
  using Base::u_dependent_;

 private:
  pinocchio::FrameIndex id_;        //!< Reference frame id
  Motion vref_;                     //!< Reference velocity
  pinocchio::ReferenceFrame type_;  //!< Reference type of velocity
  std::shared_ptr<typename StateMultibody::PinocchioModel>
      pin_model_;  //!< Pinocchio model
};

template <typename _Scalar>
struct ResidualDataFrameVelocityTpl : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::Matrix6xs Matrix6xs;

  template <template <typename Scalar> class Model>
  ResidualDataFrameVelocityTpl(Model<Scalar>* const model,
                               DataCollectorAbstract* const data)
      : Base(model, data) {
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
  virtual ~ResidualDataFrameVelocityTpl() = default;

  pinocchio::DataTpl<Scalar>* pinocchio;  //!< Pinocchio data
  using Base::r;
  using Base::Ru;
  using Base::Rx;
  using Base::shared;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/residuals/frame-velocity.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::ResidualModelFrameVelocityTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::ResidualDataFrameVelocityTpl)

#endif  // CROCODDYL_MULTIBODY_RESIDUALS_FRAME_VELOCITY_HPP_
