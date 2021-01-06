///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_RESIDUALS_FRAME_VELOCITY_HPP_
#define CROCODDYL_MULTIBODY_RESIDUALS_FRAME_VELOCITY_HPP_

#include <pinocchio/multibody/fwd.hpp>
#include <pinocchio/spatial/motion.hpp>

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief Frame velocity residual
 *
 * This residual function defines a tracking of frame velocity as \f$\mathbf{r}=\mathbf{v}-\mathbf{v}^*\f$, where
 * \f$\mathbf{v},\mathbf{v}^*\in~T_{\mathbf{p}}~\mathbb{SE(3)}\f$ are the current and reference frame velocities,
 * respectively. Note that the tangent vector is described by the frame placement \f$\mathbf{p}\f$, and the dimension
 * of the residual vector is 6.
 * Furthermore, the Jacobians of the residual function are computed analytically.
 *
 * As described in `ResidualModelAbstractTpl`, the residual vector and its Jacobians are calculated by `calc` and
 * `calcDiff`, respectively.
 *
 * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ResidualModelFrameVelocityTpl : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataFrameVelocityTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef pinocchio::MotionTpl<Scalar> Motion;

  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize the frame velocity residual model
   *
   * @param[in] state       State of the multibody system
   * @param[in] id          Reference frame id
   * @param[in] velocity    Reference velocity
   * @param[in] type        Reference type of velocity (WORLD, LOCAL, LOCAL_WORLD_ALIGNED)
   * @param[in] nu          Dimension of the control vector
   */
  ResidualModelFrameVelocityTpl(boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
                                const Motion& velocity, const pinocchio::ReferenceFrame type, const std::size_t nu);

  /**
   * @brief Initialize the frame velocity residual model
   *
   * The default `nu` value is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state       State of the multibody system
   * @param[in] id          Reference frame id
   * @param[in] velocity    Reference velocity
   * @param[in] type        Reference type of velocity (WORLD, LOCAL, LOCAL_WORLD_ALIGNED)
   */
  ResidualModelFrameVelocityTpl(boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
                                const Motion& velocity, const pinocchio::ReferenceFrame type);
  virtual ~ResidualModelFrameVelocityTpl();

  /**
   * @brief Compute the frame velocity residual vector
   *
   * @param[in] data  Frame velocity residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the Jacobians of the frame velocity residual
   *
   * @param[in] data  Frame velocity residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Create the frame velocity residual data
   */
  virtual boost::shared_ptr<ResidualDataAbstract> createData(DataCollectorAbstract* const data);

  void set_id(const pinocchio::FrameIndex id);
  void set_velocity(const Motion& velocity);
  void set_type(const pinocchio::ReferenceFrame type);
  pinocchio::FrameIndex get_id() const;
  const Motion& get_velocity() const;
  pinocchio::ReferenceFrame get_type() const;

 protected:
  using Base::nr_;
  using Base::nu_;
  using Base::state_;
  using Base::unone_;

 private:
  pinocchio::FrameIndex id_;                                              //!< Reference frame id
  Motion velocity_;                                                       //!< Reference velocity
  pinocchio::ReferenceFrame type_;                                        //!< Reference type of velocity
  boost::shared_ptr<typename StateMultibody::PinocchioModel> pin_model_;  //!< Pinocchio model
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
  ResidualDataFrameVelocityTpl(Model<Scalar>* const model, DataCollectorAbstract* const data) : Base(model, data) {
    // Check that proper shared data has been passed
    DataCollectorMultibodyTpl<Scalar>* d = dynamic_cast<DataCollectorMultibodyTpl<Scalar>*>(shared);
    if (d == NULL) {
      throw_pretty("Invalid argument: the shared data should be derived from DataCollectorMultibody");
    }

    // Avoids data casting at runtime
    pinocchio = d->pinocchio;
  }

  pinocchio::DataTpl<Scalar>* pinocchio;
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

#endif  // CROCODDYL_MULTIBODY_RESIDUALS_FRAME_VELOCITY_HPP_
