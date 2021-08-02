///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_RESIDUALS_FRAME_PLACEMENT_HPP_
#define CROCODDYL_MULTIBODY_RESIDUALS_FRAME_PLACEMENT_HPP_

#include <pinocchio/multibody/fwd.hpp>
#include <pinocchio/spatial/se3.hpp>

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief Frame placement residual
 *
 * This residual function defines the frame placement tracking as \f$\mathbf{r}=\mathbf{p}\ominus\mathbf{p}^*\f$, where
 * \f$\mathbf{p},\mathbf{p}^*\in~\mathbb{SE(3)}\f$ are the current and reference frame placements, respectively. Note
 * that the dimension of the residual vector is 6. Furthermore, the Jacobians of the residual function are
 * computed analytically.
 *
 * As described in `ResidualModelAbstractTpl()`, the residual value and its Jacobians are calculated by `calc` and
 * `calcDiff`, respectively.
 *
 * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ResidualModelFramePlacementTpl : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataFramePlacementTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef pinocchio::SE3Tpl<Scalar> SE3;

  /**
   * @brief Initialize the frame placement residual model
   *
   * @param[in] state  State of the multibody system
   * @param[in] id     Reference frame id
   * @param[in] pref   Reference frame placement
   * @param[in] nu     Dimension of the control vector
   */
  ResidualModelFramePlacementTpl(boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
                                 const SE3& pref, const std::size_t nu);

  /**
   * @brief Initialize the frame placement residual model
   *
   * The default `nu` is obtained from `StateAbstractTpl::get_nv()`.
   *
   * @param[in] state  State of the multibody system
   * @param[in] id     Reference frame id
   * @param[in] pref   Reference frame placement
   */
  ResidualModelFramePlacementTpl(boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
                                 const SE3& pref);
  virtual ~ResidualModelFramePlacementTpl();

  /**
   * @brief Compute the frame placement residual
   *
   * @param[in] data  Frame placement residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the derivatives of the frame placement residual
   *
   * @param[in] data  Frame-placement residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Create the frame placement residual data
   */
  virtual boost::shared_ptr<ResidualDataAbstract> createData(DataCollectorAbstract* const data);

  /**
   * @brief Return the reference frame id
   */
  pinocchio::FrameIndex get_id() const;

  /**
   * @brief Return the reference frame placement
   */
  const SE3& get_reference() const;

  /**
   * @brief Modify the reference frame id
   */
  void set_id(const pinocchio::FrameIndex id);

  /**
   * @brief Modify the reference frame placement
   */
  void set_reference(const SE3& reference);

  /**
   * @brief Print relevant information of the frame-placement residual
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const;

 protected:
  using Base::nu_;
  using Base::state_;
  using Base::u_dependent_;
  using Base::unone_;
  using Base::v_dependent_;

 private:
  pinocchio::FrameIndex id_;                                              //!< Reference frame id
  SE3 pref_;                                                              //!< Reference frame placement
  pinocchio::SE3Tpl<Scalar> oMf_inv_;                                     //!< Inverse reference placement
  boost::shared_ptr<typename StateMultibody::PinocchioModel> pin_model_;  //!< Pinocchio model
};

template <typename _Scalar>
struct ResidualDataFramePlacementTpl : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::Matrix6xs Matrix6xs;
  typedef typename MathBase::Matrix6s Matrix6s;
  typedef typename MathBase::Vector6s Vector6s;

  template <template <typename Scalar> class Model>
  ResidualDataFramePlacementTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : Base(model, data), rJf(6, 6), fJf(6, model->get_state()->get_nv()) {
    r.setZero();
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

  pinocchio::DataTpl<Scalar>* pinocchio;  //!< Pinocchio data
  pinocchio::SE3Tpl<Scalar> rMf;          //!< Error frame placement of the frame
  Matrix6s rJf;                           //!< Error Jacobian of the frame
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
#include "crocoddyl/multibody/residuals/frame-placement.hxx"

#endif  // CROCODDYL_MULTIBODY_RESIDUALS_FRAME_PLACEMENT_HPP_
