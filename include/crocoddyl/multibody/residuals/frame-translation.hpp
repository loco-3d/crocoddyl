
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_RESIDUALS_FRAME_TRANSLATION_HPP_
#define CROCODDYL_MULTIBODY_RESIDUALS_FRAME_TRANSLATION_HPP_

#include <pinocchio/multibody/fwd.hpp>

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/data/multibody.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief Frame translation residual
 *
 * This residual function defines the tracking of a frame translation as \f$\mathbf{r}=\mathbf{t}-\mathbf{t}^*\f$,
 * where \f$\mathbf{t},\mathbf{t}^*\in~\mathbb{R}^3\f$ are the current and reference frame translations, respectively.
 * Note that the dimension of the residual vector is 3. Furthermore, the Jacobians of the residual function are
 * computed analytically.
 *
 * As described in `ResidualModelAbstractTpl()`, the residual value and its Jacobians are calculated by `calc` and
 * `calcDiff`, respectively.
 *
 * \sa `ResidualModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ResidualModelFrameTranslationTpl : public ResidualModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualModelAbstractTpl<Scalar> Base;
  typedef ResidualDataFrameTranslationTpl<Scalar> Data;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ResidualDataAbstractTpl<Scalar> ResidualDataAbstract;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Vector3s Vector3s;

  /**
   * @brief Initialize the frame translation residual model
   *
   * @param[in] state  State of the multibody system
   * @param[in] id     Reference frame id
   * @param[in] xref   Reference frame translation
   * @param[in] nu     Dimension of the control vector
   */
  ResidualModelFrameTranslationTpl(boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex,
                                   const Vector3s& xref, const std::size_t nu);

  /**
   * @brief Initialize the frame translation residual model
   *
   * The default `nu` is equals to StateAbstractTpl::get_nv().
   *
   * @param[in] state  State of the multibody system
   * @param[in] id     Reference frame id
   * @param[in] xref   Reference frame translation
   */
  ResidualModelFrameTranslationTpl(boost::shared_ptr<StateMultibody> state, const pinocchio::FrameIndex id,
                                   const Vector3s& xref);
  virtual ~ResidualModelFrameTranslationTpl();

  /**
   * @brief Compute the frame translation residual
   *
   * @param[in] data  Frame translation residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calc(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                    const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Compute the derivatives of the frame translation residual
   *
   * @param[in] data  Frame translation residual data
   * @param[in] x     State point \f$\mathbf{x}\in\mathbb{R}^{ndx}\f$
   * @param[in] u     Control input \f$\mathbf{u}\in\mathbb{R}^{nu}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const VectorXs>& x,
                        const Eigen::Ref<const VectorXs>& u);

  /**
   * @brief Create the frame translation residual data
   */
  virtual boost::shared_ptr<ResidualDataAbstract> createData(DataCollectorAbstract* const data);

  /**
   * @brief Return the reference frame id
   */
  pinocchio::FrameIndex get_id() const;

  /**
   * @brief Return the reference frame translation
   */
  const Vector3s& get_reference() const;

  /**
   * @brief Modify the reference frame id
   */
  void set_id(const pinocchio::FrameIndex id);

  /**
   * @brief Modify the reference frame translation reference
   */
  void set_reference(const Vector3s& reference);

  using Base::nu_;
  using Base::state_;
  using Base::u_dependent_;
  using Base::unone_;
  using Base::v_dependent_;

 private:
  pinocchio::FrameIndex id_;                                              //!< Reference frame id
  Vector3s xref_;                                                         //!< Reference frame translation
  boost::shared_ptr<typename StateMultibody::PinocchioModel> pin_model_;  //!< Pinocchio model
};

template <typename _Scalar>
struct ResidualDataFrameTranslationTpl : public ResidualDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ResidualDataAbstractTpl<Scalar> Base;
  typedef DataCollectorAbstractTpl<Scalar> DataCollectorAbstract;
  typedef typename MathBase::Matrix6xs Matrix6xs;

  template <template <typename Scalar> class Model>
  ResidualDataFrameTranslationTpl(Model<Scalar>* const model, DataCollectorAbstract* const data)
      : Base(model, data), fJf(6, model->get_state()->get_nv()) {
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
#include "crocoddyl/multibody/residuals/frame-translation.hxx"

#endif  // CROCODDYL_MULTIBODY_RESIDUALS_FRAME_TRANSLATION_HPP_
