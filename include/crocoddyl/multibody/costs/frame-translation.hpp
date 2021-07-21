
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2021, LAAS-CNRS, University of Edinburgh, INRIA
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_MULTIBODY_COSTS_FRAME_TRANSLATION_HPP_
#define CROCODDYL_MULTIBODY_COSTS_FRAME_TRANSLATION_HPP_

#include "crocoddyl/multibody/fwd.hpp"
#include "crocoddyl/core/costs/residual.hpp"
#include "crocoddyl/multibody/states/multibody.hpp"
#include "crocoddyl/multibody/residuals/frame-translation.hpp"
#include "crocoddyl/core/utils/exception.hpp"

#define CROCODDYL_IGNORE_DEPRECATED_HEADER  // TODO: Remove once the deprecated FrameXX has been removed in a future
                                            // release
#include "crocoddyl/multibody/frames.hpp"
#undef CROCODDYL_IGNORE_DEPRECATED_HEADER

namespace crocoddyl {

#pragma GCC diagnostic push  // TODO: Remove once the deprecated FrameXX has been removed in a future release
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

/**
 * @brief Frame translation cost
 *
 * This cost function defines a residual vector as \f$\mathbf{r}=\mathbf{t}-\mathbf{t}^*\f$, where
 * \f$\mathbf{t},\mathbf{t}^*\in~\mathbb{R}^3\f$ are the current and reference frame translations, respectively. Note
 * that the dimension of the residual vector is 3.
 *
 * Both cost and residual derivatives are computed analytically. For the computation of the cost Hessian, we use the
 * Gauss-Newton approximation, e.g. \f$\mathbf{l_{xu}} = \mathbf{l_{x}}^T \mathbf{l_{u}} \f$.
 *
 * As described in `CostModelResidualTpl()`, the cost value and its derivatives are calculated by `calc` and
 * `calcDiff`, respectively.
 *
 * \sa `CostModelResidualTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class CostModelFrameTranslationTpl : public CostModelResidualTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef CostModelResidualTpl<Scalar> Base;
  typedef StateMultibodyTpl<Scalar> StateMultibody;
  typedef ActivationModelAbstractTpl<Scalar> ActivationModelAbstract;
  typedef ResidualModelFrameTranslationTpl<Scalar> ResidualModelFrameTranslation;
  typedef FrameTranslationTpl<Scalar> FrameTranslation;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::Matrix3xs Matrix3xs;

  /**
   * @brief Initialize the frame translation cost model
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   * @param[in] xref        Reference frame translation
   * @param[in] nu          Dimension of the control vector
   */
  CostModelFrameTranslationTpl(boost::shared_ptr<StateMultibody> state,
                               boost::shared_ptr<ActivationModelAbstract> activation, const FrameTranslation& xref,
                               const std::size_t nu);

  /**
   * @brief Initialize the frame translation cost model
   *
   * The default `nu` is equals to StateAbstractTpl::get_nv().
   *
   * @param[in] state       State of the multibody system
   * @param[in] activation  Activation model
   * @param[in] xref        Reference frame translation
   */
  CostModelFrameTranslationTpl(boost::shared_ptr<StateMultibody> state,
                               boost::shared_ptr<ActivationModelAbstract> activation, const FrameTranslation& xref);

  /**
   * @brief Initialize the frame translation cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$).
   *
   * @param[in] state       State of the multibody system
   * @param[in] xref        Reference frame translation
   * @param[in] nu          Dimension of the control vector
   */
  CostModelFrameTranslationTpl(boost::shared_ptr<StateMultibody> state, const FrameTranslation& xref,
                               const std::size_t nu);

  /**
   * @brief Initialize the frame translation cost model
   *
   * We use `ActivationModelQuadTpl` as a default activation model (i.e. \f$a=\frac{1}{2}\|\mathbf{r}\|^2\f$).
   * Furthermore, the default `nu` is equals to StateAbstractTpl::get_nv()
   *
   * @param[in] state       State of the multibody system
   * @param[in] xref        Reference frame translation
   */
  CostModelFrameTranslationTpl(boost::shared_ptr<StateMultibody> state, const FrameTranslation& xref);
  virtual ~CostModelFrameTranslationTpl();

 protected:
  /**
   * @brief Return the frame translation reference
   */
  virtual void get_referenceImpl(const std::type_info& ti, void* pv);

  /**
   * @brief Modify the frame translation reference
   */
  virtual void set_referenceImpl(const std::type_info& ti, const void* pv);

  using Base::activation_;
  using Base::nu_;
  using Base::residual_;
  using Base::state_;
  using Base::unone_;

 private:
  FrameTranslation xref_;  //!< Reference frame translation
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/multibody/costs/frame-translation.hxx"

#pragma GCC diagnostic pop

#endif  // CROCODDYL_MULTIBODY_COSTS_FRAME_TRANSLATION_HPP_
