///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_GUIDANCE_SMOOTH_SATURATION_HPP_
#define CROCODDYL_CORE_GUIDANCE_SMOOTH_SATURATION_HPP_

#include "crocoddyl/core/guidance-base.hpp"
#include "crocoddyl/core/utils/conversions.hpp"

namespace crocoddyl {

/**
 * @brief Direction-preserving, smoothly bounded guidance model.
 *
 * This law is intended for Euclidean tasks with homogeneous units and
 * implements the smooth saturation law
 * \f[
 *   g(e) =
 *   -v_{\max}\tanh\left(\frac{k r_\epsilon}{v_{\max}}\right)
 *   \frac{e}{r_\epsilon}, \qquad
 *   r_\epsilon = \sqrt{e^\top e + \epsilon^2}.
 * \f]
 *
 * It is smooth at the origin, approaches \f$-k e\f$ locally for a
 * sufficiently small \f$\epsilon\f$, preserves direction, and bounds the
 * desired-rate norm.
 */
template <typename _Scalar>
class GuidanceModelSmoothSaturationTpl
    : public GuidanceModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(GuidanceModelBase, GuidanceModelSmoothSaturationTpl)

  typedef _Scalar Scalar;
  typedef GuidanceModelAbstractTpl<Scalar> Base;
  typedef typename Base::GuidanceDataAbstract GuidanceDataAbstract;
  typedef GuidanceDataSmoothSaturationTpl<Scalar> Data;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the smoothly saturated guidance model.
   *
   * @param[in] nr        Task dimension.
   * @param[in] gain      Local linear gain \f$k\f$.
   * @param[in] max_rate   Maximum desired-rate norm \f$v_{\max}\f$.
   * @param[in] epsilon   Smoothing parameter \f$\epsilon > 0\f$.
   */
  GuidanceModelSmoothSaturationTpl(const std::size_t nr, const Scalar& gain,
                                   const Scalar& max_rate,
                                   const Scalar epsilon = Scalar(1e-6));
  virtual ~GuidanceModelSmoothSaturationTpl() = default;

  /**
   * @brief Compute the smooth-saturation guidance output.
   *
   * @param[in,out] data   Preallocated guidance data.
   * @param[in] error      Task error \f$e\f$.
   */
  virtual void calc(const std::shared_ptr<GuidanceDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& error) const override;

  /**
   * @brief Compute the Jacobian of the smooth-saturation law.
   *
   * @param[in,out] data   Preallocated guidance data.
   * @param[in] error      Task error \f$e\f$.
   */
  virtual void calcDiff(const std::shared_ptr<GuidanceDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& error) const override;

  /**
   * @brief Allocate the cached guidance data.
   *
   * @return Data object storing \f$r_\epsilon\f$, the output scale, and the
   * cached saturation values.
   */
  virtual std::shared_ptr<GuidanceDataAbstract> createData() const override;

  /**
   * @brief Return the local linear gain.
   *
   * @return The gain \f$k\f$.
   */
  const Scalar& get_gain() const;

  /**
   * @brief Cast the guidance model to a different scalar type.
   */
  template <typename NewScalar>
  GuidanceModelSmoothSaturationTpl<NewScalar> cast() const;

  /**
   * @brief Return the maximum desired-rate norm.
   *
   * @return The saturation bound \f$v_{\max}\f$.
   */
  const Scalar& get_max_rate() const;

  /**
   * @brief Return the smoothing parameter.
   *
   * @return The smoothing value \f$\epsilon\f$.
   */
  const Scalar& get_epsilon() const;

  /**
   * @brief Update the local linear gain.
   *
   * @param[in] gain  New local linear gain \f$k \ge 0\f$.
   */
  void set_gain(const Scalar& gain);

  /**
   * @brief Update the maximum desired-rate norm.
   *
   * @param[in] max_rate  New saturation bound \f$v_{\max} > 0\f$.
   */
  void set_max_rate(const Scalar& max_rate);

  /**
   * @brief Update the smoothing parameter.
   *
   * @param[in] epsilon  New smoothing value \f$\epsilon > 0\f$.
   */
  void set_epsilon(const Scalar& epsilon);

  /**
   * @brief Print the model parameters.
   *
   * @param[out] os  Output stream.
   */
  virtual void print(std::ostream& os) const override;

 private:
  void checkParameters(const Scalar& gain, const Scalar& max_rate,
                       const Scalar& epsilon) const;

  Scalar gain_;
  Scalar max_rate_;
  Scalar epsilon_;

  using Base::checkErrorDimension;
  using Base::checkJacobianDimension;
  using Base::checkRateDimension;
  using Base::nr_;
};

/**
 * @brief Cached values for the smooth saturation guidance model.
 *
 * The cached values avoid recomputing the radial norm and saturation
 * arguments during calcDiff().
 */
template <typename _Scalar>
struct GuidanceDataSmoothSaturationTpl
    : public GuidanceDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef GuidanceDataAbstractTpl<Scalar> Base;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <typename Guidance>
  explicit GuidanceDataSmoothSaturationTpl(Guidance* const guidance)
      : Base(guidance),
        radius(Scalar(0)),
        scale(Scalar(0)),
        z(Scalar(0)),
        tanh_z(Scalar(0)) {}
  virtual ~GuidanceDataSmoothSaturationTpl() = default;

  Scalar radius;  //!< Cached radial term \f$r_\epsilon\f$.
  Scalar scale;   //!< Cached scalar scale \f$v_{\max}\tanh(z)/r_\epsilon\f$.
  Scalar z;       //!< Cached saturation argument \f$k r_\epsilon / v_{\max}\f$.
  Scalar tanh_z;  //!< Cached value of \f$\tanh(z)\f$.

  using Base::g;
  using Base::Ge;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/guidance/smooth-saturation.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::GuidanceModelSmoothSaturationTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::GuidanceDataSmoothSaturationTpl)

#endif  // CROCODDYL_CORE_GUIDANCE_SMOOTH_SATURATION_HPP_
