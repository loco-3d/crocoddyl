///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_GUIDANCE_COMPONENTWISE_SATURATION_HPP_
#define CROCODDYL_CORE_GUIDANCE_COMPONENTWISE_SATURATION_HPP_

#include "crocoddyl/core/guidance-base.hpp"

namespace crocoddyl {

/**
 * @brief Independently bounded guidance model for heterogeneous task channels.
 *
 * For each component, it implements
 * \f[
 *   g_i =
 *   -v_{\max,i}\tanh\left(\frac{k_i e_i}{v_{\max,i}}\right).
 * \f]
 *
 * The Jacobian is diagonal, with
 * \f[
 *   G_e(i,i) = -k_i \left(1 - \tanh^2(z_i)\right), \qquad
 *   z_i = \frac{k_i e_i}{v_{\max,i}}.
 * \f]
 *
 * This is useful when channels have different units or physical limits, such
 * as translation and rotation blocks.
 */
template <typename _Scalar>
class GuidanceModelComponentwiseSaturationTpl
    : public GuidanceModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(GuidanceModelBase,
                         GuidanceModelComponentwiseSaturationTpl)

  typedef _Scalar Scalar;
  typedef GuidanceModelAbstractTpl<Scalar> Base;
  typedef typename Base::GuidanceDataAbstract GuidanceDataAbstract;
  typedef GuidanceDataComponentwiseSaturationTpl<Scalar> Data;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the componentwise saturation guidance model.
   *
   * @param[in] gain      Nonnegative componentwise gains \f$k_i\f$.
   * @param[in] max_rate  Positive componentwise saturation bounds
   *                      \f$v_{\max,i}\f$.
   */
  GuidanceModelComponentwiseSaturationTpl(const VectorXs& gain,
                                          const VectorXs& max_rate);
  virtual ~GuidanceModelComponentwiseSaturationTpl() = default;

  /**
   * @brief Compute the saturated guidance output.
   *
   * @param[in,out] data   Preallocated guidance data.
   * @param[in] error      Task error \f$e\f$.
   */
  virtual void calc(const std::shared_ptr<GuidanceDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& error) const override;

  /**
   * @brief Compute the Jacobian of the saturated guidance law.
   *
   * @param[in,out] data   Preallocated guidance data.
   * @param[in] error      Task error \f$e\f$.
   */
  virtual void calcDiff(const std::shared_ptr<GuidanceDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& error) const override;

  /**
   * @brief Allocate the cached guidance data.
   *
   * @return Data object storing the componentwise saturation arguments.
   */
  virtual std::shared_ptr<GuidanceDataAbstract> createData() const override;

  /**
   * @brief Return the componentwise gains.
   *
   * @return The vector of nonnegative gains \f$k_i\f$.
   */
  const VectorXs& get_gain() const;

  /**
   * @brief Cast the guidance model to a different scalar type.
   */
  template <typename NewScalar>
  GuidanceModelComponentwiseSaturationTpl<NewScalar> cast() const;

  /**
   * @brief Return the componentwise saturation bounds.
   *
   * @return The vector of positive bounds \f$v_{\max,i}\f$.
   */
  const VectorXs& get_max_rate() const;

  /**
   * @brief Update the componentwise gains.
   *
   * @param[in] gain  Nonnegative componentwise gains.
   */
  void set_gain(const VectorXs& gain);

  /**
   * @brief Update the componentwise saturation bounds.
   *
   * @param[in] max_rate  Positive componentwise bounds.
   */
  void set_max_rate(const VectorXs& max_rate);

  /**
   * @brief Print the model parameters.
   *
   * @param[out] os  Output stream.
   */
  virtual void print(std::ostream& os) const override;

 private:
  void checkParameters(const VectorXs& gain, const VectorXs& max_rate) const;

  VectorXs gain_;
  VectorXs max_rate_;

  using Base::checkErrorDimension;
  using Base::checkJacobianDimension;
  using Base::checkRateDimension;
  using Base::nr_;
};

/**
 * @brief Cached values for the componentwise saturation guidance model.
 *
 * The cached values avoid recomputing the per-component saturation arguments
 * and hyperbolic tangent values in calcDiff().
 */
template <typename _Scalar>
struct GuidanceDataComponentwiseSaturationTpl
    : public GuidanceDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef GuidanceDataAbstractTpl<Scalar> Base;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <typename Guidance>
  explicit GuidanceDataComponentwiseSaturationTpl(Guidance* const guidance)
      : Base(guidance),
        z(VectorXs::Zero(guidance->get_nr())),
        tanh_z(VectorXs::Zero(guidance->get_nr())) {}
  virtual ~GuidanceDataComponentwiseSaturationTpl() = default;

  VectorXs z;       //!< Cached componentwise saturation arguments.
  VectorXs tanh_z;  //!< Cached componentwise tanh values.

  using Base::g;
  using Base::Ge;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/guidance/componentwise-saturation.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::GuidanceModelComponentwiseSaturationTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::GuidanceDataComponentwiseSaturationTpl)

#endif  // CROCODDYL_CORE_GUIDANCE_COMPONENTWISE_SATURATION_HPP_
