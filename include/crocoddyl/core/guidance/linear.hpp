///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_GUIDANCE_LINEAR_HPP_
#define CROCODDYL_CORE_GUIDANCE_LINEAR_HPP_

#include "crocoddyl/core/guidance-base.hpp"

namespace crocoddyl {

/**
 * @brief Linear converging guidance model.
 *
 * It implements
 * \f[
 *   g(e) = -K e, \qquad G_e = -K,
 * \f]
 * where \f$K\f$ is a constant feedback-gain matrix.
 */
template <typename _Scalar>
class GuidanceModelLinearTpl : public GuidanceModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(GuidanceModelBase, GuidanceModelLinearTpl)

  typedef _Scalar Scalar;
  typedef GuidanceModelAbstractTpl<Scalar> Base;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename Base::GuidanceDataAbstract GuidanceDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the model from a full gain matrix.
   *
   * @param[in] gain  Square feedback-gain matrix \f$K\f$.
   */
  explicit GuidanceModelLinearTpl(const MatrixXs& gain);

  /**
   * @brief Initialize the model from a diagonal gain vector.
   *
   * @param[in] diagonal_gain  Diagonal entries of \f$K\f$.
   */
  explicit GuidanceModelLinearTpl(const VectorXs& diagonal_gain);

  /**
   * @brief Initialize the model from a scalar gain.
   *
   * @param[in] nr    Task dimension.
   * @param[in] gain  Scalar gain used on the identity matrix.
   */
  GuidanceModelLinearTpl(const std::size_t nr, const Scalar& gain);
  virtual ~GuidanceModelLinearTpl() = default;

  /**
   * @brief Compute \f$g = -K e\f$.
   *
   * @param[in,out] data   Preallocated guidance data.
   * @param[in] error      Task error \f$e\f$.
   */
  virtual void calc(const std::shared_ptr<GuidanceDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& error) const override;

  /**
   * @brief Compute \f$G_e = -K\f$.
   *
   * @param[in,out] data   Preallocated guidance data.
   * @param[in] error      Task error \f$e\f$.
   */
  virtual void calcDiff(const std::shared_ptr<GuidanceDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& error) const override;

  /**
   * @brief Return the feedback-gain matrix.
   *
   * @return The constant gain matrix \f$K\f$.
   */
  const MatrixXs& get_gain() const;

  /**
   * @brief Cast the guidance model to a different scalar type.
   */
  template <typename NewScalar>
  GuidanceModelLinearTpl<NewScalar> cast() const;

  /**
   * @brief Set the feedback-gain matrix.
   *
   * @param[in] gain  Square feedback-gain matrix \f$K\f$.
   */
  void set_gain(const MatrixXs& gain);

  /**
   * @brief Print the model parameters.
   *
   * @param[out] os  Output stream.
   */
  virtual void print(std::ostream& os) const override;

 private:
  MatrixXs gain_;

  using Base::checkErrorDimension;
  using Base::checkJacobianDimension;
  using Base::checkRateDimension;
  using Base::nr_;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/guidance/linear.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::GuidanceModelLinearTpl)

#endif  // CROCODDYL_CORE_GUIDANCE_LINEAR_HPP_
