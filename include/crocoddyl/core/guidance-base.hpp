///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_GUIDANCE_BASE_HPP_
#define CROCODDYL_CORE_GUIDANCE_BASE_HPP_

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/mathbase.hpp"

namespace crocoddyl {

class GuidanceModelBase {
 public:
  virtual ~GuidanceModelBase() = default;

  CROCODDYL_BASE_CAST(GuidanceModelBase, GuidanceModelAbstractTpl)
};

/**
 * @brief Differentiable mapping from a task error to a desired task rate.
 *
 * A guidance model implements
 * \f[
 *   g = \phi(e), \qquad
 *   G_e = \frac{\partial g}{\partial e}.
 * \f]
 *
 * The input and output dimensions are deliberately identical. Mixed-unit
 * tasks (for example, an SE(3) error) should use a block-structured model or a
 * task-specific metric instead of a single radial norm.
 *
 * Implementations must not allocate in calc() or calcDiff(). The caller owns
 * the preallocated data storage.
 *
 * The model exposes the generic relation
 * \f[
 *   g = \phi(e), \qquad G_e = \frac{\partial g}{\partial e}.
 * \f]
 */
template <typename _Scalar>
class GuidanceModelAbstractTpl : public GuidanceModelBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef GuidanceDataAbstractTpl<Scalar> GuidanceDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the guidance-model dimension.
   *
   * @param[in] nr  Dimension of the task error and desired task rate.
   */
  explicit GuidanceModelAbstractTpl(const std::size_t nr);
  virtual ~GuidanceModelAbstractTpl() = default;

  /**
   * @brief Allocate the precomputed guidance data.
   *
   * @return A zero-initialized data object sized from the model.
   */
  virtual std::shared_ptr<GuidanceDataAbstract> createData() const;

  /**
   * @brief Compute the desired task rate \f$g=\phi(e)\f$.
   *
   * @param[in,out] data   Preallocated guidance data.
   * @param[in] error      Task error \f$e\f$.
   */
  virtual void calc(const std::shared_ptr<GuidanceDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& error) const = 0;

  /**
   * @brief Compute \f$\partial g/\partial e\f$.
   *
   * @param[in,out] data   Preallocated guidance data.
   * @param[in] error      Task error \f$e\f$.
   */
  virtual void calcDiff(const std::shared_ptr<GuidanceDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& error) const = 0;

  /**
   * @brief Return the error and desired-rate dimension.
   *
   * @return Dimension of the task error and desired task rate.
   */
  std::size_t get_nr() const;

  /**
   * @brief Print relevant information about the guidance model.
   *
   * @param[out] os  Output stream.
   */
  virtual void print(std::ostream& os) const;

 protected:
  void checkErrorDimension(const Eigen::Ref<const VectorXs>& error) const;
  void checkRateDimension(const Eigen::Ref<const VectorXs>& g) const;
  void checkJacobianDimension(const Eigen::Ref<const MatrixXs>& Ge) const;

  std::size_t nr_;
};

template <typename Scalar>
std::ostream& operator<<(std::ostream& os,
                         const GuidanceModelAbstractTpl<Scalar>& model);

/**
 * @brief Preallocated values and derivatives produced by a guidance model.
 *
 * The notation matches the manuscript:
 * - \f$g\f$ is the desired task rate;
 * - \f$G_e\f$ is its Jacobian with respect to the task error.
 *
 * The storage is zero-initialized by the constructor and reused by calc() and
 * calcDiff() to avoid repeated allocations.
 */
template <typename _Scalar>
struct GuidanceDataAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef GuidanceModelAbstractTpl<Scalar> GuidanceModelAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Allocate guidance data with the size inferred from the model.
   *
   * @param[in] model  Guidance model used to size the storage.
   */
  explicit GuidanceDataAbstractTpl(const GuidanceModelAbstract* const model)
      : g(model->get_nr()), Ge(model->get_nr(), model->get_nr()) {
    g.setZero();
    Ge.setZero();
  }
  virtual ~GuidanceDataAbstractTpl() = default;

  VectorXs g;
  MatrixXs Ge;
};

}  // namespace crocoddyl

/* --- Details -------------------------------------------------------------- */
#include "crocoddyl/core/guidance-base.hxx"

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::GuidanceModelAbstractTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::GuidanceDataAbstractTpl)

#endif  // CROCODDYL_CORE_GUIDANCE_BASE_HPP_
