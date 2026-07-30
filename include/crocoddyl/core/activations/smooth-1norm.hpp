///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2026, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIVATIONS_SMOOTH_1NORM_HPP_
#define CROCODDYL_CORE_ACTIVATIONS_SMOOTH_1NORM_HPP_

#include "crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/fwd.hpp"

namespace crocoddyl {

/**
 * @brief Smooth-abs activation
 *
 * This activation function describes a smooth representation of the 1-norm of
 * a residual vector:
 * \f[
 *   a(\mathbf{r}) =
 *     \sum_{i=0}^{nr-1}
 *       \delta^2\left(\sqrt{1 + (r_i/\delta)^2} - 1\right).
 * \f]
 * Here, \f$\delta > 0\f$ is the smoothing scale, \f$r_i\f$ is a scalar
 * residual, and \f$nr\f$ is the residual dimension. This is the classical
 * pseudo-Huber form, equivalently
 * \f$\delta\sqrt{\delta^2 + r_i^2} - \delta^2\f$. Its local quadratic
 * approximation is \f$r_i^2/2\f$ for every \f$\delta\f$. The transition to the
 * linear regime occurs around \f$|r_i| = \delta\f$, with asymptotic slope
 * \f$\delta\f$.
 *
 * The computation of the function and its derivatives are carried out in
 * `calc()` and `calcDiff()`, respectively.
 *
 * \sa `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ActivationModelSmooth1NormTpl
    : public ActivationModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ActivationModelBase, ActivationModelSmooth1NormTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActivationModelAbstractTpl<Scalar> Base;
  typedef ActivationDataAbstractTpl<Scalar> ActivationDataAbstract;
  typedef ActivationDataSmooth1NormTpl<Scalar> Data;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the smooth-abs activation model
   *
   * The default `delta` value is defined as 1.
   *
   * @param[in] nr     Dimension of the residual vector
   * @param[in] delta  Strictly positive smoothing scale (default: 1)
   */
  explicit ActivationModelSmooth1NormTpl(const std::size_t nr,
                                         const Scalar delta = Scalar(1.))
      : Base(nr), delta_(delta) {
    if (delta_ <= Scalar(0.)) {
      throw_pretty(
          "Invalid argument: delta should be a strictly positive value");
    }
  };
  virtual ~ActivationModelSmooth1NormTpl() = default;

  /**
   * @brief Compute the smooth-abs function
   *
   * @param[in] data  Smooth-abs activation data
   * @param[in] r     Residual vector \f$\mathbf{r}\in\mathbb{R}^{nr}\f$
   */
  virtual void calc(const std::shared_ptr<ActivationDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& r) override {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty(
          "Invalid argument: " << "r has wrong dimension (it should be " +
                                      std::to_string(nr_) + ")");
    }
    std::shared_ptr<Data> d = std::static_pointer_cast<Data>(data);

    d->a = (r.array().square() + delta_ * delta_).sqrt().matrix();
    data->a_value =
        delta_ * (r.array().square() / (d->a.array() + delta_)).sum();
  };

  /**
   * @brief Compute the derivatives of the smooth-abs function
   *
   * @param[in] data  Smooth-abs activation data
   * @param[in] r     Residual vector \f$\mathbf{r}\in\mathbb{R}^{nr}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ActivationDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& r) override {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty(
          "Invalid argument: " << "r has wrong dimension (it should be " +
                                      std::to_string(nr_) + ")");
    }

    std::shared_ptr<Data> d = std::static_pointer_cast<Data>(data);
    const VectorXs a_inv = d->a.cwiseInverse();

    data->Ar = delta_ * r.cwiseProduct(a_inv);
    data->Arr.diagonal() = delta_ * delta_ * delta_ *
                           a_inv.cwiseProduct(a_inv).cwiseProduct(a_inv);
  };

  /**
   * @brief Create the smooth-abs activation data
   *
   * @return the activation data
   */
  virtual std::shared_ptr<ActivationDataAbstract> createData() override {
    return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
  };

  template <typename NewScalar>
  ActivationModelSmooth1NormTpl<NewScalar> cast() const {
    typedef ActivationModelSmooth1NormTpl<NewScalar> ReturnType;
    ReturnType res(nr_, scalar_cast<NewScalar>(delta_));
    return res;
  }

  Scalar get_delta() const { return delta_; }

  /**
   * @brief Print relevant information of the smooth-1norm model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override {
    os << "ActivationModelSmooth1Norm {nr=" << nr_ << ", delta=" << delta_
       << "}";
  }

 protected:
  using Base::nr_;  //!< Dimension of the residual vector
  Scalar delta_;    //!< Smoothing scale
};

template <typename _Scalar>
struct ActivationDataSmooth1NormTpl
    : public ActivationDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ActivationDataAbstractTpl<Scalar> Base;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::DiagonalMatrixXs DiagonalMatrixXs;

  template <typename Activation>
  explicit ActivationDataSmooth1NormTpl(Activation* const activation)
      : Base(activation), a(VectorXs::Zero(activation->get_nr())) {}
  virtual ~ActivationDataSmooth1NormTpl() = default;

  VectorXs a;

  using Base::a_value;
  using Base::Ar;
  using Base::Arr;
};

}  // namespace crocoddyl

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::ActivationModelSmooth1NormTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::ActivationDataSmooth1NormTpl)

#endif  // CROCODDYL_CORE_ACTIVATIONS_SMOOTH_1NORM_HPP_
