///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIVATIONS_WEIGHTED_SMOOTH_1NORM_HPP_
#define CROCODDYL_CORE_ACTIVATIONS_WEIGHTED_SMOOTH_1NORM_HPP_

#include "crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/fwd.hpp"

namespace crocoddyl {

/**
 * @brief Weighted smooth-1norm activation
 *
 * This activation function describes a weighted smooth representation of the
 * 1-norm of a residual vector:
 * \f[
 *   a(\mathbf{r}) = \sum_{i=0}^{nr-1} w_i\sqrt{\epsilon + r_i^2},
 * \f]
 * where \f$\epsilon > 0\f$ is the smoothing factor, \f$w_i \geq 0\f$ is the
 * weight associated with the scalar residual \f$r_i\f$, and \f$nr\f$ is the
 * residual dimension. This activation represents a shifted and weighted
 * Charbonnier function.
 *
 * The weights scale the activation without changing the smoothing transition,
 * which occurs around \f$|r_i| = \sqrt{\epsilon}\f$.
 *
 * \sa `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ActivationModelWeightedSmooth1NormTpl
    : public ActivationModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ActivationModelBase,
                         ActivationModelWeightedSmooth1NormTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActivationModelAbstractTpl<Scalar> Base;
  typedef ActivationDataAbstractTpl<Scalar> ActivationDataAbstract;
  typedef ActivationDataWeightedSmooth1NormTpl<Scalar> Data;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize the weighted smooth-1norm activation model
   *
   * @param[in] weights  Nonnegative residual weights
   * @param[in] eps      Strictly positive smoothing factor (default: 1)
   */
  explicit ActivationModelWeightedSmooth1NormTpl(const VectorXs& weights,
                                                 const Scalar eps = Scalar(1.))
      : Base(weights.size()), weights_(weights), eps_(eps) {
    check_weights(weights_);
    if (eps_ <= Scalar(0.)) {
      throw_pretty("Invalid argument: eps should be a strictly positive value");
    }
  }
  virtual ~ActivationModelWeightedSmooth1NormTpl() = default;

  /**
   * @brief Compute the weighted smooth-1norm activation
   *
   * @param[in] data  Weighted smooth-1norm activation data
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

    d->a = (r.array().square() + eps_).sqrt().matrix();
    data->a_value = weights_.dot(d->a);
  }

  /**
   * @brief Compute the activation derivatives
   *
   * @param[in] data  Weighted smooth-1norm activation data
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
    data->Ar = weights_.cwiseProduct(r).cwiseProduct(a_inv);
    data->Arr.diagonal() =
        eps_ *
        weights_.cwiseProduct(a_inv.cwiseProduct(a_inv).cwiseProduct(a_inv));
  }

  /**
   * @brief Create the weighted smooth-1norm activation data
   *
   * @return the activation data
   */
  virtual std::shared_ptr<ActivationDataAbstract> createData() override {
    return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
  }

  template <typename NewScalar>
  ActivationModelWeightedSmooth1NormTpl<NewScalar> cast() const {
    typedef ActivationModelWeightedSmooth1NormTpl<NewScalar> ReturnType;
    return ReturnType(weights_.template cast<NewScalar>(),
                      scalar_cast<NewScalar>(eps_));
  }

  const VectorXs& get_weights() const { return weights_; }
  Scalar get_eps() const { return eps_; }

  void set_weights(const VectorXs& weights) {
    if (weights.size() != weights_.size()) {
      throw_pretty("Invalid argument: "
                   << "weight vector has wrong dimension (it should be " +
                          std::to_string(weights_.size()) + ")");
    }
    check_weights(weights);
    weights_ = weights;
  }

  /**
   * @brief Print relevant information of the weighted smooth-1norm model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override {
    os << "ActivationModelWeightedSmooth1Norm {nr=" << nr_ << ", eps=" << eps_
       << "}";
  }

 protected:
  using Base::nr_;  //!< Dimension of the residual vector

 private:
  static void check_weights(const VectorXs& weights) {
    for (Eigen::Index i = 0; i < weights.size(); ++i) {
      if (weights[i] < Scalar(0.)) {
        throw_pretty(
            "Invalid argument: weights should contain nonnegative values");
      }
    }
  }

  VectorXs weights_;  //!< Residual weights
  Scalar eps_;        //!< Smoothing factor
};

template <typename _Scalar>
struct ActivationDataWeightedSmooth1NormTpl
    : public ActivationDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ActivationDataAbstractTpl<Scalar> Base;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;

  template <typename Activation>
  explicit ActivationDataWeightedSmooth1NormTpl(Activation* const activation)
      : Base(activation), a(VectorXs::Zero(activation->get_nr())) {}
  virtual ~ActivationDataWeightedSmooth1NormTpl() = default;

  VectorXs a;  //!< Element-wise values \f$\sqrt{\epsilon + r_i^2}\f$

  using Base::a_value;
  using Base::Ar;
  using Base::Arr;
};

}  // namespace crocoddyl

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::ActivationModelWeightedSmooth1NormTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::ActivationDataWeightedSmooth1NormTpl)

#endif  // CROCODDYL_CORE_ACTIVATIONS_WEIGHTED_SMOOTH_1NORM_HPP_
