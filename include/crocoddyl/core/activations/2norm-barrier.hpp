///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021-2025, LAAS-CNRS, Airbus, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIVATIONS_2NORM_BARRIER_HPP_
#define CROCODDYL_CORE_ACTIVATIONS_2NORM_BARRIER_HPP_

#include "crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/fwd.hpp"

namespace crocoddyl {

/**
 * @brief 2-norm barrier activation
 *
 * This activation function describes a quadratic barrier of the 2-norm of a
 * residual vector, i.e.,
 * \f[
 * \Bigg\{\begin{aligned}
 * &\frac{1}{2} (d - \alpha)^2, &\textrm{if} \,\,\, d < \alpha \\
 * &0, &\textrm{otherwise},
 * \end{aligned}
 * \f]
 * where \f$d = \|r\|\f$ is the norm of the residual, \f$\alpha\f$ the threshold
 * distance from which the barrier is active, \f$nr\f$ is the dimension of the
 * residual vector.
 *
 * The computation of the function and it derivatives are carried out in
 * `calc()` and `calcDiff()`, respectively.
 *
 * \sa `ActivationModelAbstractTpl`, `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ActivationModel2NormBarrierTpl
    : public ActivationModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ActivationModelBase, ActivationModel2NormBarrierTpl)

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActivationModelAbstractTpl<Scalar> Base;
  typedef ActivationDataAbstractTpl<Scalar> ActivationDataAbstract;
  typedef ActivationData2NormBarrierTpl<Scalar> Data;
  typedef typename MathBase::VectorXs VectorXs;

  /**
   * @brief Initialize the 2-norm barrier activation model
   *
   * The default `alpha` value is defined as 0.1.
   *
   * @param[in] nr            Dimension of the residual vector
   * @param[in] alpha         Threshold factor (default 0.1)
   * @param[in] true_hessian  Boolean indicating whether to use the Gauss-Newton
   * approximation or true Hessian in computing the derivatives (default: false)
   */
  explicit ActivationModel2NormBarrierTpl(const std::size_t nr,
                                          const Scalar alpha = Scalar(0.1),
                                          const bool true_hessian = false)
      : Base(nr), alpha_(alpha), true_hessian_(true_hessian) {
    if (alpha < Scalar(0.)) {
      throw_pretty("Invalid argument: " << "alpha should be a positive value");
    }
  };
  virtual ~ActivationModel2NormBarrierTpl() = default;

  /**
   * @brief Compute the 2-norm barrier function
   *
   * @param[in] data  2-norm barrier activation data
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

    d->d = r.norm();
    if (d->d < alpha_) {
      data->a_value = Scalar(0.5) * (d->d - alpha_) * (d->d - alpha_);
    } else {
      data->a_value = Scalar(0.0);
    }
  };

  /**
   * @brief Compute the derivatives of the 2norm-barrier function
   *
   * @param[in] data  2-norm barrier activation data
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

    if (d->d < alpha_) {
      data->Ar = (d->d - alpha_) / d->d * r;
      if (true_hessian_) {
        data->Arr.diagonal() =
            alpha_ * r.array().square() / pow(d->d, Scalar(3));  // True Hessian
        data->Arr.diagonal().array() += (d->d - alpha_) / d->d;
      } else {
        data->Arr.diagonal() =
            r.array().square() /
            pow(d->d, Scalar(2));  // GN Hessian approximation
      }
    } else {
      data->Ar.setZero();
      data->Arr.setZero();
    }
  };

  /**
   * @brief Create the 2norm-barrier activation data
   *
   * @return the activation data
   */
  virtual std::shared_ptr<ActivationDataAbstract> createData() override {
    return std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
  };

  template <typename NewScalar>
  ActivationModel2NormBarrierTpl<NewScalar> cast() const {
    typedef ActivationModel2NormBarrierTpl<NewScalar> ReturnType;
    ReturnType res(nr_, scalar_cast<NewScalar>(alpha_), true_hessian_);
    return res;
  }

  /**
   * @brief Get and set the threshold factor
   */
  const Scalar& get_alpha() const { return alpha_; };
  void set_alpha(const Scalar& alpha) { alpha_ = alpha; };

  /**
   * @brief Print relevant information of the 2-norm barrier model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override {
    os << "ActivationModel2NormBarrier {nr=" << nr_ << ", alpha=" << alpha_
       << ", Hessian=" << true_hessian_ << "}";
  }

 protected:
  using Base::nr_;     //!< Dimension of the residual vector
  Scalar alpha_;       //!< Threshold factor
  bool true_hessian_;  //!< Use true Hessian in calcDiff if true, Gauss-Newton
                       //!< approximation if false
};

template <typename _Scalar>
struct ActivationData2NormBarrierTpl
    : public ActivationDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::DiagonalMatrixXs DiagonalMatrixXs;
  typedef ActivationDataAbstractTpl<Scalar> Base;

  template <typename Activation>
  explicit ActivationData2NormBarrierTpl(Activation* const activation)
      : Base(activation), d(Scalar(0)) {}
  virtual ~ActivationData2NormBarrierTpl() = default;

  Scalar d;  //!< Norm of the residual

  using Base::a_value;
  using Base::Ar;
  using Base::Arr;
};

}  // namespace crocoddyl

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(
    crocoddyl::ActivationModel2NormBarrierTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(
    crocoddyl::ActivationData2NormBarrierTpl)

#endif  // CROCODDYL_CORE_ACTIVATIONS_2NORM_BARRIER_HPP_
