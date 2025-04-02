///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
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
 * This activation function describes a smooth representation of an absolute
 * activation (1-norm) for each element of a residual vector, i.e. \f[
 * \begin{equation} sum^nr_{i=0} \sqrt{\epsilon + \|r_i\|^2} \end{equation} \f]
 * where \f$\epsilon\f$ defines the smoothing factor, \f$r_i\f$ is the scalar
 * residual for the \f$i\f$ constraints, \f$nr\f$ is the dimension of the
 * residual vector.
 *
 * The computation of the function and it derivatives are carried out in
 * `calc()` and `caldDiff()`, respectively.
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
   * The default `eps` value is defined as 1.
   *
   * @param[in] nr   Dimension of the residual vector
   * @param[in] eps  Smoothing factor (default: 1.)
   */
  explicit ActivationModelSmooth1NormTpl(const std::size_t nr,
                                         const Scalar eps = Scalar(1.))
      : Base(nr), eps_(eps) {
    if (eps < Scalar(0.)) {
      throw_pretty("Invalid argument: " << "eps should be a positive value");
    }
    if (eps == Scalar(0.)) {
      std::cerr << "Warning: eps=0 leads to derivatives discontinuities in the "
                   "origin, it becomes the absolute function"
                << std::endl;
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

    d->a = (r.array().cwiseAbs2().array() + eps_).array().cwiseSqrt();
    data->a_value = d->a.sum();
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
    data->Ar = r.cwiseProduct(d->a.cwiseInverse());
    data->Arr.diagonal() =
        d->a.cwiseProduct(d->a).cwiseProduct(d->a).cwiseInverse();
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
    ReturnType res(nr_, scalar_cast<NewScalar>(eps_));
    return res;
  }

  /**
   * @brief Print relevant information of the smooth-1norm model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const override {
    os << "ActivationModelSmooth1Norm {nr=" << nr_ << ", eps=" << eps_ << "}";
  }

 protected:
  using Base::nr_;  //!< Dimension of the residual vector
  Scalar eps_;      //!< Smoothing factor
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
