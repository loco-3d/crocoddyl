///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2025, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef COLMPC_ACTIVATIONS_QUADRATIC_EXP_HPP_
#define COLMPC_ACTIVATIONS_QUADRATIC_EXP_HPP_

#include "colmpc/fwd.hpp"
#include "crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace colmpc {
using namespace crocoddyl;

/*
 * @brief Quadratic-exp activation
 *
 * This activation function describes a quadratic exponential activation
 * depending on the square norm of a residual vector, i.e. \f[ \begin{equation}
 * \sum_i exp(- \left(\frac{r_i}{\alpha}\right)^N) \end{equation} \f] where
 * \f$\alpha\f$ defines the width of the quadratic basin, \f$r\f$ is the
 * residual vector,
 * \f$nr\f$ is the dimension of the residual vector.
 *
 * The computation of the function and it derivatives are carried out in
 * `calc()` and `calcDiff()`, respectively.
 *
 * \sa `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar, int N>
class ActivationModelExpTpl : public ActivationModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActivationModelAbstractTpl<Scalar> Base;
  typedef ActivationDataAbstractTpl<Scalar> ActivationDataAbstract;
  typedef ActivationDataExpTpl<Scalar, N> Data;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /*
   * @brief Initialize the quadratic-exp activation model
   *
   * The default `alpha` value is defined as 1.
   *
   * @param[in] nr     Dimension of the residual vector
   * @param[in] alpha  Width of quadratic basin (default: 1.)
   */

  explicit ActivationModelExpTpl(const std::size_t& nr,
                                 const Scalar& alpha = Scalar(1.))
      : Base(nr), alpha_(alpha) {
    static_assert(N > 0 && N < 3,
                  "N should be strictly positive. Value of 3 and above have "
                  "not been tested.");
    if (alpha <= Scalar(0.)) {
      throw_pretty(
          "Invalid argument: " << "alpha should be a strictly positive value");
    }
  };
  template <typename OtherScalar>
  explicit ActivationModelExpTpl(
      const ActivationModelExpTpl<OtherScalar, N>& other)
      : crocoddyl::ActivationModelAbstractTpl<_Scalar>(other.get_nr()) {}

  virtual ~ActivationModelExpTpl() {};

  /*
   * @brief Compute the quadratic-exp function
   *
   * @param[in] data  Quadratic activation data
   * @param[in] r     Residual vector \f$\mathbf{r}\in\mathbb{R}^{nr}\f$
   */
  virtual void calc(const std::shared_ptr<ActivationDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& r) {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty(
          "Invalid argument: " << "r has wrong dimension (it should be " +
                                      std::to_string(nr_) + ")");
    }
    std::shared_ptr<Data> d = std::static_pointer_cast<Data>(data);

    d->v = r.array() / alpha_;
    switch (N) {
      case 1:
        d->exp_minus_vn = (-d->v).exp();
        break;
      case 2:
        d->vn = d->v.square();
        d->exp_minus_vn = (-d->vn).exp();
        break;
      default:
        d->vn = d->v.pow(N);
        d->exp_minus_vn = (-d->vn).exp();
        break;
    }

    data->a_value = d->exp_minus_vn.sum();
  };

  /*
   * @brief Compute the derivatives of the quadratic-exp function
   *
   * @param[in] data  Quadratic activation data
   * @param[in] r     Residual vector \f$\mathbf{r}\in\mathbb{R}^{nr}\f$
   */
  virtual void calcDiff(const std::shared_ptr<ActivationDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& r) {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty(
          "Invalid argument: " << "r has wrong dimension (it should be " +
                                      std::to_string(nr_) + ")");
    }
    std::shared_ptr<Data> d = std::static_pointer_cast<Data>(data);

    switch (N) {
      case 1:
        data->Ar = -d->exp_minus_vn / alpha_;
        data->Arr.diagonal() = -data->Ar / alpha_;
        break;
      case 2:
        data->Ar = Scalar(-2.0) / alpha_ * d->v * d->exp_minus_vn;
        data->Arr.diagonal() = Scalar(2.0) / (alpha_ * alpha_) *
                               (Scalar(2.0) * d->vn - Scalar(1.0)) *
                               d->exp_minus_vn;
      default:
        // TODO this code is not optimized. It is also likely not used.
        Eigen::Array<Scalar, Eigen::Dynamic, 1> vn2 = d->v.pow(N - 2);
        data->Ar = Scalar(-N) / alpha_ * vn2 * d->v * d->exp_minus_vn;
        data->Arr.diagonal() = Scalar(N) / (alpha_ * alpha_) * vn2 *
                               (Scalar(N) * d->vn - Scalar(N - 1)) *
                               d->exp_minus_vn;
        break;
    }
  };

  // Explicit template instanciation
  std::shared_ptr<crocoddyl::ActivationModelBase> cloneAsDouble()
      const override {
    return std::make_shared<ActivationModelExpTpl<double, N>>(*this);
  }

  // Explicit template instanciation
  std::shared_ptr<crocoddyl::ActivationModelBase> cloneAsFloat()
      const override {
    return std::make_shared<ActivationModelExpTpl<float, N>>(*this);
  }

  /**
   * @brief Create the quadratic-exp activation data
   *
   * @return the activation data
   */
  virtual std::shared_ptr<ActivationDataAbstract> createData() {
    std::shared_ptr<Data> data =
        std::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
    return data;
  };

  Scalar get_alpha() const { return alpha_; };
  void set_alpha(const Scalar alpha) { alpha_ = alpha; };

  /**
   * @brief Print relevant information of the quadratic-exp model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const {
    os << "ActivationModelExp<" << N << "> {nr=" << nr_ << ", a=" << alpha_
       << "}";
  }

 protected:
  using Base::nr_;  //!< Dimension of the residual vector

 private:
  Scalar alpha_;  //!< Width of quadratic basin
};

/*
 * @brief Data structure of the quadratic-exp activation
 *
 * @param[in] a0  computed in calc to avoid recomputation
 * @param[in] a1  computed in calcDiff to avoid recomputation
 */
template <typename _Scalar, int N>
struct ActivationDataExpTpl : public ActivationDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActivationDataAbstractTpl<Scalar> Base;

  // Constructor: accept the concrete ActivationModelExpTpl<Scalar, N>*.
  explicit ActivationDataExpTpl(
      ActivationModelExpTpl<_Scalar, N>* const activation)
      : Base(static_cast<crocoddyl::ActivationModelAbstractTpl<_Scalar>*>(
            activation)),
        v(activation->get_nr()),
        vn(activation->get_nr()),
        exp_minus_vn(activation->get_nr()) {}

  Eigen::Array<Scalar, Eigen::Dynamic, 1> v;   // Residual divided by alpha
  Eigen::Array<Scalar, Eigen::Dynamic, 1> vn;  // Residual divided by alpha
  Eigen::Array<Scalar, Eigen::Dynamic, 1> exp_minus_vn;
};

}  // namespace colmpc

#endif  // MPC_ACTIVATIONS_QUADRATIC_EXP_HPP_
