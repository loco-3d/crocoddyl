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
 * residual vector, \f$nr\f$ is the dimension of the residual vector.
 *
 * The computation of the function and it derivatives are carried out in
 * `calc()` and `calcDiff()`, respectively.
 *
 * \sa `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ActivationModelDistanceQuadTpl
    : public ActivationModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActivationModelAbstractTpl<Scalar> Base;
  typedef ActivationDataAbstractTpl<Scalar> ActivationDataAbstract;
  typedef ActivationDataDistanceQuadTpl<Scalar> Data;
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

  explicit ActivationModelDistanceQuadTpl(const std::size_t& nr,
                                          const Scalar& d0 = Scalar(1.))
      : Base(nr) {
    if (d0 <= Scalar(0.)) {
      throw_pretty(
          "Invalid argument: " << "alpha should be a strictly positive value");
    }
    set_d0(d0);
  };
  virtual ~ActivationModelDistanceQuadTpl() {};

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

    d->dd = r * d0inv_;
    d->one_minus_dd = 1 - d->dd;
    data->a_value = d->one_minus_dd.cwiseMax(0).square().sum();
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
    typedef Eigen::Array<Scalar, Eigen::Dynamic, 1> Array;

    data->Ar = Scalar(-2.0) * d0inv_ * d->one_minus_dd.cwiseMax(0);
    data->Arr.diagonal() =
        (d->one_minus_dd > 0)
            .select(Array::Constant(nr_, Scalar(2.0) * d0inv_ * d0inv_),
                    Array::Zero(nr_));
  };

  // Explicit template instanciation
  std::shared_ptr<crocoddyl::ActivationModelBase> cloneAsDouble()
      const override {
    return std::make_shared<ActivationModelDistanceQuadTpl<double>>(
        this->get_nr(), this->get_d0());
  }

  // Explicit template instanciation
  std::shared_ptr<crocoddyl::ActivationModelBase> cloneAsFloat()
      const override {
    return std::make_shared<ActivationModelDistanceQuadTpl<float>>(
        this->get_nr(), this->get_d0());
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

  Scalar get_d0() const { return d0_; };
  void set_d0(const Scalar d0) {
    d0_ = d0;
    d0inv_ = 1 / d0;
  };

  /**
   * @brief Print relevant information of the quadratic-exp model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const {
    os << "ActivationModelDistanceQuad {nr=" << nr_ << ", d0=" << d0_ << "}";
  }

 protected:
  using Base::nr_;  //!< Dimension of the residual vector

 private:
  Scalar d0_, d0inv_;  //!< Width of quadratic basin
};

/*
 * @brief Data structure of the quadratic-exp activation
 *
 * @param[in] a0  computed in calc to avoid recomputation
 * @param[in] a1  computed in calcDiff to avoid recomputation
 */
template <typename _Scalar>
struct ActivationDataDistanceQuadTpl
    : public ActivationDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActivationDataAbstractTpl<Scalar> Base;

  template <typename Activation>
  explicit ActivationDataDistanceQuadTpl(Activation* const activation)
      : Base(activation),
        dd(activation->get_nr()),
        one_minus_dd(activation->get_nr()) {}

  Eigen::Array<Scalar, Eigen::Dynamic, 1> dd;  // Residual divided by d0
  Eigen::Array<Scalar, Eigen::Dynamic, 1> one_minus_dd;
};

}  // namespace colmpc

#endif  // MPC_ACTIVATIONS_QUADRATIC_EXP_HPP_
