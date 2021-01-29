///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_FLAT_LOG_HPP_
#define CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_FLAT_LOG_HPP_

#include "crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief Quadratic-flat-log activation
 *
 * This activation function describes a logarithmic quadratic activation
 * depending on the quadratic norm of a residual vector, i.e. \f[
 * \begin{equation} log(1 + \|\mathbf{r}\|^2 / \alpha) \end{equation} \f] where
 * \f$\alpha\f$ defines the width of the quadratic basin, \f$r\f$ is the scalar
 * residual, \f$nr\f$ is the dimension of the residual vector.
 *
 * The computation of the function and it derivatives are carried out in
 * `calc()` and `caldDiff()`, respectively.
 *
 * \sa `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ActivationModelQuadFlatLogTpl
    : public ActivationModelAbstractTpl<_Scalar> {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActivationModelAbstractTpl<Scalar> Base;
  typedef ActivationDataAbstractTpl<Scalar> ActivationDataAbstract;
  typedef ActivationDataQuadFlatLogTpl<Scalar> Data;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /*
   * @brief Initialize the quadratic-flat-log activation model
   *
   * The default `alpha` value is defined as 1.
   *
   * @param[in] nr   Dimension of the residual vector
   * @param[in] alpha  Width of quadratic basin (default: 1.)
   */

  explicit ActivationModelQuadFlatLogTpl(const std::size_t &nr,
                                         const Scalar &alpha = Scalar(1.))
      : Base(nr), alpha_(alpha) {
    if (alpha < Scalar(0.)) {
      throw_pretty("Invalid argument: "
                   << "alpha should be a positive value");
    }
  };
  virtual ~ActivationModelQuadFlatLogTpl(){};

  /*
   * @brief Compute the quadratic-flat-log function
   *
   * @param[in] data  Quadratic-log activation data
   * @param[in] r     Residual vector \f$\mathbf{r}\in\mathbb{R}^{nr}\f$
   */
  virtual void calc(const boost::shared_ptr<ActivationDataAbstract> &data,
                    const Eigen::Ref<const VectorXs> &r) {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty("Invalid argument: "
                   << "r has wrong dimension (it should be " +
                          std::to_string(nr_) + ")");
    }
    boost::shared_ptr<Data> d = boost::static_pointer_cast<Data>(data);
    d->a0 = r.squaredNorm() / alpha_;
    data->a_value = log(Scalar(1.0) + d->a0);
  };

  /*
   * @brief Compute the derivatives of the quadratic-flat-log function
   *
   * @param[in] data  Quadratic-log activation data
   * @param[in] r     Residual vector \f$\mathbf{r}\in\mathbb{R}^{nr}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ActivationDataAbstract> &data,
                        const Eigen::Ref<const VectorXs> &r) {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty("Invalid argument: "
                   << "r has wrong dimension (it should be " +
                          std::to_string(nr_) + ")");
    }
    boost::shared_ptr<Data> d = boost::static_pointer_cast<Data>(data);

    d->a1 = Scalar(2.0) / (alpha_ + alpha_ * d->a0);
    data->Ar = d->a1 * r;
    data->Arr.diagonal() = -d->a1 * d->a1 * r.array().square();
    data->Arr.diagonal().array() += d->a1;
  };

  /*
   * @brief Create the quadratic-flat-log activation data
   *
   * @return the activation data
   */
  virtual boost::shared_ptr<ActivationDataAbstract> createData() {
    boost::shared_ptr<Data> data =
        boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
    return data;
  };

  Scalar get_alpha() const { return alpha_; };
  void set_alpha(const Scalar alpha) { alpha_ = alpha; };

protected:
  using Base::nr_; //!< Dimension of the residual vector

private:
  Scalar alpha_; //!< Width of quadratic basin
};

/*
 * @brief Data structure of the quadratic-flat-log activation
 *
 * @param[in] a0  computed in calc to avoid recomputation
 * @param[in] a1  computed in calcDiff to avoid recomputation
 */
template <typename _Scalar>
struct ActivationDataQuadFlatLogTpl
    : public ActivationDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActivationDataAbstractTpl<Scalar> Base;

  template <typename Activation>
  explicit ActivationDataQuadFlatLogTpl(Activation *const activation)
      : Base(activation), a0(0), a1(0) {}

  Scalar a0;
  Scalar a1;
};

} // namespace crocoddyl

#endif // CROCODDYL_CORE_ACTIVATIONS_QUADRATIC_FLAT_LOG_HPP_
