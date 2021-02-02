///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIVATIONS_SMOOTH_1NORM_HPP_
#define CROCODDYL_CORE_ACTIVATIONS_SMOOTH_1NORM_HPP_

#include <iostream>
#include <stdexcept>

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief Smooth-abs activation
 *
 * This activation function describes a smooth representation of an absolute activation (1-norm) for each element of a
 * residual vector, i.e. \f[ \begin{equation} sum^nr_{i=0} \sqrt{\epsilon + \|r_i\|^2} \end{equation} \f] where
 * \f$\epsilon\f$ defines the smoothing factor, \f$r_i\f$ is the scalar residual for the \f$i\f$ constraints, \f$nr\f$
 * is the dimension of the residual vector.
 *
 * The computation of the function and it derivatives are carried out in `calc()` and `caldDiff()`, respectively.
 *
 * \sa `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ActivationModelSmooth1NormTpl : public ActivationModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

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
  explicit ActivationModelSmooth1NormTpl(const std::size_t nr, const Scalar eps = Scalar(1.)) : Base(nr), eps_(eps) {
    if (eps < Scalar(0.)) {
      throw_pretty("Invalid argument: "
                   << "eps should be a positive value");
    }
    if (eps == Scalar(0.)) {
      std::cerr
          << "Warning: eps=0 leads to derivatives discontinuities in the origin, it becomes the absolute function"
          << std::endl;
    }
  };
  virtual ~ActivationModelSmooth1NormTpl(){};

  /**
   * @brief Compute the smooth-abs function
   *
   * @param[in] data  Smooth-abs activation data
   * @param[in] r     Residual vector \f$\mathbf{r}\in\mathbb{R}^{nr}\f$
   */
  virtual void calc(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const VectorXs>& r) {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty("Invalid argument: "
                   << "r has wrong dimension (it should be " + std::to_string(nr_) + ")");
    }
    boost::shared_ptr<Data> d = boost::static_pointer_cast<Data>(data);

    d->a = (r.array().cwiseAbs2().array() + eps_).array().cwiseSqrt();
    data->a_value = d->a.sum();
  };

  /**
   * @brief Compute the derivatives of the smooth-abs function
   *
   * @param[in] data  Smooth-abs activation data
   * @param[in] r     Residual vector \f$\mathbf{r}\in\mathbb{R}^{nr}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const VectorXs>& r) {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty("Invalid argument: "
                   << "r has wrong dimension (it should be " + std::to_string(nr_) + ")");
    }

    boost::shared_ptr<Data> d = boost::static_pointer_cast<Data>(data);
    data->Ar = r.cwiseProduct(d->a.cwiseInverse());
    data->Arr.diagonal() = d->a.cwiseProduct(d->a).cwiseProduct(d->a).cwiseInverse();
  };

  /**
   * @brief Create the smooth-abs activation data
   *
   * @return the activation data
   */
  virtual boost::shared_ptr<ActivationDataAbstract> createData() {
    return boost::allocate_shared<Data>(Eigen::aligned_allocator<Data>(), this);
  };

 protected:
  using Base::nr_;  //!< Dimension of the residual vector
  Scalar eps_;      //!< Smoothing factor
};

template <typename _Scalar>
struct ActivationDataSmooth1NormTpl : public ActivationDataAbstractTpl<_Scalar> {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef ActivationDataAbstractTpl<Scalar> Base;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <typename Activation>
  explicit ActivationDataSmooth1NormTpl(Activation* const activation)
      : Base(activation), a(VectorXs::Zero(activation->get_nr())) {}

  VectorXs a;
  using Base::Arr;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIVATIONS_SMOOTH_1NORM_HPP_
