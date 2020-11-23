///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIVATIONS_SMOOTH_2NORM_HPP_
#define CROCODDYL_CORE_ACTIVATIONS_SMOOTH_2NORM_HPP_

#include <stdexcept>

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/activation-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {

/**
 * @brief Smooth-2Norm activation
 *
 * This activation function describes a smooth representation of a 2-norm of a
 * residual vector, i.e. \f[ \begin{equation} \sqrt{\epsilon + sum^nr_{i=0} \|r_i\|^2} \end{equation} \f] where
 * \f$\epsilon\f$ defines the smoothing factor, \f$r_i\f$ is the scalar residual for the \f$i\f$ constraints, \f$nr\f$
 * is the dimension of the residual vector.
 *
 * The computation of the function and it derivatives are carried out in `calc()` and `caldDiff()`, respectively.
 *
 * \sa `calc()`, `calcDiff()`, `createData()`
 */
template <typename _Scalar>
class ActivationModelSmooth2NormTpl : public ActivationModelAbstractTpl<_Scalar> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActivationModelAbstractTpl<Scalar> Base;
  typedef ActivationDataAbstractTpl<Scalar> ActivationDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  /**
   * @brief Initialize the smooth-2Norm activation model
   *
   * The default `eps` value is defined as 1.
   *
   * @param[in] nr   Dimension of the residual vector
   * @param[in] eps  Smoothing factor (default: 1.)
   */
  explicit ActivationModelSmooth2NormTpl(const std::size_t& nr, const Scalar& eps = Scalar(1.)) : Base(nr), eps_(eps) {
    if (eps < Scalar(0.)) {
      throw_pretty("Invalid argument: "
                   << "eps should be a positive value");
    }
  };
  virtual ~ActivationModelSmooth2NormTpl(){};

  /**
   * @brief Compute the smooth-2Norm function
   *
   * @param[in] data  Smooth-2Norm activation data
   * @param[in] r     Residual vector \f$\mathbf{r}\in\mathbb{R}^{nr}\f$
   */
  virtual void calc(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const VectorXs>& r) {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty("Invalid argument: "
                   << "r has wrong dimension (it should be " + std::to_string(nr_) + ")");
    }
    using std::sqrt;
    data->a_value = sqrt(r.squaredNorm() + eps_);
  };

  /**
   * @brief Compute the derivatives of the smooth-2Norm function
   *
   * @param[in] data  Smooth-2Norm activation data
   * @param[in] r     Residual vector \f$\mathbf{r}\in\mathbb{R}^{nr}\f$
   */
  virtual void calcDiff(const boost::shared_ptr<ActivationDataAbstract>& data, const Eigen::Ref<const VectorXs>& r) {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty("Invalid argument: "
                   << "r has wrong dimension (it should be " + std::to_string(nr_) + ")");
    }

    data->Ar = r / data->a_value;
    using std::pow;
    data->Arr.diagonal().array() = Scalar(1) / pow(data->a_value, 3);
  };

  /**
   * @brief Create the smooth-2Norm activation data
   *
   * @return the activation data
   */
  virtual boost::shared_ptr<ActivationDataAbstract> createData() {
    return boost::allocate_shared<ActivationDataAbstract>(Eigen::aligned_allocator<ActivationDataAbstract>(), this);
  };

 protected:
  using Base::nr_;  //!< Dimension of the residual vector
  Scalar eps_;      //!< Smoothing factor
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIVATIONS_SMOOTH_2NORM_HPP_
