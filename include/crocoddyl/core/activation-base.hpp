///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIVATION_BASE_HPP_
#define CROCODDYL_CORE_ACTIVATION_BASE_HPP_

#include "crocoddyl/core/fwd.hpp"

namespace crocoddyl {

class ActivationModelBase {
 public:
  virtual ~ActivationModelBase() = default;

  CROCODDYL_BASE_CAST(ActivationModelBase, ActivationModelAbstractTpl)
};

template <typename _Scalar>
class ActivationModelAbstractTpl : public ActivationModelBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef ActivationDataAbstractTpl<Scalar> ActivationDataAbstract;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  explicit ActivationModelAbstractTpl(const std::size_t nr) : nr_(nr) {};
  virtual ~ActivationModelAbstractTpl() = default;

  virtual void calc(const std::shared_ptr<ActivationDataAbstract>& data,
                    const Eigen::Ref<const VectorXs>& r) = 0;
  virtual void calcDiff(const std::shared_ptr<ActivationDataAbstract>& data,
                        const Eigen::Ref<const VectorXs>& r) = 0;
  virtual std::shared_ptr<ActivationDataAbstract> createData() {
    return std::allocate_shared<ActivationDataAbstract>(
        Eigen::aligned_allocator<ActivationDataAbstract>(), this);
  };

  std::size_t get_nr() const { return nr_; };

  /**
   * @brief Print information on the activation model
   */
  friend std::ostream& operator<<(
      std::ostream& os, const ActivationModelAbstractTpl<Scalar>& model) {
    model.print(os);
    return os;
  }

  /**
   * @brief Print relevant information of the activation model
   *
   * @param[out] os  Output stream object
   */
  virtual void print(std::ostream& os) const {
    os << boost::core::demangle(typeid(*this).name());
  }

 protected:
  std::size_t nr_;
  ActivationModelAbstractTpl() : nr_(0) {};
};

template <typename _Scalar>
struct ActivationDataAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;
  typedef typename MathBase::DiagonalMatrixXs DiagonalMatrixXs;

  template <template <typename Scalar> class Activation>
  explicit ActivationDataAbstractTpl(Activation<Scalar>* const activation)
      : a_value(Scalar(0.)),
        Ar(VectorXs::Zero(activation->get_nr())),
        Arr(DiagonalMatrixXs(activation->get_nr())) {
    Arr.setZero();
  }
  virtual ~ActivationDataAbstractTpl() = default;

  Scalar a_value;
  VectorXs Ar;
  DiagonalMatrixXs Arr;

  static MatrixXs getHessianMatrix(
      const ActivationDataAbstractTpl<Scalar>& data) {
    return data.Arr.diagonal().asDiagonal();
  }
  static void setHessianMatrix(ActivationDataAbstractTpl<Scalar>& data,
                               const MatrixXs& Arr) {
    data.Arr.diagonal() = Arr.diagonal();
  }
};

}  // namespace crocoddyl

CROCODDYL_DECLARE_EXTERN_TEMPLATE_CLASS(crocoddyl::ActivationModelAbstractTpl)
CROCODDYL_DECLARE_EXTERN_TEMPLATE_STRUCT(crocoddyl::ActivationDataAbstractTpl)

#endif  // CROCODDYL_CORE_ACTIVATION_BASE_HPP_
