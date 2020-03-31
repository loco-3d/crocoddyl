///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef CROCODDYL_CORE_ACTIVATION_BASE_HPP_
#define CROCODDYL_CORE_ACTIVATION_BASE_HPP_

#include <stdexcept>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include "crocoddyl/core/fwd.hpp"
#include "crocoddyl/core/mathbase.hpp"
#include "crocoddyl/core/utils/to-string.hpp"

namespace crocoddyl {

template <typename _Scalar>
class ActivationModelAbstractTpl {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  explicit ActivationModelAbstractTpl(const std::size_t& nr) : nr_(nr){};
  virtual ~ActivationModelAbstractTpl(){};

  virtual void calc(const boost::shared_ptr<ActivationDataAbstractTpl<Scalar> >& data,
                    const Eigen::Ref<const VectorXs>& r) = 0;
  virtual void calcDiff(const boost::shared_ptr<ActivationDataAbstractTpl<Scalar> >& data,
                        const Eigen::Ref<const VectorXs>& r) = 0;
  virtual boost::shared_ptr<ActivationDataAbstractTpl<Scalar> > createData() {
    return boost::make_shared<ActivationDataAbstractTpl<Scalar> >(this);
  };

  const std::size_t& get_nr() const { return nr_; };

 protected:
  std::size_t nr_;
};

template <typename _Scalar>
struct ActivationDataAbstractTpl {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef _Scalar Scalar;
  typedef MathBaseTpl<Scalar> MathBase;
  typedef typename MathBase::VectorXs VectorXs;
  typedef typename MathBase::MatrixXs MatrixXs;

  template <template <typename Scalar> class Activation>
  explicit ActivationDataAbstractTpl(Activation<Scalar>* const activation)
      : a_value(0.),
        Ar(VectorXs::Zero(activation->get_nr())),
        Arr(MatrixXs::Zero(activation->get_nr(), activation->get_nr())) {}
  virtual ~ActivationDataAbstractTpl() {}

  Scalar a_value;
  VectorXs Ar;
  MatrixXs Arr;
};

}  // namespace crocoddyl

#endif  // CROCODDYL_CORE_ACTIVATION_BASE_HPP_
