///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, University of Edinburgh, IRI: CSIC-UPC
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_SQUASHING_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_SQUASHING_BASE_HPP_

#include "crocoddyl/core/actuation/squashing-base.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

template <typename Scalar>
class SquashingModelAbstractTpl_wrap
    : public SquashingModelAbstractTpl<Scalar>,
      public bp::wrapper<SquashingModelAbstractTpl<Scalar>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(SquashingModelBase, SquashingModelAbstractTpl_wrap)

  typedef typename crocoddyl::SquashingModelAbstractTpl<Scalar> SquashingModel;
  typedef typename crocoddyl::SquashingDataAbstractTpl<Scalar> SquashingData;
  typedef typename SquashingModel::VectorXs VectorXs;
  using SquashingModel::ns_;

  SquashingModelAbstractTpl_wrap(const std::size_t ns)
      : SquashingModel(ns), bp::wrapper<SquashingModel>() {}

  void calc(const std::shared_ptr<SquashingData>& data,
            const Eigen::Ref<const VectorXs>& s) override {
    assert_pretty(static_cast<std::size_t>(s.size()) == ns_,
                  "s has wrong dimension");
    return bp::call<void>(this->get_override("calc").ptr(), data, (VectorXs)s);
  }

  void calcDiff(const std::shared_ptr<SquashingData>& data,
                const Eigen::Ref<const VectorXs>& s) override {
    assert_pretty(static_cast<std::size_t>(s.size()) == ns_,
                  "s has wrong dimension");
    return bp::call<void>(this->get_override("calcDiff").ptr(), data,
                          (VectorXs)s);
  }

  template <typename NewScalar>
  SquashingModelAbstractTpl_wrap<NewScalar> cast() const {
    typedef SquashingModelAbstractTpl_wrap<NewScalar> ReturnType;
    ReturnType ret(ns_);
    return ret;
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_SQUASHING_BASE_HPP_
