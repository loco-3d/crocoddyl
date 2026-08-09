///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_GUIDANCE_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_GUIDANCE_BASE_HPP_

#include "crocoddyl/core/guidance-base.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

template <typename Scalar>
class GuidanceModelAbstractTpl_wrap
    : public GuidanceModelAbstractTpl<Scalar>,
      public bp::wrapper<GuidanceModelAbstractTpl<Scalar>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(GuidanceModelBase, GuidanceModelAbstractTpl_wrap)

  typedef crocoddyl::GuidanceModelAbstractTpl<Scalar> GuidanceModel;
  typedef crocoddyl::GuidanceDataAbstractTpl<Scalar> GuidanceData;
  typedef typename GuidanceModel::VectorXs VectorXs;
  using GuidanceModel::nr_;

  explicit GuidanceModelAbstractTpl_wrap(const std::size_t nr)
      : GuidanceModel(nr), bp::wrapper<GuidanceModel>() {}

  void calc(const std::shared_ptr<GuidanceData>& data,
            const Eigen::Ref<const VectorXs>& error) const override {
    if (static_cast<std::size_t>(error.size()) != nr_) {
      throw_pretty(
          "Invalid argument: " << "error has wrong dimension (it should be " +
                                      std::to_string(nr_) + ")");
    }
    return bp::call<void>(this->get_override("calc").ptr(), data,
                          (VectorXs)error);
  }

  void calcDiff(const std::shared_ptr<GuidanceData>& data,
                const Eigen::Ref<const VectorXs>& error) const override {
    if (static_cast<std::size_t>(error.size()) != nr_) {
      throw_pretty(
          "Invalid argument: " << "error has wrong dimension (it should be " +
                                      std::to_string(nr_) + ")");
    }
    return bp::call<void>(this->get_override("calcDiff").ptr(), data,
                          (VectorXs)error);
  }

  std::shared_ptr<GuidanceData> createData() const override {
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<std::shared_ptr<GuidanceData>>(createData.ptr());
    }
    return GuidanceModel::createData();
  }

  std::shared_ptr<GuidanceData> default_createData() const {
    return this->GuidanceModel::createData();
  }

  template <typename NewScalar>
  GuidanceModelAbstractTpl_wrap<NewScalar> cast() const {
    typedef GuidanceModelAbstractTpl_wrap<NewScalar> ReturnType;
    ReturnType ret(nr_);
    return ret;
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_GUIDANCE_BASE_HPP_
