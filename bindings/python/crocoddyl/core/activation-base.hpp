///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_ACTIVATION_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_ACTIVATION_BASE_HPP_

#include "crocoddyl/core/activation-base.hpp"
#include "python/crocoddyl/core/core.hpp"

namespace crocoddyl {
namespace python {

template <typename Scalar>
class ActivationModelAbstractTpl_wrap
    : public ActivationModelAbstractTpl<Scalar>,
      public bp::wrapper<ActivationModelAbstractTpl<Scalar>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ActivationModelBase, ActivationModelAbstractTpl_wrap)

  typedef typename crocoddyl::ActivationModelAbstractTpl<Scalar>
      ActivationModel;
  typedef typename crocoddyl::ActivationDataAbstractTpl<Scalar> ActivationData;
  typedef typename ActivationModel::VectorXs VectorXs;
  using ActivationModel::nr_;

  explicit ActivationModelAbstractTpl_wrap(const std::size_t nr)
      : ActivationModel(nr), bp::wrapper<ActivationModel>() {}

  void calc(const std::shared_ptr<ActivationData>& data,
            const Eigen::Ref<const VectorXs>& r) override {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty(
          "Invalid argument: " << "r has wrong dimension (it should be " +
                                      std::to_string(nr_) + ")");
    }
    return bp::call<void>(this->get_override("calc").ptr(), data, (VectorXs)r);
  }

  void calcDiff(const std::shared_ptr<ActivationData>& data,
                const Eigen::Ref<const VectorXs>& r) override {
    if (static_cast<std::size_t>(r.size()) != nr_) {
      throw_pretty(
          "Invalid argument: " << "r has wrong dimension (it should be " +
                                      std::to_string(nr_) + ")");
    }
    return bp::call<void>(this->get_override("calcDiff").ptr(), data,
                          (VectorXs)r);
  }

  std::shared_ptr<ActivationData> createData() override {
    enableMultithreading() = false;
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<std::shared_ptr<ActivationData>>(createData.ptr());
    }
    return ActivationModel::createData();
  }

  std::shared_ptr<ActivationData> default_createData() {
    return this->ActivationModel::createData();
  }

  template <typename NewScalar>
  ActivationModelAbstractTpl_wrap<NewScalar> cast() const {
    typedef ActivationModelAbstractTpl_wrap<NewScalar> ReturnType;
    ReturnType ret(nr_);
    return ret;
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_ACTIVATION_BASE_HPP_
