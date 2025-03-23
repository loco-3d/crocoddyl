///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_IMPULSE_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_IMPULSE_BASE_HPP_

#include "crocoddyl/multibody/impulse-base.hpp"
#include "python/crocoddyl/multibody/multibody.hpp"

namespace crocoddyl {
namespace python {

template <typename Scalar>
class ImpulseModelAbstractTpl_wrap
    : public ImpulseModelAbstractTpl<Scalar>,
      public bp::wrapper<ImpulseModelAbstractTpl<Scalar>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ImpulseModelBase, ImpulseModelAbstractTpl_wrap)

  typedef typename crocoddyl::ImpulseModelAbstractTpl<Scalar> ImpulseModel;
  typedef typename crocoddyl::ImpulseDataAbstractTpl<Scalar> ImpulseData;
  typedef typename ImpulseModel::VectorXs VectorXs;
  typedef typename ImpulseModel::StateMultibody State;
  using ImpulseModel::nc_;
  using ImpulseModel::state_;
  using ImpulseModel::type_;

  ImpulseModelAbstractTpl_wrap(std::shared_ptr<State> state,
                               const pinocchio::ReferenceFrame type,
                               std::size_t nc)
      : ImpulseModel(state, type, nc) {}

  ImpulseModelAbstractTpl_wrap(std::shared_ptr<State> state, std::size_t nc)
      : ImpulseModel(state, pinocchio::ReferenceFrame::LOCAL, nc) {
    std::cerr << "Deprecated: Use constructor that passes the type of contact, "
                 "this assumes is pinocchio::LOCAL."
              << std::endl;
  }

  void calc(const std::shared_ptr<ImpulseData>& data,
            const Eigen::Ref<const VectorXs>& x) override {
    assert_pretty(static_cast<std::size_t>(x.size()) == state_->get_nx(),
                  "x has wrong dimension");
    return bp::call<void>(this->get_override("calc").ptr(), data, (VectorXs)x);
  }

  void calcDiff(const std::shared_ptr<ImpulseData>& data,
                const Eigen::Ref<const VectorXs>& x) override {
    assert_pretty(static_cast<std::size_t>(x.size()) == state_->get_nx(),
                  "x has wrong dimension");
    return bp::call<void>(this->get_override("calcDiff").ptr(), data,
                          (VectorXs)x);
  }

  void updateForce(const std::shared_ptr<ImpulseData>& data,
                   const VectorXs& force) override {
    assert_pretty(static_cast<std::size_t>(force.size()) == nc_,
                  "force has wrong dimension");
    return bp::call<void>(this->get_override("updateForce").ptr(), data, force);
  }

  std::shared_ptr<ImpulseData> createData(
      pinocchio::DataTpl<Scalar>* const data) override {
    enableMultithreading() = false;
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<std::shared_ptr<ImpulseData>>(createData.ptr(),
                                                    boost::ref(data));
    }
    return ImpulseModel::createData(data);
  }

  std::shared_ptr<ImpulseData> default_createData(
      pinocchio::DataTpl<Scalar>* const data) {
    return this->ImpulseModel::createData(data);
  }

  template <typename NewScalar>
  ImpulseModelAbstractTpl_wrap<NewScalar> cast() const {
    typedef ImpulseModelAbstractTpl_wrap<NewScalar> ReturnType;
    typedef StateMultibodyTpl<NewScalar> StateType;
    ReturnType ret(
        std::make_shared<StateType>(state_->template cast<NewScalar>()), type_,
        nc_);
    return ret;
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_IMPULSE_BASE_HPP_
