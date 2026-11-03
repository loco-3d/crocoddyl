///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2026, Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_IMPLICIT_CONSTRAINT_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_IMPLICIT_CONSTRAINT_BASE_HPP_

#include "crocoddyl/multibody/implicit-constraint-base.hpp"
#include "python/crocoddyl/multibody/multibody.hpp"

namespace crocoddyl {
namespace python {

template <typename Scalar>
class ImplicitConstraintModelAbstractTpl_wrap
    : public ImplicitConstraintModelAbstractTpl<Scalar>,
      public bp::wrapper<ImplicitConstraintModelAbstractTpl<Scalar>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ImplicitConstraintModelBase,
                         ImplicitConstraintModelAbstractTpl_wrap)

  typedef crocoddyl::ImplicitConstraintModelAbstractTpl<Scalar>
      ImplicitConstraintModel;
  typedef crocoddyl::ImplicitConstraintDataAbstractTpl<Scalar>
      ImplicitConstraintData;
  typedef typename ImplicitConstraintModel::VectorXs VectorXs;
  typedef typename ImplicitConstraintModel::StateMultibody State;
  using ImplicitConstraintModel::nc_;
  using ImplicitConstraintModel::nu_;
  using ImplicitConstraintModel::state_;
  using ImplicitConstraintModel::type_;

  ImplicitConstraintModelAbstractTpl_wrap(std::shared_ptr<State> state,
                                          const pinocchio::ReferenceFrame type,
                                          std::size_t nc, std::size_t nu)
      : ImplicitConstraintModel(state, type, nc, nu) {}
  ImplicitConstraintModelAbstractTpl_wrap(std::shared_ptr<State> state,
                                          const pinocchio::ReferenceFrame type,
                                          std::size_t nc)
      : ImplicitConstraintModel(state, type, nc) {}

  void calc(const std::shared_ptr<ImplicitConstraintData>& data,
            const Eigen::Ref<const VectorXs>& x) override {
    assert_pretty(static_cast<std::size_t>(x.size()) == state_->get_nx(),
                  "x has wrong dimension");
    return bp::call<void>(this->get_override("calc").ptr(), data, (VectorXs)x);
  }

  void calcDiff(const std::shared_ptr<ImplicitConstraintData>& data,
                const Eigen::Ref<const VectorXs>& x) override {
    assert_pretty(static_cast<std::size_t>(x.size()) == state_->get_nx(),
                  "x has wrong dimension");
    return bp::call<void>(this->get_override("calcDiff").ptr(), data,
                          (VectorXs)x);
  }

  void updateForce(const std::shared_ptr<ImplicitConstraintData>& data,
                   const VectorXs& force) override {
    assert_pretty(static_cast<std::size_t>(force.size()) == nc_,
                  "force has wrong dimension");
    return bp::call<void>(this->get_override("updateForce").ptr(), data, force);
  }

  std::shared_ptr<ImplicitConstraintData> createData(
      pinocchio::DataTpl<Scalar>* const data) override {
    enableMultithreading() = false;
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<std::shared_ptr<ImplicitConstraintData>>(
          createData.ptr(), boost::ref(data));
    }
    return ImplicitConstraintModel::createData(data);
  }

  std::shared_ptr<ImplicitConstraintData> default_createData(
      pinocchio::DataTpl<Scalar>* const data) {
    return this->ImplicitConstraintModel::createData(data);
  }

  template <typename NewScalar>
  ImplicitConstraintModelAbstractTpl_wrap<NewScalar> cast() const {
    typedef ImplicitConstraintModelAbstractTpl_wrap<NewScalar> ReturnType;
    typedef StateMultibodyTpl<NewScalar> StateType;
    ReturnType ret(
        std::make_shared<StateType>(state_->template cast<NewScalar>()), type_,
        nc_, nu_);
    return ret;
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_IMPLICIT_CONSTRAINT_BASE_HPP_
