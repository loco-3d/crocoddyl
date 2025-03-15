///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2025, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_CONTACT_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_CONTACT_BASE_HPP_

#include "crocoddyl/multibody/contact-base.hpp"
#include "python/crocoddyl/multibody/multibody.hpp"

namespace crocoddyl {
namespace python {

template <typename Scalar>
class ContactModelAbstractTpl_wrap
    : public ContactModelAbstractTpl<Scalar>,
      public bp::wrapper<ContactModelAbstractTpl<Scalar>> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  CROCODDYL_DERIVED_CAST(ContactModelBase, ContactModelAbstractTpl_wrap)

  typedef typename crocoddyl::ContactModelAbstractTpl<Scalar> ContactModel;
  typedef typename crocoddyl::ContactDataAbstractTpl<Scalar> ContactData;
  typedef typename ContactModel::VectorXs VectorXs;
  typedef typename ContactModel::StateMultibody State;
  using ContactModel::nc_;
  using ContactModel::nu_;
  using ContactModel::state_;
  using ContactModel::type_;

  ContactModelAbstractTpl_wrap(std::shared_ptr<State> state,
                               const pinocchio::ReferenceFrame type,
                               std::size_t nc, std::size_t nu)
      : ContactModel(state, type, nc, nu) {}
  ContactModelAbstractTpl_wrap(std::shared_ptr<State> state,
                               const pinocchio::ReferenceFrame type,
                               std::size_t nc)
      : ContactModel(state, type, nc) {}

  void calc(const std::shared_ptr<ContactData>& data,
            const Eigen::Ref<const VectorXs>& x) override {
    assert_pretty(static_cast<std::size_t>(x.size()) == state_->get_nx(),
                  "x has wrong dimension");
    return bp::call<void>(this->get_override("calc").ptr(), data, (VectorXs)x);
  }

  void calcDiff(const std::shared_ptr<ContactData>& data,
                const Eigen::Ref<const VectorXs>& x) override {
    assert_pretty(static_cast<std::size_t>(x.size()) == state_->get_nx(),
                  "x has wrong dimension");
    return bp::call<void>(this->get_override("calcDiff").ptr(), data,
                          (VectorXs)x);
  }

  void updateForce(const std::shared_ptr<ContactData>& data,
                   const VectorXs& force) override {
    assert_pretty(static_cast<std::size_t>(force.size()) == nc_,
                  "force has wrong dimension");
    return bp::call<void>(this->get_override("updateForce").ptr(), data, force);
  }

  std::shared_ptr<ContactData> createData(
      pinocchio::DataTpl<Scalar>* const data) override {
    enableMultithreading() = false;
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<std::shared_ptr<ContactData>>(createData.ptr(),
                                                    boost::ref(data));
    }
    return ContactModel::createData(data);
  }

  std::shared_ptr<ContactData> default_createData(
      pinocchio::DataTpl<Scalar>* const data) {
    return this->ContactModel::createData(data);
  }

  template <typename NewScalar>
  ContactModelAbstractTpl_wrap<NewScalar> cast() const {
    typedef ContactModelAbstractTpl_wrap<NewScalar> ReturnType;
    typedef StateMultibodyTpl<NewScalar> StateType;
    ReturnType ret(
        std::make_shared<StateType>(state_->template cast<NewScalar>()), type_,
        nc_, nu_);
    return ret;
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_CONTACT_BASE_HPP_
