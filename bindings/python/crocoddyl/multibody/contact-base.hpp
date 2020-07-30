///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_CONTACT_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_CONTACT_BASE_HPP_

#include "python/crocoddyl/multibody/multibody.hpp"
#include "crocoddyl/multibody/contact-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace python {

class ContactModelAbstract_wrap : public ContactModelAbstract, public bp::wrapper<ContactModelAbstract> {
 public:
  ContactModelAbstract_wrap(boost::shared_ptr<StateMultibody> state, int nc, int nu)
      : ContactModelAbstract(state, nc, nu) {}
  ContactModelAbstract_wrap(boost::shared_ptr<StateMultibody> state, int nc) : ContactModelAbstract(state, nc) {}

  void calc(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x) {
    assert_pretty(static_cast<std::size_t>(x.size()) == state_->get_nx(), "x has wrong dimension");
    return bp::call<void>(this->get_override("calc").ptr(), data, (Eigen::VectorXd)x);
  }

  void calcDiff(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x) {
    assert_pretty(static_cast<std::size_t>(x.size()) == state_->get_nx(), "x has wrong dimension");
    return bp::call<void>(this->get_override("calcDiff").ptr(), data, (Eigen::VectorXd)x);
  }

  void updateForce(const boost::shared_ptr<ContactDataAbstract>& data, const Eigen::VectorXd& force) {
    assert_pretty(static_cast<std::size_t>(force.size()) == nc_, "force has wrong dimension");
    return bp::call<void>(this->get_override("updateForce").ptr(), data, force);
  }

  boost::shared_ptr<ContactDataAbstract> createData(pinocchio::DataTpl<Scalar>* const data) {
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<boost::shared_ptr<ContactDataAbstract> >(createData.ptr(), boost::ref(data));
    }
    return ContactModelAbstract::createData(data);
  }

  boost::shared_ptr<ContactDataAbstract> default_createData(pinocchio::DataTpl<Scalar>* const data) {
    return this->ContactModelAbstract::createData(data);
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_CONTACT_BASE_HPP_