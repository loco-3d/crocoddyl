///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_CONTACT_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_CONTACT_BASE_HPP_

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/contact-base.hpp"
#include "python/crocoddyl/multibody/multibody.hpp"

namespace crocoddyl {
namespace python {

class ContactModelAbstract_wrap : public ContactModelAbstract,
                                  public bp::wrapper<ContactModelAbstract> {
 public:
  ContactModelAbstract_wrap(std::shared_ptr<StateMultibody> state,
                            const pinocchio::ReferenceFrame type,
                            std::size_t nc, std::size_t nu)
      : ContactModelAbstract(state, type, nc, nu) {}
  ContactModelAbstract_wrap(std::shared_ptr<StateMultibody> state,
                            const pinocchio::ReferenceFrame type,
                            std::size_t nc)
      : ContactModelAbstract(state, type, nc) {}

  void calc(const std::shared_ptr<ContactDataAbstract>& data,
            const Eigen::Ref<const Eigen::VectorXd>& x) {
    assert_pretty(static_cast<std::size_t>(x.size()) == state_->get_nx(),
                  "x has wrong dimension");
    return bp::call<void>(this->get_override("calc").ptr(), data,
                          (Eigen::VectorXd)x);
  }

  void calcDiff(const std::shared_ptr<ContactDataAbstract>& data,
                const Eigen::Ref<const Eigen::VectorXd>& x) {
    assert_pretty(static_cast<std::size_t>(x.size()) == state_->get_nx(),
                  "x has wrong dimension");
    return bp::call<void>(this->get_override("calcDiff").ptr(), data,
                          (Eigen::VectorXd)x);
  }

  void updateForce(const std::shared_ptr<ContactDataAbstract>& data,
                   const Eigen::VectorXd& force) {
    assert_pretty(static_cast<std::size_t>(force.size()) == nc_,
                  "force has wrong dimension");
    return bp::call<void>(this->get_override("updateForce").ptr(), data, force);
  }

  std::shared_ptr<ContactDataAbstract> createData(
      pinocchio::DataTpl<Scalar>* const data) {
    enableMultithreading() = false;
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<std::shared_ptr<ContactDataAbstract> >(createData.ptr(),
                                                             boost::ref(data));
    }
    return ContactModelAbstract::createData(data);
  }

  std::shared_ptr<ContactDataAbstract> default_createData(
      pinocchio::DataTpl<Scalar>* const data) {
    return this->ContactModelAbstract::createData(data);
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_CONTACT_BASE_HPP_
