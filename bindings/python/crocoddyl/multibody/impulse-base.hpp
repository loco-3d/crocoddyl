///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2023, LAAS-CNRS, University of Edinburgh,
//                          Heriot-Watt University
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_IMPULSE_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_IMPULSE_BASE_HPP_

#include "crocoddyl/core/utils/exception.hpp"
#include "crocoddyl/multibody/impulse-base.hpp"
#include "python/crocoddyl/multibody/multibody.hpp"

namespace crocoddyl {
namespace python {

class ImpulseModelAbstract_wrap : public ImpulseModelAbstract,
                                  public bp::wrapper<ImpulseModelAbstract> {
 public:
  ImpulseModelAbstract_wrap(std::shared_ptr<StateMultibody> state,
                            const pinocchio::ReferenceFrame type,
                            std::size_t nc)
      : ImpulseModelAbstract(state, type, nc) {}

  ImpulseModelAbstract_wrap(std::shared_ptr<StateMultibody> state,
                            std::size_t nc)
      : ImpulseModelAbstract(state, pinocchio::ReferenceFrame::LOCAL, nc) {
    std::cerr << "Deprecated: Use constructor that passes the type of contact, "
                 "this assumes is pinocchio::LOCAL."
              << std::endl;
  }

  void calc(const std::shared_ptr<ImpulseDataAbstract>& data,
            const Eigen::Ref<const Eigen::VectorXd>& x) {
    assert_pretty(static_cast<std::size_t>(x.size()) == state_->get_nx(),
                  "x has wrong dimension");
    return bp::call<void>(this->get_override("calc").ptr(), data,
                          (Eigen::VectorXd)x);
  }

  void calcDiff(const std::shared_ptr<ImpulseDataAbstract>& data,
                const Eigen::Ref<const Eigen::VectorXd>& x) {
    assert_pretty(static_cast<std::size_t>(x.size()) == state_->get_nx(),
                  "x has wrong dimension");
    return bp::call<void>(this->get_override("calcDiff").ptr(), data,
                          (Eigen::VectorXd)x);
  }

  void updateForce(const std::shared_ptr<ImpulseDataAbstract>& data,
                   const Eigen::VectorXd& force) {
    assert_pretty(static_cast<std::size_t>(force.size()) == nc_,
                  "force has wrong dimension");
    return bp::call<void>(this->get_override("updateForce").ptr(), data, force);
  }

  std::shared_ptr<ImpulseDataAbstract> createData(
      pinocchio::DataTpl<Scalar>* const data) {
    enableMultithreading() = false;
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<std::shared_ptr<ImpulseDataAbstract> >(createData.ptr(),
                                                             boost::ref(data));
    }
    return ImpulseModelAbstract::createData(data);
  }

  std::shared_ptr<ImpulseDataAbstract> default_createData(
      pinocchio::DataTpl<Scalar>* const data) {
    return this->ImpulseModelAbstract::createData(data);
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_IMPULSE_BASE_HPP_
