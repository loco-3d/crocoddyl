
///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_ACTUATION_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_ACTUATION_BASE_HPP_

#include "python/crocoddyl/core/core.hpp"
#include "crocoddyl/core/actuation-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace python {

class ActuationModelAbstract_wrap : public ActuationModelAbstract, public bp::wrapper<ActuationModelAbstract> {
 public:
  ActuationModelAbstract_wrap(boost::shared_ptr<StateAbstract> state, const std::size_t& nu)
      : ActuationModelAbstract(state, nu), bp::wrapper<ActuationModelAbstract>() {}

  void calc(const boost::shared_ptr<ActuationDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u) {
    assert_pretty(static_cast<std::size_t>(x.size()) == state_->get_nx(), "x has wrong dimension");
    assert_pretty(static_cast<std::size_t>(u.size()) == nu_, "u has wrong dimension");
    return bp::call<void>(this->get_override("calc").ptr(), data, (Eigen::VectorXd)x, (Eigen::VectorXd)u);
  }

  void calcDiff(const boost::shared_ptr<ActuationDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u) {
    assert_pretty(static_cast<std::size_t>(x.size()) == state_->get_nx(), "x has wrong dimension");
    assert_pretty(static_cast<std::size_t>(u.size()) == nu_, "u has wrong dimension");
    return bp::call<void>(this->get_override("calcDiff").ptr(), data, (Eigen::VectorXd)x, (Eigen::VectorXd)u);
  }

  boost::shared_ptr<ActuationDataAbstract> createData() {
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<boost::shared_ptr<ActuationDataAbstract> >(createData.ptr());
    }
    return ActuationModelAbstract::createData();
  }

  boost::shared_ptr<ActuationDataAbstract> default_createData() { return this->ActuationModelAbstract::createData(); }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_ACTUATION_BASE_HPP_
