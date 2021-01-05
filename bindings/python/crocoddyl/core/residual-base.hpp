///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2020, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_RESIDUAL_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_RESIDUAL_BASE_HPP_

#include "crocoddyl/core/residual-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace python {

class ResidualModelAbstract_wrap : public ResidualModelAbstract, public bp::wrapper<ResidualModelAbstract> {
 public:
  ResidualModelAbstract_wrap(boost::shared_ptr<StateAbstract> state, const std::size_t nr, const std::size_t nu)
      : ResidualModelAbstract(state, nr, nu), bp::wrapper<ResidualModelAbstract>() {}

  ResidualModelAbstract_wrap(boost::shared_ptr<StateAbstract> state, const std::size_t nr)
      : ResidualModelAbstract(state, nr), bp::wrapper<ResidualModelAbstract>() {}

  void calc(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u) {
    if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
    }
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw_pretty("Invalid argument: "
                   << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
    }
    return bp::call<void>(this->get_override("calc").ptr(), data, (Eigen::VectorXd)x, (Eigen::VectorXd)u);
  }

  void calcDiff(const boost::shared_ptr<ResidualDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u) {
    if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
      throw_pretty("Invalid argument: "
                   << "x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
    }
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw_pretty("Invalid argument: "
                   << "u has wrong dimension (it should be " + std::to_string(nu_) + ")");
    }
    return bp::call<void>(this->get_override("calcDiff").ptr(), data, (Eigen::VectorXd)x, (Eigen::VectorXd)u);
  }

  boost::shared_ptr<ResidualDataAbstract> createData(DataCollectorAbstract* const data) {
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<boost::shared_ptr<ResidualDataAbstract> >(createData.ptr(), boost::ref(data));
    }
    return ResidualModelAbstract::createData(data);
  }

  boost::shared_ptr<ResidualDataAbstract> default_createData(DataCollectorAbstract* const data) {
    return this->ResidualModelAbstract::createData(data);
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_RESIDUAL_BASE_HPP_