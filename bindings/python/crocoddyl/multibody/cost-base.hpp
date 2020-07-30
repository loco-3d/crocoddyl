///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_COST_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_COST_BASE_HPP_

#include "crocoddyl/multibody/cost-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace python {

class CostModelAbstract_wrap : public CostModelAbstract, public bp::wrapper<CostModelAbstract> {
 public:
  CostModelAbstract_wrap(boost::shared_ptr<StateMultibody> state,
                         boost::shared_ptr<ActivationModelAbstract> activation, int nu)
      : CostModelAbstract(state, activation, nu) {}

  CostModelAbstract_wrap(boost::shared_ptr<StateMultibody> state,
                         boost::shared_ptr<ActivationModelAbstract> activation)
      : CostModelAbstract(state, activation) {}

  CostModelAbstract_wrap(boost::shared_ptr<StateMultibody> state, int nr, int nu)
      : CostModelAbstract(state, nr, nu), bp::wrapper<CostModelAbstract>() {}

  CostModelAbstract_wrap(boost::shared_ptr<StateMultibody> state, int nr)
      : CostModelAbstract(state, nr), bp::wrapper<CostModelAbstract>() {}

  void calc(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u) {
    assert_pretty(static_cast<std::size_t>(x.size()) == state_->get_nx(), "x has wrong dimension");
    assert_pretty((static_cast<std::size_t>(u.size()) == nu_ || nu_ == 0), "u has wrong dimension");
    return bp::call<void>(this->get_override("calc").ptr(), data, (Eigen::VectorXd)x, (Eigen::VectorXd)u);
  }

  void calcDiff(const boost::shared_ptr<CostDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u) {
    assert_pretty(static_cast<std::size_t>(x.size()) == state_->get_nx(), "x has wrong dimension");
    assert_pretty((static_cast<std::size_t>(u.size()) == nu_ || nu_ == 0), "u has wrong dimension");
    return bp::call<void>(this->get_override("calcDiff").ptr(), data, (Eigen::VectorXd)x, (Eigen::VectorXd)u);
  }

  boost::shared_ptr<CostDataAbstract> createData(DataCollectorAbstract* const data) {
    if (boost::python::override createData = this->get_override("createData")) {
      return bp::call<boost::shared_ptr<CostDataAbstract> >(createData.ptr(), boost::ref(data));
    }
    return CostModelAbstract::createData(data);
  }

  boost::shared_ptr<CostDataAbstract> default_createData(DataCollectorAbstract* const data) {
    return this->CostModelAbstract::createData(data);
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_COST_BASE_HPP_
