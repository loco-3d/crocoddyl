///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2019-2020, University of Edinburgh, IRI: CSIC-UPC
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_SQUASHING_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_SQUASHING_BASE_HPP_

#include "crocoddyl/core/actuation/squashing-base.hpp"
#include "crocoddyl/core/utils/exception.hpp"

namespace crocoddyl {
namespace python {

class SquashingModelAbstract_wrap : public SquashingModelAbstract, public bp::wrapper<SquashingModelAbstract> {
 public:
  SquashingModelAbstract_wrap(std::size_t ns) : SquashingModelAbstract(ns), bp::wrapper<SquashingModelAbstract>() {}

  void calc(const boost::shared_ptr<SquashingDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& s) {
    assert_pretty(static_cast<std::size_t>(s.size()) == ns_, "s has wrong dimension");
    return bp::call<void>(this->get_override("calc").ptr(), data, (Eigen::VectorXd)s);
  }

  void calcDiff(const boost::shared_ptr<SquashingDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& s) {
    assert_pretty(static_cast<std::size_t>(s.size()) == ns_, "s has wrong dimension");
    return bp::call<void>(this->get_override("calcDiff").ptr(), data, (Eigen::VectorXd)s);
  }
};

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_SQUASHING_BASE_HPP_