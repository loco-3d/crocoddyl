///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_MULTIBODY_COSTS_COST_SUM_HPP_
#define BINDINGS_PYTHON_CROCODDYL_MULTIBODY_COSTS_COST_SUM_HPP_

#include <functional>
#include <map>
#include <memory>
#include <utility>
#include <string>
#include "crocoddyl/multibody/costs/cost-sum.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

void exposeCostSum() {
  // Register custom converters between std::map and Python dict
  bp::to_python_converter<std::map<std::string, CostItem, std::less<std::string>,
                                   std::allocator<std::pair<const std::string, CostItem> > >,
                          map_to_dict<std::string, CostItem> >();
  dict_to_map<std::string, CostItem>().from_python();

  bp::class_<CostItem, boost::noncopyable>("CostItem", "Describe a cost item.\n\n",
                                           bp::init<std::string, CostModelAbstract*, double>(
                                               bp::args(" self", " name", " cost", " weight"),
                                               "Initialize the cost item.\n\n"
                                               ":param name: cost name\n"
                                               ":param cost: cost model\n"
                                               ":param weight: cost weight")[bp::with_custodian_and_ward<1, 3>()])
      .def_readwrite("name", &CostItem::name, "cost name")
      .add_property("cost", bp::make_getter(&CostItem::cost, bp::return_internal_reference<>()), "cost model")
      .def_readwrite("weight", &CostItem::weight, "cost weight");

  bp::class_<CostModelSum, bp::bases<CostModelAbstract> >(
      "CostModelSum",
      bp::init<StateMultibody&, unsigned int, bp::optional<bool> >(
          bp::args(" self", " state", " nu", " withResiduals"),
          "Initialize the total cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param nu: dimension of control vector\n"
          ":param withResiduals: true if the cost function has residuals")[bp::with_custodian_and_ward<1, 2>()])
      .def(bp::init<StateMultibody&, bp::optional<bool> >(
          bp::args(" self", " state", " withResiduals"),
          "Initialize the total cost model.\n\n"
          "For this case the default nu is equals to model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param withResiduals: true if the cost function has residuals")[bp::with_custodian_and_ward<1, 2>()])
      .def("addCost", &CostModelSum::addCost, bp::with_custodian_and_ward<1, 3>(), "add cost item")
      .def("removeCost", &CostModelSum::removeCost, "remove cost item")
      .def("calc", &CostModelSum::calc_wrap,
           CostModel_calc_wraps(bp::args(" self", " data", " x", " u=None"),
                                "Compute the total cost.\n\n"
                                ":param data: cost-sum data\n"
                                ":param x: time-discrete state vector\n"
                                ":param u: time-discrete control input"))
      .def<void (CostModelSum::*)(const boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&,
                                  const Eigen::VectorXd&, const bool&)>(
          "calcDiff", &CostModelSum::calcDiff_wrap, bp::args(" self", " data", " x", " u=None", " recalc=True"),
          "Compute the derivatives of the total cost.\n\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n"
          ":param recalc: If true, it updates the state evolution and the cost value.")
      .def<void (CostModelSum::*)(const boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&,
                                  const Eigen::VectorXd&)>("calcDiff", &CostModelSum::calcDiff_wrap,
                                                           bp::args(" self", " data", " x", " u"))
      .def<void (CostModelSum::*)(const boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&)>(
          "calcDiff", &CostModelSum::calcDiff_wrap, bp::args(" self", " data", " x"))
      .def<void (CostModelSum::*)(const boost::shared_ptr<CostDataAbstract>&, const Eigen::VectorXd&, const bool&)>(
          "calcDiff", &CostModelSum::calcDiff_wrap, bp::args(" self", " data", " x", " recalc"))
      .def("createData", &CostModelSum::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args(" self", " data"),
           "Create the total cost data.\n\n"
           ":param data: Pinocchio data\n"
           ":return total cost data.")
      .add_property("costs",
                    bp::make_function(&CostModelSum::get_costs, bp::return_value_policy<bp::return_by_value>()),
                    "stack of costs")
      .add_property("nr", bp::make_function(&CostModelSum::get_nr, bp::return_value_policy<bp::return_by_value>()),
                    "dimension of the total residual vector");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_COSTS_COST_SUM_HPP_
