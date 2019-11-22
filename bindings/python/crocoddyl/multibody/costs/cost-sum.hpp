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
#include "python/crocoddyl/utils/map-converter.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(CostModelSum_calc_wraps, CostModelSum::calc_wrap, 2, 3)

void exposeCostSum() {
  // Register custom converters between std::map and Python dict
  typedef boost::shared_ptr<CostDataAbstract> CostDataPtr;
  bp::to_python_converter<std::map<std::string, CostItem, std::less<std::string>,
                                   std::allocator<std::pair<const std::string, CostItem> > >,
                          map_to_dict<std::string, CostItem> >();
  bp::to_python_converter<std::map<std::string, CostDataPtr, std::less<std::string>,
                                   std::allocator<std::pair<const std::string, CostDataPtr> > >,
                          map_to_dict<std::string, CostDataPtr, false> >();
  dict_to_map<std::string, CostItem>().from_python();
  dict_to_map<std::string, CostDataPtr>().from_python();

  bp::class_<CostItem, boost::noncopyable>("CostItem", "Describe a cost item.\n\n",
                                           bp::init<std::string, boost::shared_ptr<CostModelAbstract>, double>(
                                               bp::args(" self", " name", " cost", " weight"),
                                               "Initialize the cost item.\n\n"
                                               ":param name: cost name\n"
                                               ":param cost: cost model\n"
                                               ":param weight: cost weight"))
      .def_readwrite("name", &CostItem::name, "cost name")
      .add_property("cost", bp::make_getter(&CostItem::cost, bp::return_value_policy<bp::return_by_value>()),
                    "cost model")
      .def_readwrite("weight", &CostItem::weight, "cost weight");

  bp::register_ptr_to_python<boost::shared_ptr<CostModelSum> >();

  bp::class_<CostModelSum, boost::noncopyable>("CostModelSum",
                                               bp::init<boost::shared_ptr<StateMultibody>, std::size_t, bool>(
                                                   bp::args(" self", " state", " nu=state.nv", " withResiduals=True"),
                                                   "Initialize the total cost model.\n\n"
                                                   ":param state: state of the multibody system\n"
                                                   ":param nu: dimension of control vector\n"
                                                   ":param withResiduals: true if the cost function has residuals"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, std::size_t>(
          bp::args(" self", " state", " nu"),
          "Initialize the total cost model.\n\n"
          "For this case the default nu is equals to model.nv.\n"
          ":param state: state of the multibody system\n"
          ":param nu: dimension of control vector\n"
          ":param withResiduals: true if the cost function has residuals"))
      .def(bp::init<boost::shared_ptr<StateMultibody> >(bp::args(" self", " state"),
                                                        "Initialize the total cost model.\n\n"
                                                        "For this case the default nu is equals to model.nv.\n"
                                                        ":param state: state of the multibody system"))
      .def("addCost", &CostModelSum::addCost, bp::args(" self", " name", " cost", " weight"),
           "Add a cost item.\n\n"
           ":param name: cost name\n"
           ":param cost: cost model\n"
           ":param weight: cost weight")
      .def("removeCost", &CostModelSum::removeCost, bp::args(" self", " name"),
           "Remove a cost item.\n\n"
           ":param name: cost name")
      .def("calc", &CostModelSum::calc_wrap,
           CostModelSum_calc_wraps(bp::args(" self", " data", " x", " u=None"),
                                   "Compute the total cost.\n\n"
                                   ":param data: cost-sum data\n"
                                   ":param x: time-discrete state vector\n"
                                   ":param u: time-discrete control input"))
      .def<void (CostModelSum::*)(const boost::shared_ptr<CostDataSum>&, const Eigen::VectorXd&,
                                  const Eigen::VectorXd&, const bool&)>(
          "calcDiff", &CostModelSum::calcDiff_wrap, bp::args(" self", " data", " x", " u=None", " recalc=True"),
          "Compute the derivatives of the total cost.\n\n"
          ":param data: action data\n"
          ":param x: time-discrete state vector\n"
          ":param u: time-discrete control input\n"
          ":param recalc: If true, it updates the state evolution and the cost value.")
      .def<void (CostModelSum::*)(const boost::shared_ptr<CostDataSum>&, const Eigen::VectorXd&,
                                  const Eigen::VectorXd&)>("calcDiff", &CostModelSum::calcDiff_wrap,
                                                           bp::args(" self", " data", " x", " u"))
      .def<void (CostModelSum::*)(const boost::shared_ptr<CostDataSum>&, const Eigen::VectorXd&)>(
          "calcDiff", &CostModelSum::calcDiff_wrap, bp::args(" self", " data", " x"))
      .def<void (CostModelSum::*)(const boost::shared_ptr<CostDataSum>&, const Eigen::VectorXd&, const bool&)>(
          "calcDiff", &CostModelSum::calcDiff_wrap, bp::args(" self", " data", " x", " recalc"))
      .def("createData", &CostModelSum::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args(" self", " data"),
           "Create the total cost data.\n\n"
           ":param data: Pinocchio data\n"
           ":return total cost data.")
      .add_property("state",
                    bp::make_function(&CostModelSum::get_state, bp::return_value_policy<bp::return_by_value>()),
                    "state of the multibody system")
      .add_property("costs",
                    bp::make_function(&CostModelSum::get_costs, bp::return_value_policy<bp::return_by_value>()),
                    "stack of costs")
      .add_property("nu", bp::make_function(&CostModelSum::get_nu, bp::return_value_policy<bp::return_by_value>()),
                    "dimension of control vector")
      .add_property("nr", bp::make_function(&CostModelSum::get_nr, bp::return_value_policy<bp::return_by_value>()),
                    "dimension of the total residual vector");

  bp::class_<CostDataSum, boost::shared_ptr<CostDataSum>, boost::noncopyable>(
      "CostDataSum", "Class for total cost data.\n\n",
      bp::init<CostModelSum*, pinocchio::Data*>(bp::args(" self", " model", " data"),
                                                "Create total cost data.\n\n"
                                                ":param model: total cost model\n"
                                                ":param data: Pinocchio data")[bp::with_custodian_and_ward<1, 3>()])
      .def("shareMemory", &CostDataSum::shareMemory<DifferentialActionDataAbstract>, bp::args(" self", " model"),
           "Share memory with a given differential action data\n\n"
           ":param model: differential action data that we want to share memory")
      .def("shareMemory", &CostDataSum::shareMemory<ActionDataAbstract>, bp::args(" self", " model"),
           "Share memory with a given action data\n\n"
           ":param model: action data that we want to share memory")
      .add_property("costs", bp::make_getter(&CostDataSum::costs, bp::return_value_policy<bp::return_by_value>()),
                    "stack of costs data")
      .add_property("pinocchio", bp::make_getter(&CostDataSum::pinocchio, bp::return_internal_reference<>()),
                    "pinocchio data")
      .add_property("cost", bp::make_getter(&CostDataSum::cost, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&CostDataSum::cost), "cost value")
      .add_property("r", bp::make_function(&CostDataSum::get_r, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_function(&CostDataSum::set_r), "cost residual")
      .add_property("Lx", bp::make_function(&CostDataSum::get_Lx, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_function(&CostDataSum::set_Lx), "Jacobian of the cost")
      .add_property("Lu", bp::make_function(&CostDataSum::get_Lu, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_function(&CostDataSum::set_Lu), "Jacobian of the cost")
      .add_property("Lxx", bp::make_function(&CostDataSum::get_Lxx, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_function(&CostDataSum::set_Lxx), "Hessian of the cost")
      .add_property("Lxu", bp::make_function(&CostDataSum::get_Lxu, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_function(&CostDataSum::set_Lxu), "Hessian of the cost")
      .add_property("Luu", bp::make_function(&CostDataSum::get_Luu, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_function(&CostDataSum::set_Luu), "Hessian of the cost")
      .add_property("Rx", bp::make_getter(&CostDataSum::Rx, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&CostDataSum::Rx), "Jacobian of the cost residual")
      .add_property("Ru", bp::make_getter(&CostDataSum::Ru, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&CostDataSum::Ru), "Jacobian of the cost residual");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_MULTIBODY_COSTS_COST_SUM_HPP_
