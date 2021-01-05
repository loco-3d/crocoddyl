///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2021, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/core/core.hpp"
#include "python/crocoddyl/core/residual-base.hpp"

namespace crocoddyl {
namespace python {

void exposeResidualAbstract() {
  bp::register_ptr_to_python<boost::shared_ptr<ResidualModelAbstract> >();

  bp::class_<ResidualModelAbstract_wrap, boost::noncopyable>(
      "ResidualModelAbstract",
      "Abstract class for residual models.\n\n"
      "In crocoddyl, a residual model defines a vector function r(x,u) in R^nr.\n"
      "where nr describes its dimension in the Euclidean space.\n"
      "For each residual model, we need to provide ways of computing the residual vector and its Jacobians.\n"
      "These computations are mainly carry on inside calc() and calcDiff(), respectively.",
      bp::init<boost::shared_ptr<StateAbstract>, std::size_t, std::size_t>(
          bp::args("self", "state", "nr", "nu"),
          "Initialize the residual model.\n\n"
          ":param state: state description,\n"
          ":param nr: dimension of the residual vector\n"
          ":param nu: dimension of control vector (default state.nv)"))
      .def(bp::init<boost::shared_ptr<StateAbstract>, std::size_t>(bp::args("self", "state", "nr"),
                                                                   "Initialize the cost model.\n\n"
                                                                   ":param state: state description\n"
                                                                   ":param nr: dimension of the residual vector"))
      .def("calc", pure_virtual(&ResidualModelAbstract_wrap::calc), bp::args("self", "data", "x", "u"),
           "Compute the residual vector.\n\n"
           ":param data: residual data\n"
           ":param x: time-discrete state vector\n"
           ":param u: time-discrete control input")
      .def<void (ResidualModelAbstract::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                           const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calc", &ResidualModelAbstract::calc, bp::args("self", "data", "x"))
      .def("calcDiff", pure_virtual(&ResidualModelAbstract_wrap::calcDiff), bp::args("self", "data", "x", "u"),
           "Compute the Jacobians of the residual function.\n\n"
           ":param data: residual data\n"
           ":param x: time-discrete state vector\n"
           ":param u: time-discrete control input\n")
      .def<void (ResidualModelAbstract::*)(const boost::shared_ptr<ResidualDataAbstract>&,
                                           const Eigen::Ref<const Eigen::VectorXd>&)>(
          "calcDiff", &ResidualModelAbstract::calcDiff, bp::args("self", "data", "x"))
      .def("createData", &ResidualModelAbstract_wrap::createData, &ResidualModelAbstract_wrap::default_createData,
           bp::args("self"),
           "Create the residual data.\n\n"
           "Each residual model might has its own data that needs to be allocated.")
      .add_property(
          "state",
          bp::make_function(&ResidualModelAbstract_wrap::get_state, bp::return_value_policy<bp::return_by_value>()),
          "state")
      .add_property(
          "nr", bp::make_function(&ResidualModelAbstract_wrap::get_nr, bp::return_value_policy<bp::return_by_value>()),
          "dimension of residual vector")
      .add_property(
          "nu", bp::make_function(&ResidualModelAbstract_wrap::get_nu, bp::return_value_policy<bp::return_by_value>()),
          "dimension of control vector");

  bp::register_ptr_to_python<boost::shared_ptr<ResidualDataAbstract> >();

  bp::class_<ResidualDataAbstract, boost::noncopyable>(
      "ResidualDataAbstract",
      "Abstract class for residual data.\n\n"
      "In crocoddyl, a residual data contains all the required information for processing an\n"
      "user-defined residual model. The residual data typically is allocated onces and containts\n"
      "the residual vector and its Jacobians.",
      bp::init<ResidualModelAbstract*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create common data shared between residual models.\n\n"
          ":param model: residual model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("shared", bp::make_getter(&ResidualDataAbstract::shared, bp::return_internal_reference<>()),
                    "shared data")
      .add_property("r", bp::make_getter(&ResidualDataAbstract::r, bp::return_internal_reference<>()),
                    bp::make_setter(&ResidualDataAbstract::r), "residual vector")
      .add_property("Rx", bp::make_getter(&ResidualDataAbstract::Rx, bp::return_internal_reference<>()),
                    bp::make_setter(&ResidualDataAbstract::Rx), "Jacobian of the residual")
      .add_property("Ru", bp::make_getter(&ResidualDataAbstract::Ru, bp::return_internal_reference<>()),
                    bp::make_setter(&ResidualDataAbstract::Ru), "Jacobian of the residual");
}

}  // namespace python
}  // namespace crocoddyl
