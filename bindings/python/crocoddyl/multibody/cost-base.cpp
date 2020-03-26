///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#include "python/crocoddyl/multibody/multibody.hpp"
#include "python/crocoddyl/multibody/cost-base.hpp"

namespace crocoddyl {
namespace python {

void exposeCostAbstract() {
  bp::register_ptr_to_python<boost::shared_ptr<CostModelAbstract> >();

  bp::class_<CostModelAbstract_wrap, boost::noncopyable>(
      "CostModelAbstract",
      "Abstract multibody cost model using Pinocchio.\n\n"
      "It defines a template of cost model whose residual and derivatives can be retrieved from\n"
      "Pinocchio data, through the calc and calcDiff functions, respectively.",
      bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract>, int>(
          bp::args("self", "state", "activation", "nu"),
          "Initialize the cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param activation: Activation model\n"
          ":param nu: dimension of control vector (default model.nv)"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, boost::shared_ptr<ActivationModelAbstract> >(
          bp::args("self", "state", "activation"),
          "Initialize the cost model.\n\n"
          ":param state: state of the multibody system\n"
          ":param activation: Activation model"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, int, int>(
          bp::args("self", "state", "nr", "nu"),
          "Initialize the cost model.\n\n"
          "For this case the default activation model is quadratic, i.e. crocoddyl.ActivationModelQuad(nr).\n"
          ":param state: state of the multibody system\n"
          ":param nr: dimension of cost vector\n"
          ":param nu: dimension of control vector (default model.nv)"))
      .def(bp::init<boost::shared_ptr<StateMultibody>, int>(
          bp::args("self", "state", "nr"),
          "Initialize the cost model.\n\n"
          "For this case the default activation model is quadratic, i.e. crocoddyl.ActivationModelQuad(nr).\n"
          ":param state: state of the multibody system\n"
          ":param nr: dimension of cost vector"))
      .def("calc", pure_virtual(&CostModelAbstract_wrap::calc), bp::args("self", "data", "x", "u"),
           "Compute the cost value and its residuals.\n\n"
           ":param data: cost data\n"
           ":param x: state vector\n"
           ":param u: control input")
      .def("calcDiff", pure_virtual(&CostModelAbstract_wrap::calcDiff), bp::args("self", "data", "x", "u"),
           "Compute the derivatives of the cost function and its residuals.\n\n"
           ":param data: cost data\n"
           ":param x: state vector\n"
           ":param u: control input\n")
      .def("createData", &CostModelAbstract_wrap::createData, bp::with_custodian_and_ward_postcall<0, 2>(),
           bp::args("self", "data"),
           "Create the cost data.\n\n"
           "Each cost model has its own data that needs to be allocated. This function\n"
           "returns the allocated data for a predefined cost.\n"
           ":param data: shared data\n"
           ":return cost data.")
      .add_property(
          "state",
          bp::make_function(&CostModelAbstract_wrap::get_state, bp::return_value_policy<bp::return_by_value>()),
          "state of the multibody system")
      .add_property(
          "activation",
          bp::make_function(&CostModelAbstract_wrap::get_activation, bp::return_value_policy<bp::return_by_value>()),
          "activation model")
      .add_property("nu",
                    bp::make_function(&CostModelAbstract_wrap::get_nu, bp::return_value_policy<bp::return_by_value>()),
                    "dimension of control vector");

  bp::register_ptr_to_python<boost::shared_ptr<CostDataAbstract> >();

  bp::class_<CostDataAbstract, boost::noncopyable>(
      "CostDataAbstract", "Abstract class for cost data.\n\n",
      bp::init<CostModelAbstract*, DataCollectorAbstract*>(
          bp::args("self", "model", "data"),
          "Create common data shared between cost models.\n\n"
          ":param model: cost model\n"
          ":param data: shared data")[bp::with_custodian_and_ward<1, 2, bp::with_custodian_and_ward<1, 3> >()])
      .add_property("shared", bp::make_getter(&CostDataAbstract::shared, bp::return_internal_reference<>()),
                    "shared data")
      .add_property("activation",
                    bp::make_getter(&CostDataAbstract::activation, bp::return_value_policy<bp::return_by_value>()),
                    "terminal data")
      .add_property("cost", bp::make_getter(&CostDataAbstract::cost, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&CostDataAbstract::cost), "cost value")
      .add_property("Lx", bp::make_getter(&CostDataAbstract::Lx, bp::return_internal_reference<>()),
                    bp::make_setter(&CostDataAbstract::Lx), "Jacobian of the cost")
      .add_property("Lu", bp::make_getter(&CostDataAbstract::Lu, bp::return_internal_reference<>()),
                    bp::make_setter(&CostDataAbstract::Lu), "Jacobian of the cost")
      .add_property("Lxx", bp::make_getter(&CostDataAbstract::Lxx, bp::return_internal_reference<>()),
                    bp::make_setter(&CostDataAbstract::Lxx), "Hessian of the cost")
      .add_property("Lxu", bp::make_getter(&CostDataAbstract::Lxu, bp::return_internal_reference<>()),
                    bp::make_setter(&CostDataAbstract::Lxu), "Hessian of the cost")
      .add_property("Luu", bp::make_getter(&CostDataAbstract::Luu, bp::return_internal_reference<>()),
                    bp::make_setter(&CostDataAbstract::Luu), "Hessian of the cost")
      .add_property("r", bp::make_getter(&CostDataAbstract::r, bp::return_internal_reference<>()),
                    bp::make_setter(&CostDataAbstract::r), "cost residual")
      .add_property("Rx", bp::make_getter(&CostDataAbstract::Rx, bp::return_internal_reference<>()),
                    bp::make_setter(&CostDataAbstract::Rx), "Jacobian of the cost residual")
      .add_property("Ru", bp::make_getter(&CostDataAbstract::Ru, bp::return_internal_reference<>()),
                    bp::make_setter(&CostDataAbstract::Ru), "Jacobian of the cost residual");
}

}  // namespace python
}  // namespace crocoddyl
