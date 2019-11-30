///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2020, LAAS-CNRS, University of Edinburgh
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_DIFF_ACTION_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_DIFF_ACTION_BASE_HPP_

#include "crocoddyl/core/diff-action-base.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

class DifferentialActionModelAbstract_wrap : public DifferentialActionModelAbstract,
                                             public bp::wrapper<DifferentialActionModelAbstract> {
 public:
  DifferentialActionModelAbstract_wrap(boost::shared_ptr<StateAbstract> state, int nu, int nr = 1)
      : DifferentialActionModelAbstract(state, nu, nr), bp::wrapper<DifferentialActionModelAbstract>() {}

  void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u) {
    if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
      throw CrocoddylException("x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
    }
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw CrocoddylException("u has wrong dimension (it should be " + std::to_string(nu_) + ")");
    }
    return bp::call<void>(this->get_override("calc").ptr(), data, (Eigen::VectorXd)x, (Eigen::VectorXd)u);
  }

  void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u,
                const bool& recalc = true) {
    if (static_cast<std::size_t>(x.size()) != state_->get_nx()) {
      throw CrocoddylException("x has wrong dimension (it should be " + std::to_string(state_->get_nx()) + ")");
    }
    if (static_cast<std::size_t>(u.size()) != nu_) {
      throw CrocoddylException("u has wrong dimension (it should be " + std::to_string(nu_) + ")");
    }
    return bp::call<void>(this->get_override("calcDiff").ptr(), data, (Eigen::VectorXd)x, (Eigen::VectorXd)u, recalc);
  }
};

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(DiffActionModel_calc_wraps, DifferentialActionModelAbstract::calc_wrap, 2, 3)

void exposeDifferentialActionAbstract() {
  bp::register_ptr_to_python<boost::shared_ptr<DifferentialActionModelAbstract> >();

  bp::class_<DifferentialActionModelAbstract_wrap, boost::noncopyable>(
      "DifferentialActionModelAbstract",
      "Abstract class for the differential action model.\n\n"
      "In crocoddyl, a differential action model combines dynamics and cost data described in\n"
      "continuous time. Each node, in our optimal control problem, is described through an\n"
      "action model. Every time that we want describe a problem, we need to provide ways of\n"
      "computing the dynamics, cost functions and their derivatives. These computations are\n"
      "mainly carry on inside calc() and calcDiff(), respectively.",
      bp::init<boost::shared_ptr<StateAbstract>, int, bp::optional<int> >(
          bp::args(" self", " state", " nu", " nr=1"),
          "Initialize the differential action model.\n\n"
          "You can also describe autonomous systems by setting nu = 0.\n"
          ":param state: state\n"
          ":param nu: dimension of control vector\n"
          ":param nr: dimension of cost-residual vector)"))
      .def("calc", pure_virtual(&DifferentialActionModelAbstract_wrap::calc), bp::args(" self", " data", " x", " u"),
           "Compute the state evolution and cost value.\n\n"
           "First, it describes the time-continuous evolution of our dynamical system\n"
           "in which along predefined integrated action self we might obtain the\n"
           "next discrete state. Indeed it computes the time derivatives of the\n"
           "state from a predefined dynamical system. Additionally it computes the\n"
           "cost value associated to this state and control pair.\n"
           ":param data: differential action data\n"
           ":param x: state vector\n"
           ":param u: control input")
      .def("calcDiff", pure_virtual(&DifferentialActionModelAbstract_wrap::calcDiff),
           bp::args(" self", " data", " x", " u", " recalc=True"),
           "Compute the derivatives of the dynamics and cost functions.\n\n"
           "It computes the partial derivatives of the dynamical system and the cost\n"
           "function. If recalc == True, it first updates the state evolution and\n"
           "cost value. This function builds a quadratic approximation of the\n"
           "time-continuous action model (i.e. dynamical system and cost function).\n"
           ":param data: differential action data\n"
           ":param x: state vector\n"
           ":param u: control input\n"
           ":param recalc: If true, it updates the state evolution and the cost value.")
      .def("createData", &DifferentialActionModelAbstract_wrap::createData, bp::args(" self"),
           "Create the differential action data.\n\n"
           "Each differential action model has its own data that needs to be\n"
           "allocated. This function returns the allocated data for a predefined\n"
           "DAM.\n"
           ":return DAM data.")
      .add_property("nu",
                    bp::make_function(&DifferentialActionModelAbstract_wrap::get_nu,
                                      bp::return_value_policy<bp::return_by_value>()),
                    "dimension of control vector")
      .add_property("nr",
                    bp::make_function(&DifferentialActionModelAbstract_wrap::get_nr,
                                      bp::return_value_policy<bp::return_by_value>()),
                    "dimension of cost-residual vector")
      .add_property("state",
                    bp::make_function(&DifferentialActionModelAbstract_wrap::get_state,
                                      bp::return_value_policy<bp::return_by_value>()),
                    "state")
      .add_property("has_control_limits",
                    bp::make_function(&DifferentialActionModelAbstract_wrap::get_has_control_limits,
                                      bp::return_value_policy<bp::return_by_value>()),
                    "indicates whether problem has finite control limits")
      .add_property("u_lb",
                    bp::make_function(&DifferentialActionModelAbstract_wrap::get_u_lb,
                                      bp::return_value_policy<bp::return_by_value>()),
                    &DifferentialActionModelAbstract_wrap::set_u_lb, "lower control limits")
      .add_property("u_ub",
                    bp::make_function(&DifferentialActionModelAbstract_wrap::get_u_ub,
                                      bp::return_value_policy<bp::return_by_value>()),
                    &DifferentialActionModelAbstract_wrap::set_u_ub, "upper control limits");

  bp::register_ptr_to_python<boost::shared_ptr<DifferentialActionDataAbstract> >();

  bp::class_<DifferentialActionDataAbstract, boost::noncopyable>(
      "DifferentialActionDataAbstract",
      "Abstract class for differential action datas.\n\n"
      "In crocoddyl, an action data contains all the required information for processing an\n"
      "user-defined action model. The action data typically is allocated onces by running\n"
      "model.createData() and contains the first- and second- order derivatives of the dynamics\n"
      "and cost function, respectively.",
      bp::init<DifferentialActionModelAbstract*>(
          bp::args(" self", " model"),
          "Create common data shared between DAMs.\n\n"
          "The differential action data uses the model in order to first process it.\n"
          ":param model: differential action model"))
      .add_property(
          "cost",
          bp::make_getter(&DifferentialActionDataAbstract::cost, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&DifferentialActionDataAbstract::cost), "cost value")
      .add_property(
          "xout",
          bp::make_getter(&DifferentialActionDataAbstract::xout, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&DifferentialActionDataAbstract::xout), "evolution state")
      .add_property(
          "r", bp::make_getter(&DifferentialActionDataAbstract::r, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&DifferentialActionDataAbstract::r), "cost residual")
      .add_property(
          "Fx", bp::make_getter(&DifferentialActionDataAbstract::Fx, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&DifferentialActionDataAbstract::Fx), "Jacobian of the dynamics")
      .add_property(
          "Fu", bp::make_getter(&DifferentialActionDataAbstract::Fu, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&DifferentialActionDataAbstract::Fu), "Jacobian of the dynamics")
      .add_property(
          "Lx", bp::make_getter(&DifferentialActionDataAbstract::Lx, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&DifferentialActionDataAbstract::Lx), "Jacobian of the cost")
      .add_property(
          "Lu", bp::make_getter(&DifferentialActionDataAbstract::Lu, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&DifferentialActionDataAbstract::Lu), "Jacobian of the cost")
      .add_property(
          "Lxx", bp::make_getter(&DifferentialActionDataAbstract::Lxx, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&DifferentialActionDataAbstract::Lxx), "Hessian of the cost")
      .add_property(
          "Lxu", bp::make_getter(&DifferentialActionDataAbstract::Lxu, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&DifferentialActionDataAbstract::Lxu), "Hessian of the cost")
      .add_property(
          "Luu", bp::make_getter(&DifferentialActionDataAbstract::Luu, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&DifferentialActionDataAbstract::Luu), "Hessian of the cost");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_DIFF_ACTION_BASE_HPP_
