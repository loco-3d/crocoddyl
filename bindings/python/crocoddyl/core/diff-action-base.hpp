///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef PYTHON_CROCODDYL_CORE_DIFF_ACTION_BASE_HPP_
#define PYTHON_CROCODDYL_CORE_DIFF_ACTION_BASE_HPP_

#include "crocoddyl/core/diff-action-base.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

class DifferentialActionModelAbstract_wrap : public DifferentialActionModelAbstract,
                                             public bp::wrapper<DifferentialActionModelAbstract> {
 public:
  DifferentialActionModelAbstract_wrap(StateAbstract& state, int nu, int nr = 1)
      : DifferentialActionModelAbstract(state, nu, nr), bp::wrapper<DifferentialActionModelAbstract>() {}

  void calc(const boost::shared_ptr<DifferentialActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u) {
    assert(x.size() == state_.get_nx() && "DifferentialActionModelAbstract::calc: x has wrong dimension");
    assert(u.size() == nu_ && "DifferentialActionModelAbstract::calc: u has wrong dimension");
    return bp::call<void>(this->get_override("calc").ptr(), data, (Eigen::VectorXd)x, (Eigen::VectorXd)u);
  }

  void calcDiff(const boost::shared_ptr<DifferentialActionDataAbstract>& data,
                const Eigen::Ref<const Eigen::VectorXd>& x, const Eigen::Ref<const Eigen::VectorXd>& u,
                const bool& recalc = true) {
    assert(x.size() == state_.get_nx() && "DifferentialActionModelAbstract::calcDiff: x has wrong dimension");
    assert(u.size() == nu_ && "DifferentialActionModelAbstract::calcDiff: u has wrong dimension");
    return bp::call<void>(this->get_override("calcDiff").ptr(), data, (Eigen::VectorXd)x, (Eigen::VectorXd)u, recalc);
  }

  boost::shared_ptr<DifferentialActionDataAbstract> createData() {
    return boost::make_shared<DifferentialActionDataAbstract>(this);
  }
};

void exposeDifferentialActionAbstract() {
  bp::class_<DifferentialActionModelAbstract_wrap, boost::noncopyable>(
      "DifferentialActionModelAbstract",
      "Abstract class for the differential action model.\n\n"
      "In crocoddyl, a differential action model combines dynamics and cost data described in\n"
      "continuous time. Each node, in our optimal control problem, is described through an\n"
      "action model. Every time that we want describe a problem, we need to provide ways of\n"
      "computing the dynamics, cost functions and their derivatives. These computations are\n"
      "mainly carry on inside calc() and calcDiff(), respectively.",
      bp::init<StateAbstract&, int, bp::optional<int> >(
          bp::args(" self", " state", " nu", " nr=1"),
          "Initialize the differential action model.\n\n"
          ":param state: state\n"
          ":param nu: dimension of control vector\n"
          ":param nr: dimension of cost-residual vector)")[bp::with_custodian_and_ward<1, 2>()])
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
      .add_property("State",
                    bp::make_function(&DifferentialActionModelAbstract_wrap::get_state,
                                      bp::return_internal_reference<>()),
                    "state");

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
      .def("shareCostMemory", &DifferentialActionDataAbstract::shareCostMemory, bp::args(" self", " cost"),
           "Share memory with a give cost\n\n"
           ":param cost: cost in which we want to share memory")
      .add_property(
          "cost",
          bp::make_getter(&DifferentialActionDataAbstract::cost, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&DifferentialActionDataAbstract::cost), "cost value")
      .add_property(
          "xout",
          bp::make_getter(&DifferentialActionDataAbstract::xout, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&DifferentialActionDataAbstract::xout), "evolution state")
      .add_property(
          "Fx", bp::make_getter(&DifferentialActionDataAbstract::Fx, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&DifferentialActionDataAbstract::Fx), "Jacobian of the dynamics")
      .add_property(
          "Fu", bp::make_getter(&DifferentialActionDataAbstract::Fu, bp::return_value_policy<bp::return_by_value>()),
          bp::make_setter(&DifferentialActionDataAbstract::Fu), "Jacobian of the dynamics")
      .add_property(
          "Lx",
          bp::make_function(&DifferentialActionDataAbstract::get_Lx, bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&DifferentialActionDataAbstract::set_Lx), "Jacobian of the cost")
      .add_property(
          "Lu",
          bp::make_function(&DifferentialActionDataAbstract::get_Lu, bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&DifferentialActionDataAbstract::set_Lu), "Jacobian of the cost")
      .add_property(
          "Lxx",
          bp::make_function(&DifferentialActionDataAbstract::get_Lxx, bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&DifferentialActionDataAbstract::set_Lxx), "Hessian of the cost")
      .add_property(
          "Lxu",
          bp::make_function(&DifferentialActionDataAbstract::get_Lxu, bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&DifferentialActionDataAbstract::set_Lxu), "Hessian of the cost")
      .add_property(
          "Luu",
          bp::make_function(&DifferentialActionDataAbstract::get_Luu, bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&DifferentialActionDataAbstract::set_Luu), "Hessian of the cost")
      .add_property(
          "costResiduals",
          bp::make_function(&DifferentialActionDataAbstract::get_r, bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&DifferentialActionDataAbstract::set_r), "cost residual")
      .add_property(
          "Rx",
          bp::make_function(&DifferentialActionDataAbstract::get_Rx, bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&DifferentialActionDataAbstract::set_Rx), "Jacobian of the cost residual")
      .add_property(
          "Ru",
          bp::make_function(&DifferentialActionDataAbstract::get_Ru, bp::return_value_policy<bp::return_by_value>()),
          bp::make_function(&DifferentialActionDataAbstract::set_Ru), "Jacobian of the cost residual");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // PYTHON_CROCODDYL_CORE_DIFF_ACTION_BASE_HPP_