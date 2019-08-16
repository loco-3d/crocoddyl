///////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (C) 2018-2019, LAAS-CNRS
// Copyright note valid unless otherwise stated in individual files.
// All rights reserved.
///////////////////////////////////////////////////////////////////////////////

#ifndef BINDINGS_PYTHON_CROCODDYL_CORE_ACTION_BASE_HPP_
#define BINDINGS_PYTHON_CROCODDYL_CORE_ACTION_BASE_HPP_

#include "crocoddyl/core/action-base.hpp"

namespace crocoddyl {
namespace python {

namespace bp = boost::python;

class ActionModelAbstract_wrap : public ActionModelAbstract, public bp::wrapper<ActionModelAbstract> {
 public:
  ActionModelAbstract_wrap(StateAbstract& state, unsigned int const& nu, unsigned int const& nr = 1)
      : ActionModelAbstract(state, nu, nr), bp::wrapper<ActionModelAbstract>() {}

  void calc(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
            const Eigen::Ref<const Eigen::VectorXd>& u) {
    assert(x.size() == state_.get_nx() && "ActionModelAbstract::calc: x has wrong dimension");
    assert(u.size() == nu_ && "ActionModelAbstract::calc: u has wrong dimension");
    return bp::call<void>(this->get_override("calc").ptr(), data, (Eigen::VectorXd)x, (Eigen::VectorXd)u);
  }

  void calcDiff(const boost::shared_ptr<ActionDataAbstract>& data, const Eigen::Ref<const Eigen::VectorXd>& x,
                const Eigen::Ref<const Eigen::VectorXd>& u, const bool& recalc = true) {
    assert(x.size() == state_.get_nx() && "ActionModelAbstract::calcDiff: x has wrong dimension");
    assert(u.size() == nu_ && "ActionModelAbstract::calcDiff: u has wrong dimension");
    return bp::call<void>(this->get_override("calcDiff").ptr(), data, (Eigen::VectorXd)x, (Eigen::VectorXd)u, recalc);
  }
};

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(ActionModel_calc_wraps, ActionModelAbstract::calc_wrap, 2, 3)

void exposeActionAbstract() {
  bp::class_<ActionModelAbstract_wrap, boost::noncopyable>(
      "ActionModelAbstract",
      "Abstract class for action models.\n\n"
      "In crocoddyl, an action model combines dynamics and cost data. Each node, in our optimal\n"
      "control problem, is described through an action model. Every time that we want to describe\n"
      "a problem, we need to provide ways of computing the dynamics, cost functions and their\n"
      "derivatives. These computations are mainly carry on inside calc() and calcDiff(),\n"
      "respectively.",
      bp::init<StateAbstract&, int, bp::optional<int> >(
          bp::args(" self", " state", " nu", " nr=1"),
          "Initialize the action model.\n\n"
          ":param state: state description,\n"
          ":param nu: dimension of control vector,\n"
          ":param nr: dimension of the cost-residual vector")[bp::with_custodian_and_ward<1, 2>()])
      .def("calc", pure_virtual(&ActionModelAbstract_wrap::calc), bp::args(" self", " data", " x", " u"),
           "Compute the next state and cost value.\n\n"
           "It describes the time-discrete evolution of our dynamical system\n"
           "in which we obtain the next state. Additionally it computes the\n"
           "cost value associated to this discrete state and control pair.\n"
           ":param data: action data\n"
           ":param x: time-discrete state vector\n"
           ":param u: time-discrete control input")
      .def("calcDiff", pure_virtual(&ActionModelAbstract_wrap::calcDiff),
           bp::args(" self", " data", " x", " u", " recalc=True"),
           "Compute the derivatives of the dynamics and cost functions.\n\n"
           "It computes the partial derivatives of the dynamical system and the\n"
           "cost function. If recalc == True, it first updates the state evolution\n"
           "and cost value. This function builds a quadratic approximation of the\n"
           "action model (i.e. linear dynamics and quadratic cost).\n"
           ":param data: action data\n"
           ":param x: time-discrete state vector\n"
           ":param u: time-discrete control input\n"
           ":param recalc: If true, it updates the state evolution and the cost value.")
      .def("createData", &ActionModelAbstract_wrap::createData, bp::args(" self"),
           "Create the action data.\n\n"
           "Each action model (AM) has its own data that needs to be allocated.\n"
           "This function returns the allocated data for a predefined AM.\n"
           ":return AM data.")
      .add_property(
          "nu", bp::make_function(&ActionModelAbstract_wrap::get_nu, bp::return_value_policy<bp::return_by_value>()),
          "dimension of control vector")
      .add_property(
          "nr", bp::make_function(&ActionModelAbstract_wrap::get_nr, bp::return_value_policy<bp::return_by_value>()),
          "dimension of cost-residual vector")
      .add_property("State",
                    bp::make_function(&ActionModelAbstract_wrap::get_state, bp::return_internal_reference<>()),
                    "state");

  bp::register_ptr_to_python<boost::shared_ptr<ActionDataAbstract> >();

  bp::class_<ActionDataAbstract, boost::noncopyable>(
      "ActionDataAbstract",
      "Abstract class for action datas.\n\n"
      "In crocoddyl, an action data contains all the required information for processing an\n"
      "user-defined action model. The action data typically is allocated onces by running\n"
      "model.createData() and contains the first- and second- order derivatives of the dynamics\n"
      "and cost function, respectively.",
      bp::init<ActionModelAbstract*>(bp::args(" self", " model"),
                                     "Create common data shared between AMs.\n\n"
                                     "The action data uses the model in order to first process it.\n"
                                     ":param model: action model"))
      .add_property("cost", bp::make_getter(&ActionDataAbstract::cost, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActionDataAbstract::cost), "cost value")
      .add_property("xnext",
                    bp::make_getter(&ActionDataAbstract::xnext, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActionDataAbstract::xnext), "next state")
      .add_property("Fx", bp::make_getter(&ActionDataAbstract::Fx, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActionDataAbstract::Fx), "Jacobian of the dynamics")
      .add_property("Fu", bp::make_getter(&ActionDataAbstract::Fu, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActionDataAbstract::Fu), "Jacobian of the dynamics")
      .add_property("Lx", bp::make_getter(&ActionDataAbstract::Lx, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActionDataAbstract::Lx), "Jacobian of the cost")
      .add_property("Lu", bp::make_getter(&ActionDataAbstract::Lu, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActionDataAbstract::Lu), "Jacobian of the cost")
      .add_property("Lxx", bp::make_getter(&ActionDataAbstract::Lxx, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActionDataAbstract::Lxx), "Hessian of the cost")
      .add_property("Lxu", bp::make_getter(&ActionDataAbstract::Lxu, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActionDataAbstract::Lxu), "Hessian of the cost")
      .add_property("Luu", bp::make_getter(&ActionDataAbstract::Luu, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActionDataAbstract::Luu), "Hessian of the cost")
      .add_property("r",
                    bp::make_getter(&ActionDataAbstract::r, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActionDataAbstract::r), "cost residual")
      .add_property("Rx", bp::make_getter(&ActionDataAbstract::Rx, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActionDataAbstract::Rx), "Jacobian of the cost residual")
      .add_property("Ru", bp::make_getter(&ActionDataAbstract::Ru, bp::return_value_policy<bp::return_by_value>()),
                    bp::make_setter(&ActionDataAbstract::Ru), "Jacobian of the cost residual");
}

}  // namespace python
}  // namespace crocoddyl

#endif  // BINDINGS_PYTHON_CROCODDYL_CORE_ACTION_BASE_HPP_
